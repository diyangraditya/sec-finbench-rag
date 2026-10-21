import json
import logging
import time
from typing import Any, Literal
from google import genai
from google.genai import types
from groq import Groq
from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Comprehensive and accurate answer to the question based strictly on the retrieved context."
    )
    cited_pages: list[int] = Field(
        default_factory=list,
        description="List of exact page numbers where the evidence was found."
    )
    cited_doc_names: list[str] = Field(
        default_factory=list,
        description="List of document names cited in the answer."
    )
    is_grounded: bool = Field(
        ...,
        description="True if the answer is directly supported by the context passages; False if evidence is insufficient."
    )
    provider_used: str = Field(
        ...,
        description="The LLM provider that generated the response ('gemini' or 'groq') or 'abstention_guardrail'."
    )


class RAGGenerator:
    """
    Production-grade RAG Generator with:
    - Abstention guardrail checking RERANK_SCORE_THRESHOLD.
    - Primary generation via Gemini (gemini-2.5-flash / fallback models).
    - Automatic fallback to Groq on 429 rate limits or transient errors.
    - Strict Pydantic structured output validation.
    """

    def __init__(self):
        self._gemini_client: genai.Client | None = None
        self._groq_client: Groq | None = None

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._gemini_client

    def _get_groq_client(self) -> Groq:
        if self._groq_client is None:
            self._groq_client = Groq(api_key=settings.GROQ_API_KEY)
        return self._groq_client

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        lines = []
        for i, c in enumerate(chunks, start=1):
            doc = c.doc_name or "Unknown_Doc"
            page = c.page_number
            score = f"{c.score:.3f}" if c.score is not None else "N/A"
            lines.append(f"--- Context Passage [{i}] (Doc: {doc} | Page: {page} | Score: {score}) ---")
            lines.append(c.text.strip())
            lines.append("")
        return "\n".join(lines)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a Senior Financial Analyst reviewing official SEC 10-K filings.
Answer the question based STRICTLY and ONLY on the provided context passages below.
Do not assume, hallucinate, or extrapolate facts that are not explicitly evidenced.

Context Passages:
{context}

Question:
{question}

Instructions:
1. Provide a direct, factually accurate answer incorporating the relevant financial figures, metrics, or quotes from the passages.
2. In 'cited_pages', extract the list of exact integer page numbers where your evidence was found.
3. In 'cited_doc_names', extract the document names where the evidence was found.
4. Set 'is_grounded' to true if and only if the answer is fully supported by the text.
5. If the context does not contain sufficient information to answer the question, state that clearly and set 'is_grounded' to false.
"""

    def _call_gemini(self, prompt: str) -> RAGResponse:
        client = self._get_gemini_client()
        models_to_try = [settings.GEMINI_MODEL] + settings.GEMINI_FALLBACK_MODELS

        last_error = None
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RAGResponse,
                        temperature=0.0,
                    ),
                )
                if response.text:
                    parsed = RAGResponse.model_validate_json(response.text)
                    parsed.provider_used = f"gemini ({model_name})"
                    return parsed
            except Exception as e:
                err_str = str(e).lower()
                logger.warning(f"Gemini call to {model_name} failed: {e}")
                last_error = e
                # If rate limit (429) or unavailable (503), break out to trigger Groq fallback
                if "429" in err_str or "quota" in err_str:
                    logger.info("Gemini hit rate-limit (429); triggering Groq fallback...")
                    raise e
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All Gemini model attempts failed.")

    def _call_groq(self, prompt: str) -> RAGResponse:
        client = self._get_groq_client()
        models_to_try = [settings.GROQ_MODEL] + settings.GROQ_FALLBACK_MODELS

        schema_json = json.dumps(RAGResponse.model_json_schema())
        system_msg = (
            "You are a Senior Financial Analyst and SEC 10-K specialist. "
            f"You MUST respond ONLY with a valid JSON object matching this schema:\n{schema_json}"
        )

        last_error = None
        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content:
                    parsed = RAGResponse.model_validate_json(content)
                    parsed.provider_used = f"groq ({model_name})"
                    return parsed
            except Exception as e:
                logger.warning(f"Groq call to {model_name} failed: {e}")
                last_error = e
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All Groq model attempts failed.")

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        provider: Literal["gemini", "groq"] | None = None,
        enforce_abstention: bool = True,
    ) -> tuple[RAGResponse, float]:
        """
        Executes grounded generation with abstention guardrail and fallback handling.
        Returns: (RAGResponse, latency_in_seconds)
        """
        t0 = time.time()
        active_provider = provider or settings.LLM_PROVIDER

        # 1. Abstention Guardrail: Check top reranker score
        if enforce_abstention:
            if not chunks:
                latency = time.time() - t0
                return (
                    RAGResponse(
                        answer="No relevant documents could be retrieved to answer this question.",
                        cited_pages=[],
                        cited_doc_names=[],
                        is_grounded=False,
                        provider_used="abstention_guardrail",
                    ),
                    latency,
                )

            top_score = chunks[0].score
            if top_score < settings.RERANK_SCORE_THRESHOLD:
                logger.info(
                    f"Abstention triggered: Top reranker score {top_score:.3f} is below threshold {settings.RERANK_SCORE_THRESHOLD:.3f}"
                )
                latency = time.time() - t0
                return (
                    RAGResponse(
                        answer=(
                            f"I cannot answer this question based on the provided documents. "
                            f"The highest retrieval confidence score was {top_score:.2f}, which is below "
                            f"the required threshold of {settings.RERANK_SCORE_THRESHOLD:.2f}."
                        ),
                        cited_pages=[],
                        cited_doc_names=[],
                        is_grounded=False,
                        provider_used="abstention_guardrail",
                    ),
                    latency,
                )

        # 2. Build prompt with retrieved context
        context = self._format_context(chunks)
        prompt = self._build_prompt(question, context)

        # 3. Call LLM with automatic fallback
        response: RAGResponse | None = None
        if active_provider == "groq":
            try:
                response = self._call_groq(prompt)
            except Exception as e:
                logger.warning(f"Groq failed, attempting Gemini as secondary: {e}")
                response = self._call_gemini(prompt)
        else:
            try:
                response = self._call_gemini(prompt)
            except Exception as e:
                logger.warning(f"Gemini failed ({e}), falling back automatically to Groq...")
                response = self._call_groq(prompt)

        latency = time.time() - t0
        return response, latency
