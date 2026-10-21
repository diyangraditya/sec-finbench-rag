import json
import logging
import re
from pathlib import Path
from typing import Any
import fitz
from pydantic import BaseModel, Field
import pymupdf4llm

from src.config import settings

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: str
    doc_name: str
    page_number: int  # 0-indexed to match FinanceBench ground truth
    page_num_1based: int  # 1-indexed for human readability
    chunk_index: int
    chunk_mode: str  # "naive" | "contextual"
    text: str  # embedded text (with context header if contextual, without if naive)
    raw_content: str  # text without context header
    metadata: dict[str, Any] = Field(default_factory=dict)


def parse_pdf_to_pages(pdf_path: Path, use_cache: bool = True) -> list[dict[str, Any]]:
    """
    Parses a PDF file page-by-page.
    Extracts markdown with intact tables using pymupdf4llm and plain text via fitz.
    Caches parsed output to disk to avoid expensive re-parsing.
    """
    doc_name = pdf_path.stem
    cache_dir = settings.CACHE_DIR / "parsed_pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{doc_name}.json"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} cached pages for {doc_name}")
                return data
        except Exception as e:
            logger.warning(f"Failed to read cache for {doc_name}, re-parsing: {e}")

    logger.info(f"Parsing PDF: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_data: list[dict[str, Any]] = []

    # Fast batch extraction across the entire document
    md_chunks_by_page: dict[int, str] = {}
    try:
        all_md_chunks = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            use_ocr=False
        )
        for chunk in all_md_chunks:
            meta = chunk.get("metadata", {})
            # page_number from pymupdf4llm is 1-indexed
            p_1based = meta.get("page_number")
            if p_1based is not None:
                p_0indexed = p_1based - 1
                md_chunks_by_page[p_0indexed] = chunk.get("text", "").strip()
    except Exception as e:
        logger.warning(f"Batch pymupdf4llm failed for {doc_name}, falling back to per-page: {e}")

    for page_idx in range(total_pages):
        page = doc[page_idx]
        raw_text = page.get_text().strip()

        if page_idx in md_chunks_by_page:
            markdown_text = md_chunks_by_page[page_idx]
        else:
            try:
                single_chunk = pymupdf4llm.to_markdown(
                    doc,
                    pages=[page_idx],
                    page_chunks=True,
                    use_ocr=False
                )
                markdown_text = single_chunk[0].get("text", "").strip() if single_chunk else raw_text
            except Exception:
                markdown_text = raw_text

        pages_data.append({
            "doc_name": doc_name,
            "page_number": page_idx,  # 0-indexed
            "page_num_1based": page_idx + 1,  # 1-indexed
            "markdown_text": markdown_text or raw_text,
            "raw_text": raw_text,
        })

    # Save to disk cache
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(pages_data, f, ensure_ascii=False)
        logger.info(f"Cached {len(pages_data)} pages for {doc_name}")
    except Exception as e:
        logger.warning(f"Could not write cache for {doc_name}: {e}")

    return pages_data


def naive_chunk(
    pages_data: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 100
) -> list[Chunk]:
    """
    Naive chunking: Fixed character splitting (1000 chars, 100 overlap)
    WITHOUT metadata headers.
    """
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)

    for page in pages_data:
        doc_name = page["doc_name"]
        page_num = page["page_number"]
        page_1based = page["page_num_1based"]
        text = page["raw_text"] or page["markdown_text"]

        if not text:
            continue

        chunk_idx = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_slice = text[start:end].strip()

            if chunk_slice:
                chunk_id = f"{doc_name}_naive_p{page_num}_c{chunk_idx}"
                metadata = {
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "page_num_1based": page_1based,
                    "chunk_index": chunk_idx,
                    "chunk_mode": "naive",
                }
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_name=doc_name,
                        page_number=page_num,
                        page_num_1based=page_1based,
                        chunk_index=chunk_idx,
                        chunk_mode="naive",
                        text=chunk_slice,
                        raw_content=chunk_slice,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break
            start += step

    return chunks


def _split_markdown_into_blocks(text: str) -> list[str]:
    """
    Splits markdown text into cohesive blocks:
    - Tables (| ... |) are kept together as single atomic blocks.
    - Headers (# ...) start new sections.
    - Paragraphs are delimited by blank lines.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|")

        if is_table_line:
            if not in_table:
                # Flush previous non-table block if non-empty
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
                in_table = True
            current_block.append(line)
        else:
            if in_table:
                # End of table
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
                in_table = False

            # Check for header or paragraph break
            if stripped.startswith("#"):
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
                current_block.append(line)
            elif stripped == "":
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []
            else:
                current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return [b for b in blocks if b]


def contextual_markdown_chunk(
    pages_data: list[dict[str, Any]],
    max_chunk_chars: int = 1200
) -> list[Chunk]:
    """
    Contextual Markdown chunking:
    - Splits by Markdown sections while keeping tables intact.
    - Prepends a context header to every chunk string before embedding:
      "[Document: {doc_name} | Page: {page_num}]\n{chunk_text}"
    """
    chunks: list[Chunk] = []

    for page in pages_data:
        doc_name = page["doc_name"]
        page_num = page["page_number"]
        page_1based = page["page_num_1based"]
        md_text = page["markdown_text"]

        if not md_text:
            continue

        blocks = _split_markdown_into_blocks(md_text)
        current_chunk_blocks: list[str] = []
        current_len = 0
        chunk_idx = 0

        header = f"[Document: {doc_name} | Page: {page_num}]\n"

        for block in blocks:
            block_len = len(block)
            is_table = block.strip().startswith("|")

            # If adding this block exceeds max_chunk_chars and we already have content:
            if current_chunk_blocks and (current_len + block_len + 2 > max_chunk_chars):
                body = "\n\n".join(current_chunk_blocks).strip()
                chunk_id = f"{doc_name}_contextual_p{page_num}_c{chunk_idx}"
                metadata = {
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "page_num_1based": page_1based,
                    "chunk_index": chunk_idx,
                    "chunk_mode": "contextual",
                    "has_table": any(b.strip().startswith("|") for b in current_chunk_blocks),
                }
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_name=doc_name,
                        page_number=page_num,
                        page_num_1based=page_1based,
                        chunk_index=chunk_idx,
                        chunk_mode="contextual",
                        text=f"{header}{body}",
                        raw_content=body,
                        metadata=metadata,
                    )
                )
                chunk_idx += 1
                current_chunk_blocks = [block]
                current_len = block_len
            else:
                current_chunk_blocks.append(block)
                current_len += block_len + 2

        if current_chunk_blocks:
            body = "\n\n".join(current_chunk_blocks).strip()
            chunk_id = f"{doc_name}_contextual_p{page_num}_c{chunk_idx}"
            metadata = {
                "doc_name": doc_name,
                "page_number": page_num,
                "page_num_1based": page_1based,
                "chunk_index": chunk_idx,
                "chunk_mode": "contextual",
                "has_table": any(b.strip().startswith("|") for b in current_chunk_blocks),
            }
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_name=doc_name,
                    page_number=page_num,
                    page_num_1based=page_1based,
                    chunk_index=chunk_idx,
                    chunk_mode="contextual",
                    text=f"{header}{body}",
                    raw_content=body,
                    metadata=metadata,
                )
            )

    return chunks


def ingest_all_documents(
    raw_dir: Path | None = None
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Ingests all PDFs in raw_dir and returns:
    (naive_chunks, contextual_chunks)
    """
    directory = raw_dir or settings.RAW_DIR
    pdf_files = sorted(directory.glob("*.pdf"))

    all_naive_chunks: list[Chunk] = []
    all_contextual_chunks: list[Chunk] = []

    for pdf_path in pdf_files:
        pages_data = parse_pdf_to_pages(pdf_path)
        naive = naive_chunk(pages_data)
        contextual = contextual_markdown_chunk(pages_data)
        all_naive_chunks.extend(naive)
        all_contextual_chunks.extend(contextual)
        logger.info(
            f"Doc {pdf_path.name}: {len(pages_data)} pages -> "
            f"{len(naive)} naive chunks, {len(contextual)} contextual chunks"
        )

    return all_naive_chunks, all_contextual_chunks
