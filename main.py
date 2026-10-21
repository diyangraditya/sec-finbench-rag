import argparse
import sys
import uvicorn

from src.config import settings
from src.evaluation.evaluate import run_ablation_benchmark
from src.generation.generator import RAGGenerator
from src.ingest import run_full_ingestion
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    parser = argparse.ArgumentParser(
        description="FinanceBench Production RAG & Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest all SEC 10-K PDFs into Qdrant and BM25")

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run retrieval ablation benchmark on FinanceBench"
    )
    eval_parser.add_argument(
        "--use-doc-filter",
        action="store_true",
        help="Use metadata doc_name filter during retrieval",
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument(
        "--mode",
        choices=["baseline", "production"],
        default="production",
        help="Retrieval mode",
    )
    query_parser.add_argument(
        "--provider",
        choices=["gemini", "groq"],
        default=None,
        help="LLM provider override",
    )
    query_parser.add_argument(
        "--doc-name",
        type=str,
        default=None,
        help="Optional document filter",
    )

    # API command
    api_parser = subparsers.add_parser("api", help="Start the FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Streamlit command
    subparsers.add_parser("app", help="Launch the Streamlit web application")

    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Deep trace debug inspection across all RAG pipeline stages")
    debug_parser.add_argument("question", type=str, help="Question to inspect")
    debug_parser.add_argument("--doc-name", type=str, default=None, help="Optional document filter")
    debug_parser.add_argument("--provider", choices=["gemini", "groq"], default="gemini", help="LLM provider")

    args = parser.parse_args()

    if args.command == "ingest":
        run_full_ingestion()
    elif args.command == "evaluate":
        run_ablation_benchmark(use_doc_filter=args.use_doc_filter)
    elif args.command == "query":
        retriever = HybridRetriever()
        generator = RAGGenerator()
        chunks = retriever.search(
            query=args.question,
            mode=args.mode,
            doc_name=args.doc_name,
            limit=settings.TOP_K_RERANK,
        )
        resp, lat = generator.generate(
            question=args.question,
            chunks=chunks,
            provider=args.provider,
            enforce_abstention=(args.mode == "production"),
        )
        print("\n" + "=" * 60)
        print(f"QUESTION: {args.question}")
        print(f"MODE: {args.mode} | PROVIDER: {resp.provider_used} | LATENCY: {lat:.2f}s")
        print(f"GROUNDED: {resp.is_grounded} | CITED PAGES: {resp.cited_pages}")
        print("=" * 60)
        print(f"ANSWER:\n{resp.answer}")
        print("=" * 60)
    elif args.command == "api":
        print(f"Starting FastAPI server on http://{args.host}:{args.port}")
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)
    elif args.command == "app":
        import subprocess

        cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]
        print("Launching Streamlit UI...")
        subprocess.run(cmd)
    elif args.command == "debug":
        from src.debug_pipeline import debug_query
        debug_query(query=args.question, doc_name=args.doc_name, provider=args.provider)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
