import json
from pathlib import Path
import requests
# pyrefly: ignore[missing-import]
from datasets import load_dataset

DATA_DIR = Path("../data")
PDF_DIR = DATA_DIR / "raw"
PDF_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_PDF_BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"

def download_financebench_subset(max_docs: int = 5):
    print("Fetching FinanceBench metadata from Hugging Face...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    
    doc_to_rows = {}
    for row in ds:
        doc_name = row["doc_name"]
        doc_to_rows.setdefault(doc_name, []).append(row)
    
    sorted_docs = sorted(doc_to_rows.items(), key=lambda x: len(x[1]), reverse=True)
    
    selected_eval_rows = []
    downloaded_docs = 0

    for doc_name, rows in sorted_docs:
        if downloaded_docs >= max_docs:
            break
            
        pdf_filename = f"{doc_name}.pdf"
        pdf_path = PDF_DIR / pdf_filename
        github_url = f"{GITHUB_PDF_BASE_URL}/{pdf_filename}"
        
        if pdf_path.exists():
            print(f"[SKIP] {pdf_filename} already exists locally.")
            downloaded_docs += 1
            selected_eval_rows.extend(rows)
            continue

        print(f"Downloading {pdf_filename} ({len(rows)} Q&A pairs) from GitHub mirror...")
        response = requests.get(github_url, timeout=60)
        
        if response.status_code == 200 and response.content[:4] == b"%PDF":
            pdf_path.write_bytes(response.content)
            print(f"  -> Saved to {pdf_path} ({len(response.content) // 1024} KB)")
            downloaded_docs += 1
            selected_eval_rows.extend(rows)
        else:
            print(f"  -> [WARN] Could not fetch {pdf_filename} (Status {response.status_code}), trying next...")

    # Save the matching ground-truth Q&A evaluation set
    eval_records = []
    for row in selected_eval_rows:
        # Extract evidence text & page numbers safely
        evidence_list = row.get("evidence") or []
        evidence_pages = [ev.get("evidence_page_num") for ev in evidence_list if isinstance(ev, dict)]
        evidence_texts = [ev.get("evidence_text") for ev in evidence_list if isinstance(ev, dict)]
        
        eval_records.append({
            "financebench_id": row.get("financebench_id"),
            "company": row.get("company"),
            "doc_name": row.get("doc_name"),
            "question_type": row.get("question_type"),
            "question": row.get("question"),
            "ground_truth_answer": row.get("answer"),
            "justification": row.get("justification"),
            "evidence_pages": evidence_pages,
            "evidence_texts": evidence_texts,
        })

    eval_file = DATA_DIR / "eval_dataset.json"
    eval_file.write_text(json.dumps(eval_records, indent=2), encoding="utf-8")
    
    print("\n=== Download Complete ===")
    print(f"Total PDFs ready in {PDF_DIR}: {downloaded_docs}")
    print(f"Total Ground-Truth Q&A pairs saved to {eval_file}: {len(eval_records)}")

if __name__ == "__main__":
    download_financebench_subset(max_docs=5)