from collections import Counter
from app.data_loader import load_law_documents

docs = load_law_documents()
print(f"Total docs loaded: {len(docs)}")

docs_zalo = [
    d for d in docs
    if d.get("source") == "zalo"
    and not d.get("is_amending_article", False)
]
print("BM25 docs count (same as lexical_retrieval):", len(docs_zalo))

bm25_law_ids = {d["law_id"] for d in docs_zalo}
print("BM25 law_ids (sample):", list(bm25_law_ids)[:20])