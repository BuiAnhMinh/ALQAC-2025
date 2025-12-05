import json
from collections import Counter

from app.data_loader import load_law_documents

def main():
    docs = load_law_documents()
    print(f"Total docs loaded: {len(docs)}")

    # This MUST match your BM25 filter in lexical_retrieval.py
    docs_zalo = [
        d for d in docs
        if d.get("source") == "zalo"
        and not d.get("is_amending_article", False)
    ]
    print("BM25 docs count (same as lexical_retrieval):", len(docs_zalo))

    bm25_law_ids = {d["law_id"] for d in docs_zalo}
    print("BM25 law_ids (sample):", list(bm25_law_ids)[:20])

    # ===== Load Zalo eval questions (adjust path to match experiment_zalo.py) =====
    with open("data/zalo_question.json", "r", encoding="utf-8") as f:
        zalo_questions = json.load(f)

    gold_pairs = set()
    for q in zalo_questions:
        for ra in q["relevant_articles"]:
            gold_pairs.add((ra["law_id"], ra["article_id"]))

    gold_law_ids = {law for (law, _) in gold_pairs}
    print("Distinct GOLD laws:", len(gold_law_ids))

    missing_laws_in_bm25 = gold_law_ids - bm25_law_ids
    extra_laws_in_bm25 = bm25_law_ids - gold_law_ids

    print("\nGold law_ids NOT in BM25 corpus:", missing_laws_in_bm25)
    print("BM25 corpus laws NOT in GOLD (sample):", list(extra_laws_in_bm25)[:20])

if __name__ == "__main__":
    main()
