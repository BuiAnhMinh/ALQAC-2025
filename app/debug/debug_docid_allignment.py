import json
from app import retrieval
from app.semantic_retrieval import LAW_ART_TO_DOCID

def main():
    # Same file as experiment_zalo
    with open("data/zalo_question.json", "r", encoding="utf-8") as f:
        zalo_questions = json.load(f)

    for i, q in enumerate(zalo_questions[:5]):
        qid = q["id"]
        qtext = q["text"]
        gold_articles = q["relevant_articles"]

        # GOLD (law_id, article_id) pairs
        gold_pairs = {(ra["law_id"], ra["article_id"]) for ra in gold_articles}

        # GOLD doc_ids via LAW_ART_TO_DOCID
        gold_doc_ids = {LAW_ART_TO_DOCID.get(pair) for pair in gold_pairs}
        # keep Nones separate to see mapping problems
        gold_doc_ids_none = {pair for pair in gold_pairs if LAW_ART_TO_DOCID.get(pair) is None}

        bm25_results = retrieval.retrieve_bm25(
            question_text=qtext,
            top_k=10,
        )
        bm25_pairs = {(r["law_id"], r["article_id"]) for r in bm25_results}
        bm25_doc_ids = {r["doc_id"] for r in bm25_results}

        print("=" * 60)
        print(f"Question {i} | ID={qid}")
        print("TEXT:", qtext)
        print("GOLD pairs:", gold_pairs)
        print("GOLD doc_ids:", gold_doc_ids)
        print("GOLD pairs with no doc_id:", gold_doc_ids_none)
        print("BM25 pairs:", bm25_pairs)
        print("BM25 doc_ids:", bm25_doc_ids)
        print("doc_id ∩:", gold_doc_ids & bm25_doc_ids)

if __name__ == "__main__":
    main()
