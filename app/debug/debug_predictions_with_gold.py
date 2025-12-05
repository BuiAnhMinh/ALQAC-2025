import json
from app import retrieval

def main():
    # Same file you used before
    with open("data/zalo_question.json", "r", encoding="utf-8") as f:
        zalo_questions = json.load(f)

    for i, q in enumerate(zalo_questions[:5]):  # first 5 questions
        qid = q["id"]                    # <-- use 'id'
        qtext = q["text"]                # <-- use 'text'
        gold_articles = q["relevant_articles"]  # <-- use 'relevant_articles'

        gold_pairs = {(ra["law_id"], ra["article_id"]) for ra in gold_articles}

        # BM25-only predictions
        bm25_results = retrieval.retrieve_bm25(
            question_text=qtext,
            top_k=10,
        )
        bm25_pairs = {(r["law_id"], r["article_id"]) for r in bm25_results}

        # Embedding-only predictions (will be empty for now because DB embeddings table is empty)
        try:
            emb_results = retrieval.retrieve_embedding_only(
                question_text=qtext,
                model_key="openai_small",
                top_k=10,
            )
            emb_pairs = {(r["law_id"], r["article_id"]) for r in emb_results}
        except Exception as e:
            emb_pairs = set()
            print(f"[WARN] embedding retrieval failed for {qid}: {e}")

        print("=" * 60)
        print(f"Question {i} | ID={qid}")
        print("TEXT:", qtext)
        print("GOLD:", gold_pairs)
        print("BM25:", bm25_pairs)
        print("BM25 ∩ GOLD:", bm25_pairs & gold_pairs)
        print("EMBED:", emb_pairs)
        print("EMBED ∩ GOLD:", emb_pairs & gold_pairs)

if __name__ == "__main__":
    main()
