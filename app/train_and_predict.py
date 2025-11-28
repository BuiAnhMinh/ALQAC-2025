# train_and_predict.py
import json
from pathlib import Path

from retrieval import (
    train_data,
    test_data,
    train_question_embeddings,
    test_question_embeddings,
    macro_f2_bm25_topk,
    macro_f2_rerank,
    build_logreg_model,
    build_predictions_for_questions_with_scores,
)

def main():
    # 1) BM25 baselines
    bm25_f2_top1 = macro_f2_bm25_topk(k=1, beta=2.0, verbose=False)
    bm25_f2_top3 = macro_f2_bm25_topk(k=3, beta=2.0, verbose=False)

    # 2) Train LogReg reranker
    logreg_model = build_logreg_model(top_k_lexical=200)

    # 3) Rerank macro-F2 (LogReg)
    rerank_f2_top1 = macro_f2_rerank(
        top_k_lexical=200,
        top_k_final=1,
        alpha=0.4,
        beta=2.0,
        verbose=False,
        logreg_model=logreg_model,
    )

    print("\n=== Macro-F2 summary (all questions) ===")
    print(f"BM25-only @ top-1        : {bm25_f2_top1:.4f}")
    print(f"BM25-only @ top-3        : {bm25_f2_top3:.4f}")
    print(f"BM25+Embeddings (LogReg) : {rerank_f2_top1:.4f}")
    print()

    # 4) Build predictions file for the test questions
    test_predictions_with_scores = build_predictions_for_questions_with_scores(
        test_data,
        test_question_embeddings,
        top_k_lexical=200,
        top_k_final=1,
        alpha=0.4,
        logreg_model=logreg_model,
    )

    out_path = Path("alqac25_test_predictions.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(test_predictions_with_scores, f, ensure_ascii=False, indent=2)

    print("Saved predictions with scores to", out_path)


if __name__ == "__main__":
    main()
