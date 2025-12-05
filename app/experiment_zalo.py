"""
Experiment harness to compare retrieval methods on Zalo Q&A:

Methods:
  - bm25
  - embedding
  - hybrid

Metrics:
  - per-question F_beta (default beta=2)
  - macro-F_beta over all questions
"""

from typing import Dict, Set, Tuple, List

from app import retrieval
from app.data_loader import load_zalo_questions
from app.embedding import DEFAULT_MODEL_KEY

# Limit number of questions evaluated (set to None or a large number for all).
MAX_QUESTIONS: int = 50

def fbeta_for_sets(
    gold: Set[Tuple[str, str]],
    pred: Set[Tuple[str, str]],
    beta: float = 2.0,
) -> float:
    beta2 = beta ** 2

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 and recall == 0:
        return 0.0

    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def debug_one_question():
    questions = load_zalo_questions()
    q = questions[0]

    print("DEBUG QUESTION:")
    print("id:", q.get("question_id"))
    print("text:", q["text"])
    gold_articles = {(a["law_id"], a["article_id"]) for a in q["relevant_articles"]}
    print("GOLD:", gold_articles)

    preds = retrieval.retrieve_embedding_only(
        q["text"],
        top_k=5,
        model_key=DEFAULT_MODEL_KEY,
    )
    pred_articles = {(p["law_id"], p["article_id"]) for p in preds}
    print("EMBEDDING PRED:", pred_articles)

def eval_retriever_on_zalo(
    method: str,
    model_key: str | None = None,
    top_k: int = 10,
    beta: float = 2.0,
) -> None:
    """
    method: 'bm25' | 'embedding' | 'hybrid'
    model_key: required for embedding/hybrid, ignored for bm25
    """
    questions = load_zalo_questions()  # adapt if needed
    questions = questions[:MAX_QUESTIONS]
    scores: List[float] = []

    for q in questions:
        qtext = q["text"]
        gold_articles = {
            (a["law_id"], a["article_id"])
            for a in q["relevant_articles"]
        }

        if method == "bm25":
            preds = retrieval.retrieve_bm25(qtext, top_k=top_k)
        elif method == "embedding":
            preds = retrieval.retrieve_embedding_only(
                qtext,
                top_k=top_k,
                model_key=model_key or DEFAULT_MODEL_KEY,
            )
        elif method == "hybrid":
            preds = retrieval.retrieve_hybrid(
                qtext,
                semantic_top_k=200,
                lexical_top_k=top_k,
                model_key=model_key or DEFAULT_MODEL_KEY,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        pred_articles = {
            (p["law_id"], p["article_id"])
            for p in preds
        }

        f = fbeta_for_sets(gold_articles, pred_articles, beta=beta)
        scores.append(f)

    macro_f = sum(scores) / len(scores) if scores else 0.0
    print(
        f"Method={method}, model_key={model_key}, top_k={top_k}, "
        f"beta={beta}: macro-F={macro_f:.4f} over {len(scores)} questions"
    )


def main() -> None:
    # Simple default grid of experiments
    methods = ["bm25", "embedding", "hybrid"]
    model_keys = [DEFAULT_MODEL_KEY]  # adapt to your registry
    top_k_values = [3, 5, 10]
    
    debug_one_question()

    for method in methods:
        if method == "bm25":
            for k in top_k_values:
                eval_retriever_on_zalo(
                    method=method,
                    model_key=None,
                    top_k=k,
                    beta=2.0,
                )
        else:
            for mk in model_keys:
                for k in top_k_values:
                    eval_retriever_on_zalo(
                        method=method,
                        model_key=mk,
                        top_k=k,
                        beta=2.0,
                    )


if __name__ == "__main__":
    main()
