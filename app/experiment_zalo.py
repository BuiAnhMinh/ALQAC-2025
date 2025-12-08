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
from pathlib import Path
import json

import numpy as np

from app import retrieval
from app.data_loader import load_zalo_questions
from app.embedding.embedding import DEFAULT_MODEL_KEY
from app.semantic_retrieval import semantic_retrieve_from_embedding
from app.lexical_retrieval import bm25_lexical_rerank

# Limit number of questions evaluated (set to None or a large number for all).
MAX_QUESTIONS: int = 10

# Precomputed Zalo question embeddings (built by zalo_embed_question.py)
DATA_DIR = Path("data")
ZALO_Q_EMB = np.load(DATA_DIR / "zalo_question_embeddings.npy")
with (DATA_DIR / "zalo_question_ids.json").open("r", encoding="utf-8") as f:
    ZALO_Q_IDS: List[str] = json.load(f)

QID2IDX: Dict[str, int] = {qid: i for i, qid in enumerate(ZALO_Q_IDS)}


def fbeta_for_sets(
    gold: Set[Tuple[str, str]],
    pred: Set[Tuple[str, str]],
    beta: float = 2.0,
) -> float:
    beta2 = beta ** 2

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    # No gold, no pred -> define F as 0
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def debug_one_question() -> None:
    """Quick sanity check on a single question and the embedding retriever."""
    questions = load_zalo_questions()
    q = questions[0]

    print("DEBUG QUESTION:")
    print("id:", q["id"])
    print("text:", q["text"])

    gold_articles = {(a["law_id"], a["article_id"]) for a in q["relevant_articles"]}
    print("GOLD:", gold_articles)

    qid = q["id"]
    q_idx = QID2IDX[qid]
    q_emb = ZALO_Q_EMB[q_idx]

    preds = semantic_retrieve_from_embedding(q_emb, top_k=5)
    pred_articles = {(p["law_id"], p["article_id"]) for p in preds}
    print("EMBEDDING PRED:", pred_articles)


def eval_retriever_on_zalo(
    method: str,
    model_key: str | None = None,
    top_k: int = 10,
    beta: float = 2.0,
) -> None:
    """
    Evaluate one retrieval method on a subset of Zalo questions.

    method: 'bm25' | 'embedding' | 'hybrid'
    model_key: kept for API compatibility, ignored in offline embedding mode.
    """
    questions = load_zalo_questions()
    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]

    scores: List[float] = []

    for q in questions:
        qtext = q["text"]
        qid = q["id"]
        gold_articles = {
            (a["law_id"], a["article_id"])
            for a in q["relevant_articles"]
        }

        if method == "bm25":
            # Use the BM25-only retriever we already debugged
            preds = retrieval.retrieve_bm25(
                question_text=qtext,
                top_k=top_k,
            )

        elif method == "embedding":
            # Use precomputed question embedding + pgvector
            q_idx = QID2IDX[qid]
            q_emb = ZALO_Q_EMB[q_idx]
            preds = semantic_retrieve_from_embedding(q_emb, top_k=top_k)

        elif method == "hybrid":
            # Hybrid (semantic then lexical) using precomputed question embeddings.
            # 1) Semantic candidates from pgvector
            q_idx = QID2IDX[qid]
            q_emb = ZALO_Q_EMB[q_idx]
            semantic_candidates = semantic_retrieve_from_embedding(
                q_emb,
                top_k=200,
            )
            # 2) BM25 rerank within these candidates
            preds = bm25_lexical_rerank(
                question_text=qtext,
                candidates=semantic_candidates,
                top_k=top_k,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        pred_articles = {(p["law_id"], p["article_id"]) for p in preds}

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
    model_keys = [DEFAULT_MODEL_KEY]  # kept for compatibility, unused offline
    top_k_values = [3, 5, 10]

    # Optional: quick one-question debug
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
