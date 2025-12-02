import json
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from sklearn.linear_model import LogisticRegression

from app.config import (
    STOPWORDS_PATH,
    TRAIN_Q_EMB_PATH,
    TEST_Q_EMB_PATH,
    get_connection,
)
from app.data_loader import load_law_documents, load_train_data, load_test_data
from app.embedding import embed_text

# ---------- Load data ----------
law_documents = load_law_documents()
train_data = load_train_data()
test_data = load_test_data()

# Build mapping doc_id -> metadata (from JSON)
DOCID_TO_META: Dict[int, Dict[str, Any]] = {d["doc_id"]: d for d in law_documents}

# Reverse mapping (law_id, article_id) -> doc_id for DB results
LAW_ART_TO_DOCID: Dict[tuple[str, str], int] = {
    (d["law_id"], d["article_id"]): d["doc_id"] for d in law_documents
}

# ---------- Stopwords & tokenizer ----------
def load_stopwords(path: str) -> set[str]:
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


STOPWORDS = load_stopwords(str(STOPWORDS_PATH))
LEGAL_WHITELIST = {"phải", "không", "được", "cấm", "trừ", "khi", "nếu", "vì"}
STOPWORDS = STOPWORDS - LEGAL_WHITELIST


def underthesea_tokenizer(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    tokenized = word_tokenize(text, format="text")
    tokens = tokenized.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# ---------- Load question embeddings (still from .npy is fine) ----------
train_question_embeddings = np.load(TRAIN_Q_EMB_PATH)
print("Train question embeddings shape:", train_question_embeddings.shape)

test_question_embeddings = np.load(TEST_Q_EMB_PATH)
print("Test question embeddings shape:", test_question_embeddings.shape)


# ---------- Lexical + embedding retrieval via Postgres ----------
def bm25_lexical_retrieve(question_text: str, top_k: int = 50) -> List[Dict[str, Any]]:
    """
    Lexical retrieval handled by Postgres using the `tokens` (text[]) column
    and retrieves the article `embedding` from the DB as well.

    Returns:
    [
        {
            "doc_id": int,          # from JSON mapping
            "bm25_score": float,    # ts_rank_cd score (higher = better)
            "law_id": str,
            "article_id": str,
            "text": str,
            "embedding": List[float] or np.ndarray,
        },
        ...
    ]
    """
    # 1) Tokenize question exactly like articles
    q_tokens = underthesea_tokenizer(question_text)
    if not q_tokens:
        return []

    # Limit number of terms to avoid huge tsquery
    q_tokens = q_tokens[:32]

    # Build a tsquery string using OR (|) so docs with ANY of these terms match
    # Example: "hop_dong | boi_thuong | vi_pham"
    ts_query = " | ".join(q_tokens)

    conn = get_connection()
    cur = conn.cursor()
    try:
        # tokens is a text[] column, so convert to space-separated string
        # embedding is a vector column (pgvector)
        cur.execute(
            """
            SELECT law_id,
                   article_id,
                   text,
                   embedding,
                   ts_rank_cd(
                       to_tsvector('simple', array_to_string(tokens, ' ')),
                       to_tsquery('simple', %s)
                   ) AS rank
            FROM articles
            WHERE embedding IS NOT NULL
              AND to_tsvector('simple', array_to_string(tokens, ' '))
                  @@ to_tsquery('simple', %s)
            ORDER BY rank DESC
            LIMIT %s;
            """,
            (ts_query, ts_query, top_k),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[Dict[str, Any]] = []
    for law_id, article_id, text, emb, rank in rows:
        # Map back to doc_id used in law_documents (for consistency with rest of code)
        key = (law_id, article_id)
        doc_id = LAW_ART_TO_DOCID.get(key)
        if doc_id is None:
            # If something is out of sync between JSON and DB, skip that row
            continue

        results.append(
            {
                "doc_id": int(doc_id),
                "bm25_score": float(rank),  # higher = better
                "law_id": law_id,
                "article_id": article_id,
                "text": text,
                "embedding": emb,  # pgvector -> list-like; we convert later
            }
        )

    return results


# ---------- Feature computation ----------
def compute_candidate_features(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
):
    """
    For a given question (text + embedding), retrieve lexical candidates from DB,
    then compute:
      - bm25_norm: normalized lexical scores
      - cos_sim: cosine similarity between DB embeddings and question embedding
    """
    lexical_candidates = bm25_lexical_retrieve(
        question_text,
        top_k=top_k_lexical,
    )

    if not lexical_candidates:
        # Nothing retrieved
        return [], np.array([], dtype="float32"), np.array([], dtype="float32")

    # BM25 / ts_rank_cd scores
    bm25_scores = np.array(
        [c["bm25_score"] for c in lexical_candidates],
        dtype="float32",
    )

    # Normalize BM25 scores to [0, 1]
    max_score = bm25_scores.max()
    if max_score > 0:
        bm25_norm = bm25_scores / max_score
    else:
        bm25_norm = bm25_scores

    # Candidate embeddings from DB
    # emb from psycopg2/pgvector is list-like, convert to float32 array
    cand_embs_list = []
    for c in lexical_candidates:
        emb_raw = c["embedding"]

        # If it's a string like "[-0.01,0.02,...]", parse it as JSON
        if isinstance(emb_raw, str):
            # Remove whitespace just in case and parse
            emb_list = json.loads(emb_raw)
        # If it's already a list/tuple/ndarray, use it directly
        elif isinstance(emb_raw, (list, tuple, np.ndarray)):
            emb_list = emb_raw
        else:
            # Fallback: try to cast to list
            emb_list = list(emb_raw)

        emb_arr = np.array(emb_list, dtype="float32")
        cand_embs_list.append(emb_arr)

    cand_embs = np.stack(cand_embs_list, axis=0)

    # Cosine similarity
    q_emb = question_embedding.astype("float32")  # shape (d,)
    dot = cand_embs @ q_emb
    norms = np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(q_emb)
    norms = np.where(norms == 0, 1e-8, norms)  # avoid division by zero
    cos_sim = dot / norms  # shape (n_cand,)

    return lexical_candidates, bm25_norm, cos_sim


# ---------- Reranking ----------
def retrieve_and_rerank_with_qemb(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
):
    """
    Combined BM25 + embedding retrieval.

    If logreg_model is provided, we learn a combination of bm25 + cosine.
    Otherwise, we use a simple linear combination with weight alpha.
    """
    lexical_candidates, bm25_norm, cos_similarity = compute_candidate_features(
        question_text,
        question_embedding,
        top_k_lexical=top_k_lexical,
    )

    if len(lexical_candidates) == 0:
        return []

    # Feature matrix for reranking
    # X: [bm25_norm, cos_similarity]
    X = np.stack([bm25_norm, cos_similarity], axis=1)

    if logreg_model is not None:
        # Predict probabilities of being "relevant"
        probs = logreg_model.predict_proba(X)[:, 1]
        combined_scores = probs
    else:
        # Simple linear combination (normalized cos_sim)
        cos_scores = cos_similarity
        if cos_scores.size > 0:
            max_cos = np.max(cos_scores)
            if max_cos > 0:
                cos_norm = cos_scores / max_cos
            else:
                cos_norm = cos_scores
        else:
            cos_norm = cos_scores

        combined_scores = alpha * cos_norm + (1.0 - alpha) * bm25_norm

    # Pick top-k final
    top_idx = np.argsort(combined_scores)[::-1][:top_k_final]

    ranked = []
    for idx in top_idx:
        base = lexical_candidates[int(idx)].copy()
        base["embedding_score"] = float(cos_similarity[idx])
        base["combined_score"] = float(combined_scores[idx])
        ranked.append(base)

    return ranked


def retrieve_and_rerank(
    question_text: str,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
    use_train_embedding: bool = True,
    question_index: int | None = None,
):
    """
    Convenience wrapper:
      - optionally looks up precomputed question embedding from train or test arrays
      - then calls retrieve_and_rerank_with_qemb.
    """

    if question_index is None:
        raise ValueError("question_index must be provided")

    if use_train_embedding:
        q_emb = train_question_embeddings[question_index]
    else:
        q_emb = test_question_embeddings[question_index]

    return retrieve_and_rerank_with_qemb(
        question_text,
        q_emb,
        top_k_lexical=top_k_lexical,
        top_k_final=top_k_final,
        alpha=alpha,
        logreg_model=logreg_model,
    )


# ---------- Metrics: Macro-Fbeta ----------
def fbeta_for_sets(
    gold_articles: Set[Tuple[str, str]],
    pred_articles: Set[Tuple[str, str]],
    beta: float = 2.0,
) -> float:
    """
    F-beta for a single question, where gold_articles and pred_articles are sets of
    (law_id, article_id) pairs.
    """
    if not gold_articles and not pred_articles:
        return 1.0  # nothing to find and we predicted nothing

    tp = len(gold_articles & pred_articles)
    fp = len(pred_articles - gold_articles)
    fn = len(gold_articles - pred_articles)

    beta2 = beta * beta

    if tp == 0 and fp == 0 and fn == 0:
        return 1.0

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_fbeta_bm25_topk(
    data: List[Dict[str, Any]],
    top_k: int = 5,
    beta: float = 2.0,
    verbose: bool = True,
) -> float:
    """
    Evaluate plain BM25 (ts_rank_cd) retrieval using Macro-Fbeta over all questions in data.
    """
    f_scores = []

    for q in tqdm(data, desc=f"BM25@{top_k}"):
        question_text = q["text"]
        gold_articles = {
            (a["law_id"], a["article_id"]) for a in q["relevant_articles"]
        }

        bm25_results = bm25_lexical_retrieve(question_text, top_k=top_k)
        pred_articles = {(c["law_id"], c["article_id"]) for c in bm25_results}

        f_q = fbeta_for_sets(gold_articles, pred_articles, beta=beta)
        f_scores.append(f_q)

    macro_fbeta = float(np.mean(f_scores))
    if verbose:
        print(f"BM25 top-{top_k} macro-F{beta:.1f}: {macro_fbeta:.4f}")

    return macro_fbeta


def macro_fbeta_rerank(
    data: List[Dict[str, Any]],
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
    use_train_embedding: bool = True,
    beta: float = 2.0,
    verbose: bool = True,
) -> float:
    """
    Evaluate reranking pipeline with Macro-Fbeta over all questions in data.

    data: typically train_data or test_data
    """
    f_scores = []

    for i, q in enumerate(tqdm(data, desc="Rerank eval")):
        question_text = q["text"]
        gold_articles = {
            (a["law_id"], a["article_id"]) for a in q["relevant_articles"]
        }

        ranked = retrieve_and_rerank(
            question_text,
            top_k_lexical=top_k_lexical,
            top_k_final=top_k_final,
            alpha=alpha,
            logreg_model=logreg_model,
            use_train_embedding=use_train_embedding,
            question_index=i,
        )

        pred_articles = {(r["law_id"], r["article_id"]) for r in ranked}

        f_q = fbeta_for_sets(gold_articles, pred_articles, beta=beta)
        f_scores.append(f_q)

    macro_fbeta = float(np.mean(f_scores))
    if verbose:
        print(
            f"Rerank macro-F{beta:.1f} "
            f"(Klex={top_k_lexical}, Kfinal={top_k_final}, alpha={alpha}): "
            f"{macro_fbeta:.4f}"
        )

    return macro_fbeta
