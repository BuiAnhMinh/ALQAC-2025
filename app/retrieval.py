import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression

from app.config import (
    STOPWORDS_PATH,
    TRAIN_Q_EMB_PATH,
    TEST_Q_EMB_PATH,
)
from app.config import get_connection
from app.data_loader import load_train_data, load_test_data


# ============================================================
# 0. Tokenizer + stopwords
# ============================================================

def load_stopwords() -> Set[str]:
    """
    Load Vietnamese stopwords from STOPWORDS_PATH (one token per line).
    If the file is missing, returns an empty set.
    """
    if STOPWORDS_PATH and Path(STOPWORDS_PATH).exists():
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}
    return set()


STOPWORDS: Set[str] = load_stopwords()


def underthesea_tokenizer(text: str) -> List[str]:
    """
    Tokenize text using underthesea and remove stopwords.
    """
    tokens = word_tokenize(text, format="text").split()
    return [t for t in tokens if t not in STOPWORDS]


# ============================================================
# 1. Load articles from DB (single source of truth)
# ============================================================

def load_articles_from_db() -> List[Dict[str, Any]]:
    """
    Load ALL articles from the database.

    We rely ONLY on the `articles` table now:
        id, law_fk, law_id, article_id, text, embedding, tokens, created_at

    We do NOT use law_documents.json anymore.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, law_id, article_id, text, tokens, embedding
            FROM articles
            ORDER BY id;
            """
        )
        rows = cur.fetchall()

        articles: List[Dict[str, Any]] = []

        for db_id, law_id, article_id, text, tokens, embedding in rows:
            # ---- tokens ----
            if isinstance(tokens, list):
                token_list = tokens
            else:
                # stored as JSON / text
                token_list = json.loads(tokens)

            # ---- embedding ----
            # If using pgvector, psycopg2 usually returns a Python list already.
            if isinstance(embedding, list) or isinstance(embedding, tuple):
                emb_vec = np.array(embedding, dtype="float32")
            else:
                # if stored as JSON string
                emb_vec = np.array(json.loads(embedding), dtype="float32")

            articles.append(
                {
                    "db_id": db_id,
                    "law_id": law_id,
                    "article_id": article_id,
                    "text": text,
                    "tokens": token_list,
                    "embedding": emb_vec,
                }
            )

        # assign internal doc_id = index position
        for idx, art in enumerate(articles):
            art["doc_id"] = idx

        print(f"Loaded {len(articles)} articles from DB.")
        return articles

    finally:
        cur.close()
        conn.close()


ARTICLES: List[Dict[str, Any]] = load_articles_from_db()

# Mapping internal doc_id -> metadata
DOCID_TO_META: Dict[int, Dict[str, Any]] = {
    a["doc_id"]: a for a in ARTICLES
}

# Mapping (law_id, article_id) -> internal doc_id
DOC_KEY_TO_ID: Dict[Tuple[str, str], int] = {
    (a["law_id"], a["article_id"]): a["doc_id"] for a in ARTICLES
}


# ============================================================
# 2. Build BM25 corpus and article embeddings from DB data
# ============================================================

# BM25 corpus = tokens column
corpus_tokens: List[List[str]] = [a["tokens"] for a in ARTICLES]
bm25 = BM25Okapi(corpus_tokens)

# article_embedding array = stack of embedding column
article_embedding: np.ndarray = np.stack(
    [a["embedding"] for a in ARTICLES],
    axis=0,
).astype("float32")

print("Article embeddings shape:", article_embedding.shape)


# ============================================================
# 3. Load question embeddings (still from .npy)
# ============================================================

train_question_embeddings: np.ndarray = np.load(TRAIN_Q_EMB_PATH)
test_question_embeddings: np.ndarray = np.load(TEST_Q_EMB_PATH)

print("Train question embeddings shape:", train_question_embeddings.shape)
print("Test question embeddings shape:", test_question_embeddings.shape)


# ============================================================
# 4. Load ALQAC train/test question JSON (NOT law docs!)
# ============================================================

train_data: List[Dict[str, Any]] = load_train_data()
test_data: List[Dict[str, Any]] = load_test_data()


# ============================================================
# 5. BM25 lexical retrieval
# ============================================================

def bm25_lexical_retrieve(
    question_text: str,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Pure lexical retrieval from BM25 over tokenized corpus.
    Returns a list of:
      {
        "doc_id": int,          # internal index into article_embedding / DOCID_TO_META
        "bm25_score": float,
        "law_id": str,
        "article_id": str,
        "text": str,
      }

    You can set top_k=200 (or any number) to get more candidates.
    """
    question_tokens = underthesea_tokenizer(question_text)
    scores_list = bm25.get_scores(question_tokens)
    scores = np.array(scores_list, dtype="float32")

    # indices of docs sorted by BM25 score (desc)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        doc = DOCID_TO_META[int(idx)]
        results.append(
            {
                "doc_id": int(idx),
                "bm25_score": float(scores[idx]),
                "law_id": doc["law_id"],
                "article_id": doc["article_id"],
                "text": doc["text"],
            }
        )
    return results


# ============================================================
# 6. Feature computation (BM25 + embedding cosine)
# ============================================================

def compute_candidate_features(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Get BM25 candidates + normalized BM25 scores + cosine similarities.

    Returns:
      lexical_candidates: list of candidate doc info dicts
      bm25_norm:          normalized BM25 scores in [0, 1]
      cos_similarity:     cosine similarity between question and doc embeddings
    """
    lexical_candidates = bm25_lexical_retrieve(
        question_text,
        top_k=top_k_lexical,
    )
    cand_doc_ids = [c["doc_id"] for c in lexical_candidates]
    bm25_scores = np.array(
        [c["bm25_score"] for c in lexical_candidates], dtype="float32"
    )

    # Normalize BM25 to [0, 1]
    if bm25_scores.max() > 0:
        bm25_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_norm = bm25_scores

    # Take embeddings of candidate docs
    candidate_embedding = article_embedding[cand_doc_ids]

    # Cosine similarity
    dot = candidate_embedding @ question_embedding
    norms = np.linalg.norm(candidate_embedding, axis=1) * np.linalg.norm(
        question_embedding
    )
    cos_similarity = dot / (norms + 1e-8)

    return lexical_candidates, bm25_norm, cos_similarity


# ============================================================
# 7. Ground truth utilities (for Task 1 training/eval)
# ============================================================

def _build_gold_docid_sets() -> Dict[str, Set[int]]:
    """
    Map question_id -> set of true internal doc_ids using (law_id, article_id).
    """
    gold: Dict[str, Set[int]] = {}
    for q in train_data:
        qid = q["question_id"]
        rel_set: Set[int] = set()
        for art in q.get("relevant_articles", []):
            key = (art["law_id"], art["article_id"])
            if key in DOC_KEY_TO_ID:
                rel_set.add(DOC_KEY_TO_ID[key])
        gold[qid] = rel_set
    return gold


QUESTION_GOLD_DOCIDS: Dict[str, Set[int]] = _build_gold_docid_sets()


def f_beta_for_sets(y_true: Set[int], y_pred: Set[int], beta: float = 2.0) -> float:
    """
    F-beta for two sets of internal doc_ids.
    """
    if not y_true and not y_pred:
        return 0.0

    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0 and recall == 0:
        return 0.0

    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


# ============================================================
# 8. Macro-F2 metrics for BM25 baseline
# ============================================================

def macro_f2_bm25_topk(
    k: int = 5,
    beta: float = 2.0,
    verbose: bool = False,
) -> float:
    """
    Baseline: use top-k BM25 candidates as prediction for each train question.
    """
    scores: List[float] = []
    for q in tqdm(train_data, desc=f"BM25@{k} eval"):
        qid = q["question_id"]
        gold = QUESTION_GOLD_DOCIDS.get(qid, set())

        cands = bm25_lexical_retrieve(q["text"], top_k=k)
        pred_ids = {c["doc_id"] for c in cands}

        f = f_beta_for_sets(gold, pred_ids, beta=beta)
        scores.append(f)

    macro_f2 = float(np.mean(scores)) if scores else 0.0
    if verbose:
        print(f"[BM25@{k}] Macro-F{beta}: {macro_f2:.4f}")
    return macro_f2


# ============================================================
# 9. Logistic regression reranker
# ============================================================

def build_logreg_model(
    top_k_lexical: int = 200,
) -> LogisticRegression:
    """
    Train a logistic regression model on:
      X = [bm25_norm, cosine_similarity]
      y = 1 if doc is relevant, else 0
    """
    X_rows: List[List[float]] = []
    y_labels: List[int] = []

    for i, q in enumerate(tqdm(train_data, desc="Building LogReg training set")):
        qid = q["question_id"]
        gold = QUESTION_GOLD_DOCIDS.get(qid, set())
        q_emb = train_question_embeddings[i]

        lexical_candidates, bm25_norm, cos_sim = compute_candidate_features(
            q["text"], q_emb, top_k_lexical=top_k_lexical
        )
        cand_doc_ids = [c["doc_id"] for c in lexical_candidates]

        for j, doc_id in enumerate(cand_doc_ids):
            X_rows.append([float(bm25_norm[j]), float(cos_sim[j])])
            y_labels.append(1 if doc_id in gold else 0)

    X = np.array(X_rows, dtype="float32")
    y = np.array(y_labels, dtype="int32")

    print("Training LogisticRegression on", X.shape[0], "samples")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def _rerank_scores(
    bm25_norm: np.ndarray,
    cos_sim: np.ndarray,
    alpha: float,
    logreg_model: LogisticRegression | None = None,
) -> np.ndarray:
    """
    Combine BM25 + cosine similarity either by:
      - simple weighted sum (if logreg_model is None), or
      - logistic regression probability (if logreg_model is provided).
    """
    if logreg_model is None:
        # Simple weighted sum, after normalizing cosine to [0,1]
        cos_norm = (cos_sim + 1.0) / 2.0
        return alpha * bm25_norm + (1.0 - alpha) * cos_norm

    # Use logistic regression probabilities as combined score
    X = np.stack([bm25_norm, cos_sim], axis=1)
    probs = logreg_model.predict_proba(X)[:, 1]
    return probs.astype("float32")


def macro_f2_rerank(
    beta: float = 2.0,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
    verbose: bool = False,
) -> float:
    """
    Evaluate the reranking approach (BM25 + embeddings [+ optional LogReg]).
    """
    scores: List[float] = []

    for i, q in enumerate(tqdm(train_data, desc="Rerank eval")):
        qid = q["question_id"]
        gold = QUESTION_GOLD_DOCIDS.get(qid, set())
        q_emb = train_question_embeddings[i]

        lexical_candidates, bm25_norm, cos_sim = compute_candidate_features(
            q["text"], q_emb, top_k_lexical=top_k_lexical
        )
        cand_doc_ids = [c["doc_id"] for c in lexical_candidates]

        combined = _rerank_scores(bm25_norm, cos_sim, alpha, logreg_model)
        order = np.argsort(combined)[::-1]

        top_idx = order[:top_k_final]
        pred_ids = {cand_doc_ids[j] for j in top_idx}

        f = f_beta_for_sets(gold, pred_ids, beta=beta)
        scores.append(f)

    macro_f2 = float(np.mean(scores)) if scores else 0.0
    if verbose:
        print(
            f"[Rerank] Macro-F{beta}: {macro_f2:.4f} "
            f"(top_k_lexical={top_k_lexical}, top_k_final={top_k_final}, alpha={alpha})"
        )
    return macro_f2


# ============================================================
# 10. Predictions for test set (Task 1 output)
# ============================================================

def build_predictions_for_questions_with_scores(
    questions: List[Dict[str, Any]],
    question_embeddings: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
) -> List[Dict[str, Any]]:
    """
    Build predictions for a list of questions, including BM25, embedding, and combined scores.
    """
    predictions: List[Dict[str, Any]] = []

    for i, q in enumerate(tqdm(questions, desc="Predicting for questions")):
        qid = q["question_id"]
        q_text = q["text"]
        q_emb = question_embeddings[i]

        lexical_candidates, bm25_norm, cos_sim = compute_candidate_features(
            q_text, q_emb, top_k_lexical=top_k_lexical
        )
        cand_doc_ids = [c["doc_id"] for c in lexical_candidates]

        combined = _rerank_scores(bm25_norm, cos_sim, alpha, logreg_model)
        order = np.argsort(combined)[::-1]
        top_idx = order[:top_k_final]

        pred_articles = []
        for j in top_idx:
            c = lexical_candidates[j]
            pred_articles.append(
                {
                    "law_id": c["law_id"],
                    "article_id": c["article_id"],
                    "bm25_score": float(c["bm25_score"]),
                    "embedding_score": float(cos_sim[j]),
                    "combined_score": float(combined[j]),
                }
            )

        predictions.append(
            {
                "question_id": qid,
                "relevant_articles": pred_articles,
            }
        )

    return predictions


# ============================================================
# 11. Runtime helpers for QA (Task 2)
# ============================================================

def retrieve_and_rerank(
    question_text: str,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: embed question on-the-fly is handled in Task 2.
    Here we assume you will call retrieve_and_rerank_with_qemb instead.
    Kept for backward compatibility; you can remove if unused.
    """
    raise NotImplementedError(
        "Use retrieve_and_rerank_with_qemb() after embedding the question "
        "with app.embedding.embed_text()."
    )


def retrieve_and_rerank_with_qemb(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
) -> List[Dict[str, Any]]:
    """
    Runtime helper used by Task 2: given a raw question and its embedding,
    return top_k_final articles with full scores.
    """
    lexical_candidates, bm25_norm, cos_sim = compute_candidate_features(
        question_text, question_embedding, top_k_lexical=top_k_lexical
    )
    cand_doc_ids = [c["doc_id"] for c in lexical_candidates]
    combined = _rerank_scores(bm25_norm, cos_sim, alpha, logreg_model)

    order = np.argsort(combined)[::-1]
    top_idx = order[:top_k_final]

    results: List[Dict[str, Any]] = []
    for j in top_idx:
        c = lexical_candidates[j]
        results.append(
            {
                "doc_id": cand_doc_ids[j],
                "law_id": c["law_id"],
                "article_id": c["article_id"],
                "text": c["text"],
                "bm25_score": float(c["bm25_score"]),
                "embedding_score": float(cos_sim[j]),
                "combined_score": float(combined[j]),
            }
        )
    return results
