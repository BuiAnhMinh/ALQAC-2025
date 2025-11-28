# retrieval.py
import json
from typing import List, Set, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression

from config import (
    STOPWORDS_PATH,
    ARTICLE_EMB_PATH,
    TRAIN_Q_EMB_PATH,
    TEST_Q_EMB_PATH,
)
from data_loader import load_law_documents, load_train_data, load_test_data
from embeddings import embed_text

# ---------- Load data ----------
law_documents = load_law_documents()
train_data = load_train_data()
test_data = load_test_data()

# Build mapping doc_id -> metadata
DOCID_TO_META: Dict[int, Dict[str, Any]] = {d["doc_id"]: d for d in law_documents}

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


def underthesea_tokenizer(text: str):
    if not isinstance(text, str):
        text = str(text)
    tokenized = word_tokenize(text, format="text")
    tokens = tokenized.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


corpus_tokens = [underthesea_tokenizer(doc["text"]) for doc in law_documents]
bm25 = BM25Okapi(corpus_tokens)

# ---------- Load embeddings ----------
article_embedding = np.load(ARTICLE_EMB_PATH)
print("Article embeddings shape:", article_embedding.shape)

train_question_embeddings = np.load(TRAIN_Q_EMB_PATH)
print("Train question embeddings shape:", train_question_embeddings.shape)

test_question_embeddings = np.load(TEST_Q_EMB_PATH)
print("Test question embeddings shape:", test_question_embeddings.shape)


# ---------- BM25 retrieval ----------
def bm25_lexical_retrieve(question_text: str, top_k: int = 50):
    question_tokens = underthesea_tokenizer(question_text)
    scores_list = bm25.get_scores(question_tokens)
    scores = np.array(scores_list, dtype="float32")

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
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


# ---------- Feature computation ----------
def compute_candidate_features(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
):
    lexical_candidates = bm25_lexical_retrieve(
        question_text,
        top_k=top_k_lexical,
    )
    cand_doc_ids = [c["doc_id"] for c in lexical_candidates]
    bm25_scores = np.array(
        [c["bm25_score"] for c in lexical_candidates], dtype="float32"
    )

    if bm25_scores.max() > 0:
        bm25_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_norm = bm25_scores

    candidate_embedding = article_embedding[cand_doc_ids]

    dot = candidate_embedding @ question_embedding
    norms = np.linalg.norm(candidate_embedding, axis=1) * np.linalg.norm(
        question_embedding
    )
    cos_similarity = dot / (norms + 1e-8)

    return lexical_candidates, bm25_norm, cos_similarity


# ---------- Retrieval + rerank ----------
def retrieve_and_rerank_with_qemb(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
):
    lexical_candidates, bm25_norm, cos_similarity = compute_candidate_features(
        question_text,
        question_embedding,
        top_k_lexical=top_k_lexical,
    )

    if logreg_model is not None:
        features = np.stack([bm25_norm, cos_similarity], axis=1)
        combined_scores = logreg_model.predict_proba(features)[:, 1]
    else:
        cos_min = cos_similarity.min()
        cos_max = cos_similarity.max()

        if cos_max - cos_min > 1e-6:
            cos_norm = (cos_similarity - cos_min) / (cos_max - cos_min)
        else:
            cos_norm = np.zeros_like(cos_similarity) + 0.5

        combined_scores = alpha * cos_norm + (1.0 - alpha) * bm25_norm

    order = np.argsort(combined_scores)[::-1]
    top_order = order[:top_k_final]

    ranked_results = []
    for idx in top_order:
        base = lexical_candidates[int(idx)].copy()
        base["embedding_score"] = float(cos_similarity[idx])
        base["combined_score"] = float(combined_scores[idx])
        ranked_results.append(base)

    return ranked_results


def build_logreg_model(
    top_k_lexical: int = 200,
):
    X_features = []
    y_labels = []

    for i, q in enumerate(tqdm(train_data, desc="Building LogReg training data")):
        question_text = q["text"]
        gold_articles = {
            (ra["law_id"], ra["article_id"]) for ra in q["relevant_articles"]
        }
        q_emb = train_question_embeddings[i]

        lexical_candidates, bm25_norm, cos_similarity = compute_candidate_features(
            question_text,
            q_emb,
            top_k_lexical=top_k_lexical,
        )

        for j, cand in enumerate(lexical_candidates):
            law_art = (cand["law_id"], cand["article_id"])
            label = 1 if law_art in gold_articles else 0
            X_features.append([bm25_norm[j], cos_similarity[j]])
            y_labels.append(label)

    X = np.array(X_features, dtype="float32")
    y = np.array(y_labels, dtype="int32")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X, y)
    return model


def build_predictions_for_questions_with_embs(
    questions,
    question_embeddings: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
):
    predictions = []

    for i, q in enumerate(tqdm(questions, desc="Building predictions")):
        question_text = q["text"]
        question_id = q["question_id"]
        q_emb = question_embeddings[i]

        ranked = retrieve_and_rerank_with_qemb(
            question_text,
            q_emb,
            top_k_lexical=top_k_lexical,
            top_k_final=top_k_final,
            alpha=alpha,
            logreg_model=logreg_model,
        )

        pred_articles = [
            {"law_id": r["law_id"], "article_id": r["article_id"]} for r in ranked
        ]

        predictions.append(
            {
                "question_id": question_id,
                "relevant_articles": pred_articles,
            }
        )

    return predictions


def build_predictions_for_questions_with_scores(
    questions,
    question_embeddings: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    logreg_model: LogisticRegression | None = None,
):
    predictions = []

    for i, q in enumerate(tqdm(questions, desc="Building predictions with scores")):
        question_text = q["text"]
        question_id = q["question_id"]
        q_emb = question_embeddings[i]

        ranked = retrieve_and_rerank_with_qemb(
            question_text,
            q_emb,
            top_k_lexical=top_k_lexical,
            top_k_final=top_k_final,
            alpha=alpha,
            logreg_model=logreg_model,
        )

        pred_articles = [
            {
                "law_id": r["law_id"],
                "article_id": r["article_id"],
            }
            for r in ranked
        ]

        predictions.append(
            {
                "question_id": question_id,
                "relevant_articles": pred_articles,
            }
        )

    return predictions


# ---------- Metrics ----------
def fbeta_for_sets(
    gold: Set[Tuple[str, str]],
    preds: Set[Tuple[str, str]],
    beta: float = 2.0,
) -> float:
    true_positive = len(gold & preds)
    false_positive = len(preds - gold)
    false_negative = len(gold - preds)

    if true_positive == 0 and false_negative == 0 and false_positive == 0:
        return 0.0

    if true_positive == 0:
        return 0.0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    beta2 = beta ** 2
    denom = beta2 * precision + recall

    if denom == 0:
        return 0.0

    fbeta = (1 + beta2) * precision * recall / denom
    return fbeta


def macro_f2_bm25_topk(
    k: int = 3,
    beta: float = 2.0,
    verbose: bool = False,
) -> float:
    f_scores: List[float] = []
    for q in tqdm(train_data, desc=f"[Macro] BM25 F{beta} @ top-{k}"):
        question_text = q["text"]
        gold_articles = {
            (ra["law_id"], ra["article_id"]) for ra in q["relevant_articles"]
        }

        bm25_results = bm25_lexical_retrieve(question_text, top_k=k)
        pred_articles = {(c["law_id"], c["article_id"]) for c in bm25_results}

        f_question = fbeta_for_sets(gold_articles, pred_articles, beta=beta)
        f_scores.append(f_question)

    macro_fbeta = float(np.mean(f_scores))
    if verbose:
        print(f"BM25 macro-F{beta:.1f} @ top-{k}: {macro_fbeta:.4f}")

    return macro_fbeta


def macro_f2_rerank(
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    beta: float = 2.0,
    verbose: bool = False,
    logreg_model: LogisticRegression | None = None,
) -> float:
    f_scores: List[float] = []

    desc = f"[Macro] Rerank F{beta} (Klex={top_k_lexical}, Kfinal={top_k_final}, alpha={alpha})"
    for i, q in enumerate(tqdm(train_data, desc=desc)):
        question_text = q["text"]
        gold_articles = {
            (ra["law_id"], ra["article_id"]) for ra in q["relevant_articles"]
        }

        q_emb = train_question_embeddings[i]

        ranked = retrieve_and_rerank_with_qemb(
            question_text,
            q_emb,
            top_k_lexical=top_k_lexical,
            top_k_final=top_k_final,
            alpha=alpha,
            logreg_model=logreg_model,
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
