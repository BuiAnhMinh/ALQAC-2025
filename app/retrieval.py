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
from embedding import embed_text

# ---------- Load data ----------
law_documents = load_law_documents()
train_data = load_train_data()
test_data = load_test_data()

# Build mapping doc_id -> metadata
DOCID_TO_META: Dict[int, Dict[str, Any]] = {d["doc_id"]: d for d in law_documents}

<<<<<<< HEAD
with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(zalo_law_path, "r", encoding="utf-8") as f:
    zalo_law_data = json.load(f)
    
law_documents = []

skipped_empty_alqac = 0
skipped_empty_zalo = 0

for law in law_data:
    law_id = law["id"]
    for artc in law["articles"]:
        raw_text = artc.get("text", "")
        if raw_text is None:
            raw_text = ""
        text = str(raw_text).strip()

        # skip completely empty articles
        if len(text) == 0:
            skipped_empty_alqac += 1
            continue

        law_documents.append(
            {
                "law_id": law_id,
                "article_id": artc["id"],
                "text": text,
            }
        )
        
num_alqac_docs = len(law_documents)
        
for law in zalo_law_data:
    law_id = law["id"]
    for artc in law["articles"]:
        raw_text = artc.get("text", "")
        if raw_text is None:
            raw_text = ""
        text = str(raw_text).strip()

        # skip completely empty articles
        if len(text) == 0:
            skipped_empty_zalo += 1
            continue

        law_documents.append(
            {
                "doc_id": len(law_documents),
                "law_id": law_id,
                "article_id": artc["id"],
                "text": text,
            }
        )
        
num_total_docs = len(law_documents)
num_zalo_docs = num_total_docs - num_alqac_docs

print(f"ALQAC articles: {num_alqac_docs} (skipped empty: {skipped_empty_alqac})")
print(f"Zalo  articles: {num_zalo_docs} (skipped empty: {skipped_empty_zalo})")
print(f"Total articles: {num_total_docs}")

DOCID_TO_META = {d["doc_id"]: d for d in law_documents}

# debug flattened_law_articles

# for d in law_documents:
#     t = d["text"]
#     if not isinstance(t, str):
#         t = str(t)
#     d["text_len"] = len(t)

# debug_path = "data/all_laws_flattened_debug.json"
# with open(debug_path, "w", encoding="utf-8") as f:
#     json.dump(law_documents, f, ensure_ascii=False, indent=2)

# print("Flattened law corpus saved to", debug_path)
# print(f"ALQAC docs are doc_id 0 .. {num_alqac_docs-1}")
# print(f"Zalo  docs are doc_id {num_alqac_docs} .. {num_total_docs-1}")

#loading stopwords
=======
# ---------- Stopwords & tokenizer ----------
>>>>>>> 53babd6 (hello)
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


<<<<<<< HEAD
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
emb_model = "openai/text-embedding-3-small"

"""
precompute embedding for all articles
if cache exists and force_recompute = false, load cache
if not, call OpenAI API in batches and save
return numpy array of (num_docs, emb_dim) 
"""

MAX_CHARS = 4000      # truncate super-long texts
BATCH_SIZE = 64       # you already use 64, good
SAVE_EVERY = 50       # save to disk after every 50 batches (tweak as you like)

def build_article_embedding(
    docs,
    model: str = emb_model,
    cache_path: str = "article_embeddings_all.npy",
    force_recompute: bool = False,
):
    # 1. Load existing cache if any
    if os.path.exists(cache_path) and not force_recompute:
        existing = np.load(cache_path)
        done = existing.shape[0]
        print(f"Loading cached embeddings from {cache_path}, already have {done}/{len(docs)} docs")
        # store as list of arrays so we can append easily
        all_embeddings = [emb for emb in existing]
    else:
        print(f"No existing cache, starting embeddings from scratch for {len(docs)} docs")
        done = 0
        all_embeddings = []

    # 2. Preprocess texts
    texts = []
    for d in docs:
        t = d["text"]
        if not isinstance(t, str):
            t = str(t)
        t = t.replace("\n", " ")
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
        texts.append(t)

    total = len(texts)
    batch_count_since_save = 0

    # 3. Embed from `done` onwards
    for start in range(done, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"  Embedding batch {start}-{start+len(batch)-1} / {total}")

        try:
            response = client.embeddings.create(model=model, input=batch)
        except Exception as e:
            print(f"!! Error calling embeddings API on batch {start}-{start+len(batch)-1}: {e}")
            print("   Batch lengths:", [len(x) for x in batch])
            break  # stop here so we still save what we already have

        if not hasattr(response, "data") or len(response.data) == 0:
            print(f"!! No data returned for batch {start}-{start+len(batch)-1}")
            print("   Batch lengths:", [len(x) for x in batch])
            break  # also stop, but keep progress

        batch_embeddings = [
            np.array(item.embedding, dtype="float32") for item in response.data
        ]
        all_embeddings.extend(batch_embeddings)
        batch_count_since_save += 1

        # 4. Periodically save progress
        if batch_count_since_save >= SAVE_EVERY:
            arr = np.vstack(all_embeddings)
            np.save(cache_path, arr)
            print(f"  [checkpoint] Saved {arr.shape[0]} embeddings to {cache_path}")
            batch_count_since_save = 0

    # 5. Final save at the end (or at last checkpoint point)
    if all_embeddings:
        arr = np.vstack(all_embeddings)
        np.save(cache_path, arr)
        print(f"Saved {arr.shape[0]} embeddings to {cache_path}")
        return arr
    else:
        raise RuntimeError("No embeddings computed at all")
    
"""
embed the training questions and test questions
cache to avoid API calls for the same text
"""
def build_question_embeddings(
    questions,
    model: str = emb_model,
    batch_size: int = 64,
    cache_path: str = "train_question_embeddings.npy",
    force_recompute: bool = False,
):
    if (not force_recompute) and os.path.exists(cache_path):
        print("Loading cached question embeddings from", cache_path)
        return np.load(cache_path)

    print("Computing question embeddings with OpenAI:", model)
    texts = [q["text"] for q in questions]
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        print(f"  Embedding question batch {start}-{start+len(batch)-1} / {len(texts)}")
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [
            np.array(item.embedding, dtype="float32") for item in response.data
        ]
        all_embeddings.extend(batch_embeddings)

    question_embeddings = np.vstack(all_embeddings)
    np.save(cache_path, question_embeddings)
    print("Saved question embeddings to", cache_path)
    return question_embeddings


article_embedding = build_article_embedding(
    law_documents,
    cache_path="article_embeddings.npy",
    force_recompute=False,
)
print("Article embeddings shape:", article_embedding.shape)

train_question_embeddings = build_question_embeddings(
    train_data,
    cache_path="train_question_embeddings.npy",
    force_recompute=False,
)
print("Train question embeddings shape:", train_question_embeddings.shape)

test_question_embeddings = build_question_embeddings(
    test_data,
    cache_path="test_question_embeddings.npy",
    force_recompute=False,
)
print("Test question embeddings shape:", test_question_embeddings.shape)


=======
# ---------- Feature computation ----------
>>>>>>> 53babd6 (hello)
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


<<<<<<< HEAD
"""
1. underthesea + bm25 to get top_k_lexical law documents
2. Compute cosine similarity between question embedding with article embedding
"""


article_embedding = build_article_embedding(
    law_documents,
    cache_path="article_embeddings.npy",
    force_recompute=False,
)
print("Article embeddings shape:", article_embedding.shape)

train_question_embeddings = build_question_embeddings(
    train_data,
    cache_path="train_question_embeddings.npy",
    force_recompute=False,
)
print("Train question embeddings shape:", train_question_embeddings.shape)

test_question_embeddings = build_question_embeddings(
    test_data,
    cache_path="test_question_embeddings.npy",
    force_recompute=False,
)
print("Test question embeddings shape:", test_question_embeddings.shape)


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


"""
1. underthesea + bm25 to get top_k_lexical law documents
2. Compute cosine similarity between question embedding with article embedding
3. combine bm25 score with cosine similarity into one socre : 
    combined = alpha * cos_norm + (1-alpha) * bm25_norm
"""

=======
# ---------- Retrieval + rerank ----------
>>>>>>> 53babd6 (hello)
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

<<<<<<< HEAD
    return macro_fbeta
=======
    return macro_fbeta
>>>>>>> 53babd6 (hello)
