import json
import os
from typing import List, Set, Tuple
import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

train_path = "data/alqac25_train.json"
law_path = "data/alqac25_law.json"
test_path = "data/alqac25_private_test_Task_1.json"

with open(law_path, "r", encoding="utf-8") as f:
    law_data = json.load(f)

with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

law_documents = []

for law in law_data:
    law_id = law["id"]
    for artc in law["articles"]:
        law_documents.append(
            {
                "doc_id": len(law_documents),
                "law_id": law_id,
                "article_id": artc["id"],
                "text": artc["text"],
            }
        )

DOCID_TO_META = {d["doc_id"]: d for d in law_documents}

def underthesea_tokenizer(text: str):
    if not isinstance(text, str):
        text = str(text)
    tokenized = word_tokenize(text, format="text")
    return tokenized.lower().split()

corpus_tokens = [underthesea_tokenizer(doc["text"]) for doc in law_documents]
bm25 = BM25Okapi(corpus_tokens)

"""
bm-25 all law document
return top_k candidate docs BM25 scores 
"""
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
def build_article_embedding(
    docs,
    model: str = emb_model,
    batch_size: int = 128,
    cache_path: str = "article_embeddings.npy",
    force_recompute: bool = False,
):
    if (not force_recompute) and os.path.exists(cache_path):
        print("Loading cached article embeddings from", cache_path)
        return np.load(cache_path)

    print("Computing article embeddings with OpenAI:", model)
    texts = [d["text"] for d in docs]
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        print(f"  Embedding batch {start}-{start+len(batch)-1} / {len(texts)}")
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [
            np.array(item.embedding, dtype="float32") for item in response.data
        ]
        all_embeddings.extend(batch_embeddings)

    article_embeddings = np.vstack(all_embeddings)
    np.save(cache_path, article_embeddings)
    print("Saved article embeddings to", cache_path)
    return article_embeddings

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

"""
1. underthesea + bm25 to get top_k_lexical law documents
2. Compute cosine similarity between question embedding with article embedding
3. combine bm25 score with cosine similarity into one socre : 
    combined = alpha * cos_norm + (1-alpha) * bm25_norm
"""
def retrieve_and_rerank_with_qemb(
    question_text: str,
    question_embedding: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
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

"""
return predictions in task 1 format for a list of question
"""
def build_predictions_for_questions_with_embs(
    questions,
    question_embeddings: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
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

"""
return predictions in task 1 format for a list of question with scores
"""
def build_predictions_for_questions_with_scores(
    questions,
    question_embeddings: np.ndarray,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
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
        )

        pred_articles = [
            {
                "law_id": r["law_id"],
                "article_id": r["article_id"],
                "bm25_score": r["bm25_score"],
                "embedding_score": r["embedding_score"],
                "combined_score": r["combined_score"],
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

"""
gold : set of relevant article of the training data (algac25_train.json) for the question
preds : my system predictions of said relevant articles for the question
"""
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

"""
macro f2 for BM25 only, evalute the top-k choices from bm25 with the training data (algac25_train.json) :
    find fbeta between gold and predicted sets
    the macro fbeta (the average of all questions)
"""
def macro_f2_bm25_topk(
    k: int = 3,
    beta: float = 2.0,
    verbose: bool = True,
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

"""
evaluate full bm25 + embedding rerank system on alqac25_train using macro fbeta
"""
def macro_f2_rerank(
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
    beta: float = 2.0,
    verbose: bool = True,
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


if __name__ == "__main__":
    print("=== Sanity check on first train question ===")
    ex = train_data[0]
    print("Question ID:", ex["question_id"])
    print("Text      :", ex["text"])
    print("Gold      :", ex["relevant_articles"])
    print()

    bm25_res = bm25_lexical_retrieve(ex["text"], top_k=5)
    print("Top-5 BM25 only:")
    for i, c in enumerate(bm25_res, start=1):
        print(
            f"  [{i}] law={c['law_id']} article={c['article_id']} BM25={c['bm25_score']:.2f}"
        )
    print()

    print("=== BM25-only macro-F2 for different top-k ===")
    for k in range(1, 6):
        macro_f2_bm25_topk(k=k, beta=2.0, verbose=True)
    print()

    print("=== Rerank macro-F2 (one config) ===")
    macro_f2_rerank(
        top_k_lexical=200,
        top_k_final=1,
        alpha=0.4,
        beta=2.0,
        verbose=True,
    )
    
    #Finding the best hyperparameter
    
    # print("\n=== Hyperparameter search for best macro-F2 ===")
    # best_score = -1.0
    # best_config = None
    

    # lexical_candidates = [50, 100, 150, 200]
    # final_candidates = [1, 2, 3, 5]
    # alpha_candidates = [0.2, 0.4, 0.6, 0.8]

    # for k_lex in lexical_candidates:
    #     for k_fin in final_candidates:
    #         for a in alpha_candidates:
    #             score = macro_f2_rerank(
    #                 top_k_lexical=k_lex,
    #                 top_k_final=k_fin,
    #                 alpha=a,
    #                 beta=2.0,
    #                 verbose=False,  # keep output clean inside the search
    #             )
    #             print(
    #                 f"Klex={k_lex:3d}, Kfinal={k_fin}, alpha={a:.2f} -> macro-F2={score:.4f}"
    #             )

    #             if score > best_score:
    #                 best_score = score
    #                 best_config = (k_lex, k_fin, a)

    # print("\n=== Best config by macro-F2 on TRAIN ===")
    # print(
    #     f"  top_k_lexical={best_config[0]}, "
    #     f"top_k_final={best_config[1]}, "
    #     f"alpha={best_config[2]:.2f}"
    # )
    # print(f"  macro-F2={best_score:.4f}\n")

    """
    build predictions file with scores for training questions
    """
    train_predictions_with_scores = build_predictions_for_questions_with_scores(
        test_data,
        test_question_embeddings,
        top_k_lexical=200,
        top_k_final=1,
        alpha=0.4,
    )

    out_path = "alqac25_test_predictions.json"
    # out_path = "alqac25_train_predictions_with_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(train_predictions_with_scores, f, ensure_ascii=False, indent=2)

    print("Saved predictions with scores to", out_path)
