import json
import os
import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
from config import OPENAI_API_KEY
train_path = "data/alqac25_train.json"
law_path = "data/alqac25_law.json"
test_path = "data/alqac25_private_test_Task_1.json"

# understand how underthesea tokenize the text
# text = "Ủy ban nhân dân cấp tỉnh cấp giấy phép phim."

# tokens = word_tokenize(text.lower())
# print(tokens)

with open(law_path, "r", encoding="utf-8") as f:
    law_data = json.load(f)

with open(train_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(test_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    
# print("Train samples:", len(train_data))
# print("Test samples :", len(test_data))
# print("Number of laws:", len(law_data))   


law_documents = []

for law in law_data :
    law_id = law["id"]
    for artc in law["articles"] :
        law_documents.append (
            {
                "doc_id": len(law_documents),
                "law_id": law_id,
                "article_id": artc["id"],
                "text": artc["text"],
            }
        )

DOCID_TO_META = {d["doc_id"]: d for d in law_documents}
        
# print("Total law articles (documents):", len(law_documents))

# Underthesea Tokenizer 

def underthesea_tokenizer(text:str):
    if not isinstance(text, str):
        text = str(text)
    tokenized = word_tokenize(text, format="text")
    return tokenized.lower().split()

    #tokinizer all articles in law 
corpus_tokens = [underthesea_tokenizer(doc["text"]) for doc in law_documents]
bm25 = BM25Okapi(corpus_tokens)

"""
bm-25 all law document
return top_k candidate docs BM25 scores 
"""
def bm25_lexical_retrieve(question_text: str, top_k: int = 50): #placeholder
    question_tokens = underthesea_tokenizer(question_text)
    scores_list = bm25.get_scores(question_tokens)
    scores = np.array(scores_list, dtype="float32")
    
    top_idx = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_idx :
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
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.environ["OPENROUTER_API_KEY"])
emb_model = "openai/text-embedding-3-small"

"""
precompute embedding for all articles
if cache exists and force_recompute = false, load cache
if not, call OpenAI API in batches and save
return numpy array of (num_docs, emb_dim) """

def build_article_embedding(docs, 
                            model: str=emb_model, 
                            batch_size: int=128,
                            cache_path: str = "article_embedding.npy",
                            force_recompute: bool = False):
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
        batch_embeddings = [np.array(item.embedding, dtype="float32") for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
    article_embeddings = np.vstack(all_embeddings)
    np.save(cache_path, article_embeddings)
    print("Saved article embeddings to", cache_path)
    return article_embeddings

#sanity check
article_embedding = build_article_embedding(
    law_documents,
    cache_path="article_embeddings.npy",
    force_recompute=False,
)
print("Article embeddings shape:", article_embedding.shape)


QUESTION_EMB_CACHE = {}
#embed the training questions and test questions
#cache to avoid API calls for the same text
def build_embedded_text(text: str, model: str = emb_model) -> np.ndarray:
    if text in QUESTION_EMB_CACHE:
        return QUESTION_EMB_CACHE[text]
    response = client.embeddings.create(model = model, input = [text])
    vector = np.array(response.data[0].embedding, dtype = "float32")
    QUESTION_EMB_CACHE[text] = vector
    return vector

#retrieve and rerank
"""
1. underthesea + bm25 to get top_k_lexical law documents
2. Compute cosine similarity between question embedding with article embedding
3. combine bm25 score with cosine similarity into one socre : 
    combined = alpha * cos_norm + (1-alpha) * bm25_norm
"""
def retrieve_and_rerank(
    question_text: str,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
):
    lexical_candidates = bm25_lexical_retrieve(
        question_text,
        top_k=top_k_lexical,
    )
    cand_doc_ids = [c["doc_id"] for c in lexical_candidates]
    bm25_scores = np.array([c["bm25_score"] for c in lexical_candidates], dtype="float32")

    # Normalize BM25 scores to [0, 1] e.g. 0, 0.2, 0.5, ... , 1
    if bm25_scores.max() > 0:
        bm25_norm = bm25_scores / bm25_scores.max()
    else:
        bm25_norm = bm25_scores

    # embedding similarity
    question_embedding = build_embedded_text(question_text)               # shape (d,)
    candidate_embedding = article_embedding[cand_doc_ids]    # shape (n_cand, d)

    # cosine similarity
    dot = candidate_embedding @ question_embedding
    norms = np.linalg.norm(candidate_embedding, axis=1) * np.linalg.norm(question_embedding)
    cos_similarity = dot / (norms + 1e-8)  # avoid divide by zero

    # Normalize cosine similarity to [0, 1]
    cos_min = cos_similarity.min()
    cos_max =  cos_similarity.max()
    
    if cos_max - cos_min > 1e-6:
        cos_norm = (cos_similarity - cos_min) / (cos_max - cos_min)
    else:
        cos_norm = np.zeros_like(cos_similarity) + 0.5

    # combine scores
    combined_scores = alpha * cos_norm + (1.0 - alpha) * bm25_norm

    # Sort by combined score
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

def build_predictions_for_questions(
    questions,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.6,
):
    predictions = []

    for q in tqdm(questions, desc="Building predictions"):
        question_text = q["text"]
        question_id = q["question_id"]

        ranked = retrieve_and_rerank(
            question_text,
            top_k_lexical=top_k_lexical,
            top_k_final=top_k_final,
            alpha=alpha,
        )

        pred_articles = [
            {"law_id": r["law_id"], "article_id": r["article_id"]}
            for r in ranked
        ]

        predictions.append(
            {
                "question_id": question_id,
                "relevant_articles": pred_articles,
            }
        )

    return predictions


#sanity check
if __name__ == "__main__":
    print("\n=== Sanity check on first train question ===")
    example_train = train_data[0]
    print("Question ID:", example_train["question_id"])
    print("Text      :", example_train["text"])
    print("Gold      :", example_train["relevant_articles"])
    print()

    bm25_only = bm25_lexical_retrieve(example_train["text"], top_k=5)
    print("Top-5 BM25 only:")
    for i, c in enumerate(bm25_only, start=1):
        print(f"  [{i}] law={c['law_id']} article={c['article_id']} BM25={c['bm25_score']:.2f}")
    print()

    ranked = retrieve_and_rerank(example_train["text"], top_k_lexical=200, top_k_final=5, alpha=0.6)
    print("Top-5 BM25 + Embedding:")
    for i, c in enumerate(ranked, start=1):
        print(
            f"  [{i}] law={c['law_id']} article={c['article_id']} "
            f"BM25={c['bm25_score']:.2f} Cos={c['embedding_score']:.3f} Comb={c['combined_score']:.3f}"
        )
    print()

    # Build predictions for private test and save
    print("=== Building predictions for private test Task 1 ===")
    test_predictions = build_predictions_for_questions(
        test_data,
        top_k_lexical=200,
        top_k_final=5,
        alpha=0.6,
    )

    out_path = "alqac25_task1_predictions_openai.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=2)

    print("Saved test predictions to", out_path)