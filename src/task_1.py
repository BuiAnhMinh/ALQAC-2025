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

def bm25_lexical_retrive(question_text: str, top_k: int = 50): #placeholder
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

client = OpenAI(api_key = OPENAI_API_KEY)
emb_model = "text-embedding-3-small"

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
        resp = client.embeddings.create(model=model, input=batch)
        batch_embs = [np.array(item.embedding, dtype="float32") for item in resp.data]
        all_embeddings.extend(batch_embs)

    article_embeddings = np.vstack(all_embeddings)
    np.save(cache_path, article_embeddings)
    print("Saved article embeddings to", cache_path)
    return article_embeddings



