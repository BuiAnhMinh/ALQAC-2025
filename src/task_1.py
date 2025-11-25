import json
import numpy as np
from tqdm import tqdm
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
import random

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

