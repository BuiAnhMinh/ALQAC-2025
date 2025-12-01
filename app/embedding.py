from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from app.config import (
    EMB_MODEL,
    MAX_CHARS,
    BATCH_SIZE,
    SAVE_EVERY,
    ARTICLE_EMB_PATH,
    TRAIN_Q_EMB_PATH,
    TEST_Q_EMB_PATH,
    get_client,
)
from app.data_loader import load_law_documents, load_train_data, load_test_data


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\n", " ")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def preprocess_texts(docs: List[Dict[str, Any]]) -> List[str]:
    return [_clean_text(d["text"]) for d in docs]


def build_article_embeddings(cache_path: Path = ARTICLE_EMB_PATH) -> np.ndarray:
    """
    Embed all law documents (ALQAC + Zalo) and cache to a .npy file.
    """
    docs = load_law_documents()
    total = len(docs)
    print(f"Total docs to embed: {total}")

    texts = preprocess_texts(docs)

    done = 0
    all_embeddings = []
    cache_path = Path(cache_path)

    if cache_path.exists():
        existing = np.load(cache_path)
        done = existing.shape[0]
        all_embeddings = [emb for emb in existing]
        print(f"Found existing cache: {done}/{total} docs already embedded.")
        if done >= total:
            print("All docs already embedded, nothing to do.")
            return existing

    client = get_client()
    batch_count_since_save = 0

    for start in range(done, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"Embedding batch {start}-{start+len(batch)-1} / {total}")

        try:
            resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        except Exception as e:
            print(f"!! Error calling embeddings API on batch {start}-{start+len(batch)-1}: {e}")
            if all_embeddings:
                arr = np.vstack(all_embeddings)
                np.save(cache_path, arr)
                print(f"[checkpoint on error] Saved {arr.shape[0]} embeddings to {cache_path}")
            raise

        batch_embs = [np.array(item.embedding, dtype="float32") for item in resp.data]
        all_embeddings.extend(batch_embs)
        batch_count_since_save += 1

        if batch_count_since_save >= SAVE_EVERY:
            arr = np.vstack(all_embeddings)
            np.save(cache_path, arr)
            print(f"[checkpoint] Saved {arr.shape[0]} embeddings to {cache_path}")
            batch_count_since_save = 0

    arr = np.vstack(all_embeddings)
    np.save(cache_path, arr)
    print(f"Finished! Saved {arr.shape[0]} embeddings to {cache_path}")
    return arr


def build_question_embeddings(
    questions: List[Dict[str, Any]],
    cache_path: Path,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists():
        print("Loading cached question embeddings from", cache_path)
        return np.load(cache_path)

    client = get_client()
    texts = [_clean_text(q["text"]) for q in questions]

    all_embeddings = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"Embedding question batch {start}-{start+len(batch)-1} / {total}")
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        batch_embs = [np.array(item.embedding, dtype="float32") for item in resp.data]
        all_embeddings.extend(batch_embs)

    arr = np.vstack(all_embeddings)
    np.save(cache_path, arr)
    print(f"Saved question embeddings to {cache_path}")
    return arr


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text (used at runtime / Task 2).
    """
    client = get_client()
    text = _clean_text(text)
    resp = client.embeddings.create(model=EMB_MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")


def main():
    article_embs = build_article_embeddings()
    print("Article embeddings shape:", article_embs.shape)

    train_data = load_train_data()
    test_data = load_test_data()

    train_q_embs = build_question_embeddings(train_data, cache_path=TRAIN_Q_EMB_PATH)
    print("Train question embeddings shape:", train_q_embs.shape)

    test_q_embs = build_question_embeddings(test_data, cache_path=TEST_Q_EMB_PATH)
    print("Test question embeddings shape:", test_q_embs.shape)


if __name__ == "__main__":
    main()
