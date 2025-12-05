"""
zalo_embed_questions.py

Offline script to embed all Zalo questions once and save them to disk.

Outputs (in ./data):
  - zalo_question_embeddings.npy  # float32, shape (N, d)
  - zalo_question_ids.json        # list[str], question_ids in same order
  - zalo_question_meta.json       # metadata (model_key, dim, etc.)

Run:
  python zalo_embed_questions.py

Or (if inside a package):
  python -m app.scripts.zalo_embed_questions
"""

from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from app.data_loader import load_zalo_questions
from app.embedding import embed_text, DEFAULT_MODEL_KEY


# ---- Config ----
DATA_DIR = Path("data")

EMB_PATH = DATA_DIR / "zalo_question_embeddings.npy"
ID_PATH = DATA_DIR / "zalo_question_ids.json"
META_PATH = DATA_DIR / "zalo_question_meta.json"

MAX_CHARS = 4000        # truncate very long questions (defensive)
SAVE_EVERY = 50         # checkpoint to disk every N questions


def main(model_key: str = DEFAULT_MODEL_KEY) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load all Zalo questions
    questions = load_zalo_questions()
    print(f"Loaded {len(questions)} Zalo questions")

    question_ids: list[str] = []
    texts: list[str] = []

    for q in questions:
        qid = q.get("question_id")
        text = q.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        question_ids.append(qid)
        texts.append(text)

    # 2) Embed each question
    embeddings: list[np.ndarray] = []
    dim: int | None = None

    for i, text in enumerate(tqdm(texts, desc="Embedding Zalo questions")):
        # embed_text should already use your OpenRouter/OpenAI config & model_key
        emb = embed_text(text, model_key=model_key)

        # Normalize to 1D float32 vector
        if isinstance(emb, list):
            emb = np.array(emb, dtype="float32")
        else:
            emb = np.asarray(emb, dtype="float32")

        # If embed_text returns shape (1, d), squeeze it
        if emb.ndim == 2:
            emb = emb[0]

        if dim is None:
            dim = int(emb.shape[0])
            print(f"Detected embedding dimension: {dim}")

        embeddings.append(emb)

        # Periodic checkpoint so you don't lose progress if something crashes
        if (i + 1) % SAVE_EVERY == 0:
            arr_ckpt = np.stack(embeddings, axis=0)
            np.save(EMB_PATH, arr_ckpt)
            print(f"Checkpoint: saved {i + 1} embeddings to {EMB_PATH}")

    # 3) Final save
    emb_array = np.stack(embeddings, axis=0).astype("float32")
    np.save(EMB_PATH, emb_array)
    print(f"Saved embeddings: {EMB_PATH}, shape = {emb_array.shape}")

    with ID_PATH.open("w", encoding="utf-8") as f:
        json.dump(question_ids, f, ensure_ascii=False, indent=2)
    print(f"Saved question IDs: {ID_PATH}")

    meta = {
        "model_key": model_key,
        "max_chars": MAX_CHARS,
        "num_questions": len(questions),
        "embedding_dim": int(emb_array.shape[1]),
    }
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
