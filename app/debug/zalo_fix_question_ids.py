from pathlib import Path
import json
import numpy as np

from app.data_loader import load_zalo_questions

DATA_DIR = Path("data")
EMB_PATH = DATA_DIR / "zalo_question_embeddings.npy"
ID_PATH = DATA_DIR / "zalo_question_ids.json"


def main() -> None:
    # 1) Load existing embeddings
    emb = np.load(EMB_PATH)
    n_emb, dim = emb.shape
    print(f"Loaded embeddings: {EMB_PATH}, shape = {emb.shape}")

    # 2) Load questions
    questions = load_zalo_questions()
    print(f"Loaded {len(questions)} Zalo questions")

    if len(questions) != n_emb:
        raise RuntimeError(
            f"Mismatch: {len(questions)} questions vs {n_emb} embeddings. "
            "Do NOT proceed until this is consistent."
        )

    # 3) Build question_ids list using the 'id' field
    question_ids: list[str] = []
    for i, q in enumerate(questions):
        # your JSON uses "id"
        qid = q.get("id")
        if not qid:
            qid = f"zalo_q_{i}"
            print(f"[WARN] Missing id for index {i}, using {qid}")
        question_ids.append(qid)

    # 4) Save IDs
    with ID_PATH.open("w", encoding="utf-8") as f:
        json.dump(question_ids, f, ensure_ascii=False, indent=2)
    print(f"Rewrote question IDs: {ID_PATH}")


if __name__ == "__main__":
    main()
