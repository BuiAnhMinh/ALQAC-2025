# data_loader.py
import json
from typing import List, Dict, Any
from pathlib import Path

from config import LAW_PATH, ZALO_LAW_PATH, TRAIN_PATH, TEST_PATH


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_law_documents() -> List[Dict[str, Any]]:
    """
    Build a flat list of law+article docs from ALQAC + Zalo JSON.
    Each doc: {doc_id, law_id, article_id, text}
    doc_id is assigned sequentially to match embedding rows.
    """
    law_data = load_json(LAW_PATH)
    zalo_law_data = load_json(ZALO_LAW_PATH)

    law_documents = []
    skipped_empty_alqac = 0
    skipped_empty_zalo = 0

    # ALQAC laws
    for law in law_data:
        law_id = law["id"]
        for artc in law["articles"]:
            raw_text = artc.get("text", "")
            if raw_text is None:
                raw_text = ""
            text = str(raw_text).strip()
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

    # Zalo laws
    for law in zalo_law_data:
        law_id = law["id"]
        for artc in law["articles"]:
            raw_text = artc.get("text", "")
            if raw_text is None:
                raw_text = ""
            text = str(raw_text).strip()
            if len(text) == 0:
                skipped_empty_zalo += 1
                continue

            law_documents.append(
                {
                    "law_id": law_id,
                    "article_id": artc["id"],
                    "text": text,
                }
            )

    # Assign doc_id sequentially
    for i, d in enumerate(law_documents):
        d["doc_id"] = i

    print(
        f"Loaded {len(law_documents)} documents "
        f"(skipped alqac empty: {skipped_empty_alqac}, "
        f"zalo empty: {skipped_empty_zalo})"
    )

    return law_documents


def load_train_data():
    return load_json(TRAIN_PATH)


def load_test_data():
    return load_json(TEST_PATH)
