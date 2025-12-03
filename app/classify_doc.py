import json
import re
import unicodedata
from pathlib import Path
import csv

def normalize_vn(text: str) -> str:
    """
    Lowercase + collapse whitespace.
    (If you want to strip accents later, you can extend this.)
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Patterns for classification
OPENING_AMENDING_PATTERNS = [
    "luật này sửa đổi, bổ sung",
    "thông tư này sửa đổi, bổ sung",
    "nghị định này sửa đổi, bổ sung",
    "sửa đổi, bổ sung một số điều của",
]

OPENING_BASE_PATTERNS = [
    "luật này quy định",
    "thông tư này hướng dẫn",
    "nghị định này quy định",
]

# Regex for lines like:
# "Điểm c khoản 1 Điều 2 được sửa đổi, bổ sung như sau:"
AMEND_LINE_REGEX = re.compile(
    r"điểm\s+[a-z]\s+khoản\s+\d+\s+điều\s+\d+.*được\s+sửa đổi,\s*bổ sung\s+như\s+sau",
    flags=re.IGNORECASE | re.UNICODE,
)


def classify_doc(articles):
    """
    Classify a Zalo document into:
        - 'amending'
        - 'base_or_guidance'
        - 'unknown'
    based on article 1 + amendment patterns + article count heuristic.
    """
    if not articles:
        return "unknown"

    num_articles = len(articles)

    # Get article 1 text (id "1" or first element)
    first_text = None
    for art in articles:
        if str(art.get("id")) == "1":
            first_text = art.get("text", "")
            break
    if first_text is None:
        first_text = articles[0].get("text", "")

    first_norm = normalize_vn(first_text)

    # ---- Rule A: opening sentence looks like amending law text ----
    for pattern in OPENING_AMENDING_PATTERNS:
        if pattern in first_norm:
            return "amending"

    # ---- Rule A2: opening looks like base/guidance ----
    base_or_guidance_guess = False
    for pattern in OPENING_BASE_PATTERNS:
        if first_norm.startswith(pattern):
            base_or_guidance_guess = True
            break

    # ---- Rule B: search amendment-style lines in all articles ----
    for art in articles:
        text = art.get("text", "")
        if not isinstance(text, str):
            continue
        if AMEND_LINE_REGEX.search(text):
            return "amending"

    # ---- Rule C: few-article doc + amendment-y phrasing in article 1 ----
    # e.g. 1–3 articles and clearly talking about sửa đổi, bổ sung
    if num_articles <= 4:
        if (
            "sửa đổi, bổ sung" in first_norm
            or "sửa đổi một số điều của" in first_norm
            or "sửa đổi, bổ sung một số điều của" in first_norm
            or "một số điều của luật" in first_norm
            or "một số điều của nghị định" in first_norm
            or "một số điều của thông tư" in first_norm
        ):
            return "amending"

    # If not amending but opening looks like base/guidance
    if base_or_guidance_guess:
        return "base_or_guidance"

    # Fallback: unknown (you can treat this as base for indexing, but log it)
    return "unknown"

def main(
    input_path: str = "data/zalo_corpus.json",
    output_json_path: str = "zalo_corpus_classified.json",
    output_csv_path: str = "zalo_doc_roles.csv",
):
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading Zalo corpus from: {input_file}")
    with input_file.open("r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError("Expected top-level JSON to be a list of documents.")

    stats = {
        "amending": 0,
        "base_or_guidance": 0,
        "unknown": 0,
    }

    # Add doc_role to each doc
    for doc in docs:
        articles = doc.get("articles", [])
        role = classify_doc(articles)
        doc["doc_role"] = role
        stats[role] = stats.get(role, 0) + 1

    # Save new JSON with doc_role
    out_json_file = Path(output_json_path)
    with out_json_file.open("w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Saved classified corpus to: {out_json_file}")

    # Save a small CSV summary (for quick inspection)
    out_csv_file = Path(output_csv_path)
    with out_csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "doc_role"])
        for doc in docs:
            writer.writerow([doc.get("id", ""), doc.get("doc_role", "")])

    print(f"Saved doc roles CSV to: {out_csv_file}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    # You can change paths here or pass them via CLI later if you extend the script
    main()