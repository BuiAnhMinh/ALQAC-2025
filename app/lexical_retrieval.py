from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from app.config import STOPWORDS_PATH
from app.data_loader import load_law_documents
from app.semantic_retrieval import semantic_retrieve

# ---------- Stopwords & tokenizer ----------
def _load_stopwords(path: str) -> set[str]:
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


STOPWORDS = _load_stopwords(str(STOPWORDS_PATH))
LEGAL_WHITELIST = {"phải", "không", "được", "cấm", "trừ", "khi", "nếu", "vì"}
STOPWORDS = STOPWORDS - LEGAL_WHITELIST


def underthesea_tokenizer(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    tokenized = word_tokenize(text, format="text")
    tokens = tokenized.lower().split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# Map (law_id, article_id) -> doc_id for consistent IDs across JSON/DB
LAW_ART_TO_DOCID: Dict[tuple[str, str], int] = {
    (d["law_id"], d["article_id"]): d["doc_id"] for d in load_law_documents()
}


def _get_tokens_from_candidate(cand: Dict[str, Any]) -> List[str]:
    tokens = cand.get("tokens")
    if isinstance(tokens, list) and all(isinstance(t, str) for t in tokens):
        return [t.lower() for t in tokens]
    return underthesea_tokenizer(cand.get("text", ""))


def bm25_lexical_rerank(
    question_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    BM25-Okapi rerank of a provided candidate list (typically semantic top-k).
    Uses pre-tokenized article tokens when available to avoid re-tokenizing.
    """
    if not candidates:
        return []

    q_tokens = underthesea_tokenizer(question_text)
    if not q_tokens:
        return []

    corpus_tokens = [_get_tokens_from_candidate(c) for c in candidates]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(q_tokens)

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_idx = ranked_idx[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_idx:
        base = dict(candidates[idx])
        base["bm25_score"] = float(scores[idx])
        results.append(base)

    return results


def semantic_then_lexical(
    question_text: str,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Two-stage retrieval:
      1) semantic_retrieve to get top semantic_top_k Zalo amending articles.
      2) BM25-Okapi rerank within those candidates to get top lexical_top_k.
    """
    semantic_candidates = semantic_retrieve(
        question_text=question_text,
        top_k=semantic_top_k,
    )

    lexical_results = bm25_lexical_rerank(
        question_text=question_text,
        candidates=semantic_candidates,
        top_k=lexical_top_k,
    )

    return {
        "semantic_candidates": semantic_candidates,
        "lexical_results": lexical_results,
    }


def bm25_lexical_retrieve(
    question_text: str,
    top_k: int = 10,
    semantic_top_k: int = 200,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper: semantic top semantic_top_k -> BM25-Okapi top_k.
    """
    combined = semantic_then_lexical(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=top_k,
    )
    return combined["lexical_results"]
