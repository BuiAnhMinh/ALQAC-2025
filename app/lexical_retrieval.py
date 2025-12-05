from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from app.config import STOPWORDS_PATH
from app.data_loader import load_law_documents
from app.semantic_retrieval import semantic_retrieve
from app.embedding import DEFAULT_MODEL_KEY

# ---------- Stopwords & tokenizer ----------
def _load_stopwords(path: str) -> set[str]:
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


STOPWORDS: set[str] = _load_stopwords(STOPWORDS_PATH)
LEGAL_WHITELIST: set[str] = {"văn_bản", "pháp_luật", "nghị_định", "thông_tư"}


def underthesea_tokenizer(text: str) -> List[str]:
    tokens = word_tokenize(text, format="text").split()
    cleaned = []
    for t in tokens:
        if t in LEGAL_WHITELIST:
            cleaned.append(t)
        elif t not in STOPWORDS:
            cleaned.append(t)
    return cleaned


# ---------- Build BM25 index over Zalo articles ----------
_DOCS: List[Dict[str, Any]] = []
_CORPUS_TOKENS: List[List[str]] = []
_BM25: BM25Okapi | None = None


def _build_bm25_index() -> None:
    """
    Build BM25Okapi index for ALL Zalo articles (not just amending ones).
    Uses load_law_documents() for tokens/text.
    """
    global _DOCS, _CORPUS_TOKENS, _BM25

    if _BM25 is not None:
        return

    all_docs = load_law_documents()
    docs_zalo = [d for d in all_docs if d.get("source") == "zalo" and not d.get("is_amending_article", False)]

    _DOCS = docs_zalo
    _CORPUS_TOKENS = []
    for d in docs_zalo:
        # if you already have tokens in d["tokens"], you can reuse them
        if "tokens" in d and isinstance(d["tokens"], list):
            tokens = d["tokens"]
        else:
            tokens = underthesea_tokenizer(d["text"])
        _CORPUS_TOKENS.append(tokens)

    _BM25 = BM25Okapi(_CORPUS_TOKENS)
    print(f"BM25 index built for {len(_DOCS)} Zalo articles.")


# ---------- BM25-only retrieval ----------
def bm25_pure(
    question_text: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Pure BM25 retrieval on Zalo articles.
    Returns top_k docs with bm25_score.
    """
    _build_bm25_index()
    assert _BM25 is not None

    q_tokens = underthesea_tokenizer(question_text)
    scores = _BM25.get_scores(q_tokens)

    # Get indices of top_k scores
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    top_idx_scores = idx_scores[:top_k]

    results: List[Dict[str, Any]] = []
    for idx, score in top_idx_scores:
        d = _DOCS[idx]
        results.append(
            {
                "doc_id": d["doc_id"],
                "law_id": d["law_id"],
                "article_id": d["article_id"],
                "text": d["text"],
                "tokens": d.get("tokens"),
                "bm25_score": float(score),
            }
        )

    return results


# ---------- BM25 rerank within semantic candidates (Hybrid) ----------
def bm25_lexical_rerank(
    question_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Given a list of candidate articles (from semantic search),
    rerank them using BM25 localized on these candidates.
    """
    # Build small BM25 model on candidate texts
    corpus_tokens = []
    for c in candidates:
        if "tokens" in c and isinstance(c["tokens"], list):
            toks = c["tokens"]
        else:
            toks = underthesea_tokenizer(c["text"])
        corpus_tokens.append(toks)

    bm25_local = BM25Okapi(corpus_tokens)
    q_tokens = underthesea_tokenizer(question_text)
    scores = bm25_local.get_scores(q_tokens)

    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    top_idx_scores = idx_scores[:top_k]

    results: List[Dict[str, Any]] = []
    for idx, score in top_idx_scores:
        c = candidates[idx].copy()
        c["bm25_score"] = float(score)
        results.append(c)

    return results


def semantic_then_lexical(
    question_text: str,
    semantic_top_k: int = 200,
    lexical_top_k: int = 10,
    model_key: str = DEFAULT_MODEL_KEY,
) -> Dict[str, Any]:
    """
    Hybrid: semantic top semantic_top_k -> BM25 rerank lexical_top_k.
    """
    semantic_candidates = semantic_retrieve(
        question_text=question_text,
        top_k=semantic_top_k,
        model_key=model_key,
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


def bm25_hybrid_retrieve(
    question_text: str,
    top_k: int = 10,
    semantic_top_k: int = 200,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper: semantic top semantic_top_k -> BM25 top_k.
    """
    combined = semantic_then_lexical(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=top_k,
        model_key=model_key,
    )
    return combined["lexical_results"]
