from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from app.config import STOPWORDS_PATH
from app.data_loader import load_law_documents
from app.semantic_retrieval import semantic_retrieve
from app.embedding.embedding import DEFAULT_MODEL_KEY

# ---------- Stopwords & tokenizer ----------
def _load_stopwords(path: str) -> set[str]:
    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords


STOPWORDS = _load_stopwords(STOPWORDS_PATH)


def underthesea_tokenizer(text: str) -> List[str]:
    """
    Tokenize Vietnamese text using underthesea and drop stopwords.
    """
    tokens = word_tokenize(text, format="text").split()
    return [t for t in tokens if t not in STOPWORDS]


# ---------- Load all docs & pre-tokenize ----------
_ALL_DOCS = load_law_documents()

# Filter to Zalo + non-amending for BM25 (same as semantic side)
_ZALO_DOCS = [
    d for d in _ALL_DOCS
    if d.get("source") == "zalo" and not d.get("is_amending_article", False)
]

ZALO_TEXTS: List[str] = [d["text"] for d in _ZALO_DOCS]
ZALO_TOKENS: List[List[str]] = [underthesea_tokenizer(t) for t in ZALO_TEXTS]

# Build a global BM25 model once at import time
BM25_MODEL = BM25Okapi(ZALO_TOKENS)


# ---------- BM25-only retrieval ----------
def bm25_pure(question_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Pure BM25 over Zalo non-amending articles.
    """
    q_tokens = underthesea_tokenizer(question_text)
    scores = BM25_MODEL.get_scores(q_tokens)

    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    top_idx_scores = idx_scores[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(top_idx_scores, start=1):
        d = _ZALO_DOCS[idx]
        results.append(
            {
                "rank": rank,
                "bm25_score": float(score),
                "doc_id": d["doc_id"],
                "law_id": d["law_id"],
                "article_id": d["article_id"],
                "text": d["text"],
                "tokens": ZALO_TOKENS[idx],
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
        c["bm25_local_score"] = float(score)
        results.append(c)

    return results

def bm25_db_retrieve(question_text: str, top_k: int = 200, source: str = "zalo"):
    sql = """
    SELECT
      a.id AS doc_id,
      a.law_id,
      a.article_id,
      lexical_text <@> to_bm25query(%(q)s, 'articles_content_idx') AS bm25_score
    FROM articles a
    JOIN laws l ON l.law_id = a.law_id
    WHERE l.source = %(source)s
      AND COALESCE(a.is_amending_article, FALSE) = FALSE
      AND lexical_text IS NOT NULL
    ORDER BY bm25_score
    LIMIT %(k)s;
    """

    with conn.cursor() as cur:
        cur.execute(sql, {"q": question_text, "source": source, "k": top_k})
        rows = cur.fetchall()

    results = [
        {
            "doc_id": row[0],
            "law_id": row[1],
            "article_id": row[2],
            "bm25_score": float(row[3]),
        }
        for row in rows
    ]
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


def semantic_then_lexical_top_k(
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

def bm25_hybrid_retrieve(
    question_text: str,
    top_k: int = 10,
    semantic_top_k: int = 200,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper expected by app.retrieval.

    It runs:
      1) semantic retrieval (top semantic_top_k)
      2) BM25 rerank within those candidates to get top_k
    """
    combined = semantic_then_lexical(
        question_text=question_text,
        semantic_top_k=semantic_top_k,
        lexical_top_k=top_k,
        model_key=model_key,
    )
    return combined["lexical_results"]