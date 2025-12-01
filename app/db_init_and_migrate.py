from typing import Dict, List, Set

from underthesea import word_tokenize
from pathlib import Path
from app.config import get_connection, STOPWORDS_PATH
from app.data_loader import load_law_documents

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS laws (
    id          SERIAL PRIMARY KEY,
    law_id      TEXT NOT NULL UNIQUE,
    title       TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    law_fk      INTEGER NOT NULL REFERENCES laws(id) ON DELETE CASCADE,
    law_id      TEXT NOT NULL,
    article_id  TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(1536),
    tokens      TEXT[],
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_law_article UNIQUE (law_id, article_id)
);

CREATE INDEX IF NOT EXISTS idx_articles_law_fk
    ON articles(law_fk);

CREATE INDEX IF NOT EXISTS idx_articles_law_id_article_id
    ON articles(law_id, article_id);
"""
LEGAL_WHITELIST = {"phải", "không", "được", "cấm", "trừ", "khi", "nếu", "vì"}


def _load_stopwords(path: Path) -> Set[str]:
    stopwords: Set[str] = set()
    if not path.exists():
        return stopwords
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                stopwords.add(w)
    return stopwords

STOPWORDS = _load_stopwords(Path(STOPWORDS_PATH)) - LEGAL_WHITELIST


def _tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    tok_str = word_tokenize(text, format="text")
    tokens = tok_str.split()
    return [t for t in tokens if t not in STOPWORDS]


def main():
    docs = load_law_documents()
    total_docs = len(docs)
    print(f"Loaded {total_docs} docs from loader.")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Schema ensured (laws, articles).")

        # Insert distinct laws
        seen = set()
        for d in docs:
            law_id = d["law_id"]
            if law_id in seen:
                continue
            seen.add(law_id)
            cur.execute(
                """
                INSERT INTO laws (law_id)
                VALUES (%s)
                ON CONFLICT (law_id) DO NOTHING;
                """,
                (law_id,),
            )
        conn.commit()
        print(f"Inserted/ensured {len(seen)} laws.")

        # Build mapping law_id -> laws.id
        cur.execute("SELECT id, law_id FROM laws;")
        rows = cur.fetchall()
        law_id_to_pk: Dict[str, int] = {law_id: pk for pk, law_id in rows}
        print(f"Loaded {len(law_id_to_pk)} law_id -> id mappings.")

        missing_laws = {d["law_id"] for d in docs if d["law_id"] not in law_id_to_pk}
        if missing_laws:
            print("WARNING: some law_id not in laws table, examples:", list(missing_laws)[:5])

        # Insert articles with tokens
        inserted_articles = 0
        print(f"Start inserting {total_docs} articles...")
        for d in docs:
            law_id = d["law_id"]
            if law_id not in law_id_to_pk:
                print(f"Skipping article with unknown law_id={law_id}")
                continue

            law_pk = law_id_to_pk[law_id]
            text = d["text"] or ""
            tokens = _tokenize_text(text)

            cur.execute(
                """
                INSERT INTO articles (law_fk, law_id, article_id, text, tokens)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (law_id, article_id) DO UPDATE
                SET text = EXCLUDED.text,
                    tokens = EXCLUDED.tokens;
                """,
                (law_pk, law_id, d["article_id"], text, tokens),
            )
            inserted_articles += 1

            if inserted_articles % 5000 == 0:
                print(f"Inserted {inserted_articles}/{total_docs} articles so far...")
                conn.commit()

        conn.commit()
        print(f"Finished inserting ~{inserted_articles} articles (out of {total_docs}).")

    except Exception as e:
        conn.rollback()
        print("ERROR during migration:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()
