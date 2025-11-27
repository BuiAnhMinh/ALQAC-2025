import json
from pathlib import Path

from config import get_connection

# Paths inside the container (you already mount ./data -> /app/data)
LAW_PATH = Path("data/alqac25_law.json")
ZALO_PATH = Path("data/zalo_corpus.json")


def load_law_documents():
    """
    Build a flat list of law+article docs from ALQAC + Zalo JSON.
    Each doc: {law_id, article_id, text}
    """
    with LAW_PATH.open("r", encoding="utf-8") as f:
        law_data = json.load(f)

    with ZALO_PATH.open("r", encoding="utf-8") as f:
        zalo_law_data = json.load(f)

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

    print(
        f"Loaded {len(law_documents)} documents "
        f"(skipped alqac empty: {skipped_empty_alqac}, "
        f"zalo empty: {skipped_empty_zalo})"
    )

    return law_documents


SCHEMA_SQL = """
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
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_law_article UNIQUE (law_id, article_id)
);

CREATE INDEX IF NOT EXISTS idx_articles_law_fk ON articles(law_fk);
CREATE INDEX IF NOT EXISTS idx_articles_law_id_article_id
    ON articles(law_id, article_id);
"""


def main():
    docs = load_law_documents()
    total_docs = len(docs)

    conn = get_connection()
    cur = conn.cursor()

    try:
        # 1) Ensure tables exist
        cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Schema ensured (laws, articles).")

        # 2) Insert distinct laws (by law_id only)
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

        # 3) Build mapping law_id -> laws.id
        cur.execute("SELECT id, law_id FROM laws;")
        rows = cur.fetchall()
        law_id_to_pk = {law_id: pk for pk, law_id in rows}
        print(f"Loaded {len(law_id_to_pk)} law_id -> id mappings.")

        # Sanity check: every doc's law_id must exist in mapping
        missing_laws = {d["law_id"] for d in docs if d["law_id"] not in law_id_to_pk}
        if missing_laws:
            print("WARNING: some law_id not in laws table, examples:", list(missing_laws)[:5])

        # 4) Insert articles
        inserted_articles = 0
        print(f"Start inserting {total_docs} articles...")
        for d in docs:
            law_id = d["law_id"]
            if law_id not in law_id_to_pk:
                # Skip weird ones, but log once
                print(f"Skipping article with unknown law_id={law_id}")
                continue

            law_pk = law_id_to_pk[law_id]
            cur.execute(
                """
                INSERT INTO articles (law_fk, law_id, article_id, text)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (law_id, article_id) DO NOTHING;
                """,
                (law_pk, law_id, d["article_id"], d["text"]),
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
