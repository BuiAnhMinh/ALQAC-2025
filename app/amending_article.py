from app.config import get_connection

# Phrases that indicate an article is amending other provisions.
AMEND_PATTERNS = [
    "được sửa đổi, bổ sung như sau",
    "sửa đổi, bổ sung khoản",
    "sửa đổi, bổ sung điều",
    "bổ sung khoản",
    "bổ sung điều",
    "bãi bỏ điều",
    "bãi bỏ khoản",
    "thay thế cụm từ",
]


def main():
    """
    Set is_amending_article = TRUE when article text contains one of the
    predefined amendment phrases (case/whitespace-insensitive).
    """
    conn = get_connection()

    try:
        with conn:
            with conn.cursor() as cur:
                # Reset all flags before re-marking.
                cur.execute("UPDATE articles SET is_amending_article = FALSE;")

                normalized = "regexp_replace(lower(text), '\\\\s+', ' ', 'g')"
                like_clauses = [f"{normalized} LIKE %s" for _ in AMEND_PATTERNS]
                sql = (
                    "UPDATE articles "
                    "SET is_amending_article = TRUE "
                    f"WHERE {' OR '.join(like_clauses)};"
                )
                params = [f"%{p}%" for p in AMEND_PATTERNS]
                cur.execute(sql, params)
                print(f"Marked {cur.rowcount} articles as amending.")

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM articles WHERE is_amending_article = TRUE;"
                )
                total = cur.fetchone()[0]
                print(f"Total amending articles: {total}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
