from app.config import get_connection

def main():
    conn = get_connection()
    cur = conn.cursor()

    # 1) Total rows
    cur.execute("SELECT COUNT(*) FROM article_embeddings;")
    total = cur.fetchone()[0]
    print("Total rows in article_embeddings:", total)

    # 2) Rows per model_key
    cur.execute("SELECT model_key, COUNT(*) FROM article_embeddings GROUP BY model_key;")
    print("Rows per model_key:")
    for model_key, count in cur.fetchall():
        print(f"  {model_key}: {count}")

    # 3) Rows by (source, model_key)
    cur.execute("""
        SELECT l.source, e.model_key, COUNT(*)
        FROM article_embeddings e
        JOIN articles a ON a.id = e.article_id
        JOIN laws l ON l.law_id = a.law_id
        GROUP BY l.source, e.model_key
        ORDER BY l.source, e.model_key;
    """)
    print("\nRows per (source, model_key):")
    for source, model_key, count in cur.fetchall():
        print(f"  source={source}, model_key={model_key}: {count}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
