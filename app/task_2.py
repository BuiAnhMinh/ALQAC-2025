"""
Task 2: answer generation with OpenAI LLM, using Task 1 retriever.

Usage examples:

  # Ask custom questions interactively
  python -m app.task_2 --interactive
"""
import argparse
from typing import Dict, Any, List, Tuple

from app.config import LLM_MODEL, get_client
from app.data_loader import load_law_documents
from app.retrieval import (
    retrieve_and_rerank_with_qemb,
)
from app.embedding import embed_text

# Build a quick lookup: (law_id, article_id) -> full article metadata
ARTICLE_LOOKUP: Dict[Tuple[str, str], Dict[str, Any]] = {
    (d["law_id"], d["article_id"]): d for d in load_law_documents()
}


def build_context_from_articles(articles: List[Dict[str, Any]]) -> str:
    """
    Turn retrieved articles into a context block for the LLM.
    """
    parts = []
    for a in articles:
        law_id = a["law_id"]
        article_id = a["article_id"]
        text = a["text"]
        parts.append(f"Luật: {law_id}, Điều: {article_id}\n{text}\n")
    return "\n".join(parts)


def call_llm(question: str, articles: List[Dict[str, Any]]) -> str:
    """
    Prompt the LLM with retrieved legal articles + question,
    get a Vietnamese legal-style answer.
    """
    client = get_client()
    context = build_context_from_articles(articles)

    system_prompt = (
        "Bạn là một trợ lý pháp lý tiếng Việt. "
        "Bạn phải trả lời dựa trên các điều luật được cung cấp, "
        "trích dẫn rõ Luật và Điều tương ứng (ví dụ: Điều 34, Luật Giao dịch điện tử). "
        "Nếu không đủ thông tin, hãy nói rõ là không đủ căn cứ và khuyên người dùng "
        "liên hệ luật sư hoặc cơ quan có thẩm quyền.\n"
    )

    user_prompt = (
        f"Câu hỏi:\n{question}\n\n"
        f"Các điều luật liên quan (trích dẫn):\n{context}\n\n"
        "Hãy trả lời ngắn gọn, rõ ràng, trích dẫn đúng điều luật."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def interactive_loop():
    print("=== Legal QA (Task 2 demo) ===")
    print("Nhập câu hỏi (Enter để thoát).")
    while True:
        try:
            q = input("\nCâu hỏi: ").strip()
        except EOFError:
            break

        if not q:
            break

        # Embed question
        q_emb = embed_text(q)

        # Retrieve top articles (logreg_model can be added later if you persist it)
        articles = retrieve_and_rerank_with_qemb(
            question_text=q,
            question_embedding=q_emb,
            top_k_lexical=200,
            top_k_final=5,
            alpha=0.6,
            logreg_model=None,
        )

        answer = call_llm(q, articles)
        print("\n--- Trả lời ---")
        print(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run interactive QA loop")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop()
    else:
        print("Use --interactive to start an interactive QA session.")


if __name__ == "__main__":
    main()
