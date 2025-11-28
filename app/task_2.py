# task2_qa.py
"""
Task 2: answer generation with OpenAI LLM, using Task 1 retriever.

Usage examples:

  # Run a rough evaluation on some training questions (True/False + MCQ)
  python task2_qa.py --eval-samples 50

  # Generate answers for all training questions and write to JSON
  python task2_qa.py --predict-train data/alqac25_task2_train_predictions.json

  # Ask a custom question interactively
  python task2_qa.py --interactive
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from config import LLM_MODEL, get_client
from data_loader import load_train_data
from app.retrieval import (
    law_documents,
    retrieve_and_rerank_with_qemb,
)
from app.embedding import embed_text

# Build a quick lookup: (law_id, article_id) -> full article text (if you ever need it)
ARTICLE_LOOKUP: Dict[Tuple[str, str], Dict[str, Any]] = {
    (d["law_id"], d["article_id"]): d for d in law_documents
}

client = get_client()


def build_context_from_articles(articles: List[Dict[str, Any]]) -> str:
    """
    Turn a list of article dicts into a single context string for the prompt.
    """
    parts = []
    for art in articles:
        law_id = art["law_id"]
        article_id = art["article_id"]
        text = art["text"].strip()
        parts.append(f"Luật: {law_id} - Điều: {article_id}\n{text}")
    return "\n\n".join(parts)


def retrieve_context_for_question(
    question_text: str,
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Run Task 1 retriever for a single raw question string and return the top_k_final
    article dicts (with law_id, article_id, text...).
    """
    q_emb = embed_text(question_text)
    ranked = retrieve_and_rerank_with_qemb(
        question_text,
        q_emb,
        top_k_lexical=top_k_lexical,
        top_k_final=top_k_final,
        alpha=alpha,
        logreg_model=None,  # you can plug the trained LogReg here later
    )
    return ranked


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Helper to call the chat model and return the stripped text.
    """
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def build_tf_prompt(question_text: str, context: str) -> Tuple[str, str]:
    system = (
        "Bạn là trợ lý pháp lý. Bạn sẽ được cung cấp câu hỏi và một số điều luật liên quan. "
        "Dựa hoàn toàn vào nội dung điều luật, hãy trả lời câu hỏi Đúng/Sai một cách chính xác. "
        "Chỉ trả lời đúng một từ: 'Đúng' hoặc 'Sai'. Không giải thích thêm."
    )
    user = (
        f"Câu hỏi (Đúng/Sai):\n{question_text}\n\n"
        f"Các điều luật liên quan:\n{context}\n\n"
        "Trả lời: Đúng hay Sai?"
    )
    return system, user


def build_mc_prompt(question_text: str, context: str) -> Tuple[str, str]:
    system = (
        "Bạn là trợ lý pháp lý. Bạn sẽ được cung cấp một câu hỏi trắc nghiệm và một số điều luật liên quan. "
        "Dựa vào điều luật, hãy chọn đáp án chính xác nhất trong các phương án A, B, C hoặc D. "
        "Chỉ trả lời bằng đúng một chữ cái trong tập {A, B, C, D}. Không giải thích thêm."
    )
    user = (
        f"Câu hỏi (Trắc nghiệm):\n{question_text}\n\n"
        f"Các điều luật liên quan:\n{context}\n\n"
        "Trả lời: A, B, C hay D?"
    )
    return system, user


def build_free_prompt(question_text: str, context: str) -> Tuple[str, str]:
    system = (
        "Bạn là trợ lý pháp lý. Bạn sẽ được cung cấp câu hỏi tự luận và một số điều luật liên quan. "
        "Dựa vào điều luật, hãy trả lời ngắn gọn, chính xác, dùng câu tiếng Việt tự nhiên."
    )
    user = (
        f"Câu hỏi (Tự luận):\n{question_text}\n\n"
        f"Các điều luật liên quan:\n{context}"
    )
    return system, user


def normalise_tf_answer(raw: str) -> str:
    s = raw.strip()
    # Take only first token and normalise common variants
    first = s.split()[0]
    first_norm = first.lower()
    if "đ" in first_norm:  # đúng
        return "Đúng"
    if "sai" in first_norm:
        return "Sai"
    # Fallback: just return original first token
    return first


def normalise_mc_answer(raw: str) -> str:
    s = raw.strip().upper()
    # Take first character that looks like an option letter
    for ch in s:
        if ch in {"A", "B", "C", "D"}:
            return ch
    return s[0] if s else ""


def answer_question_with_llm(
    q: Dict[str, Any],
    top_k_lexical: int = 200,
    top_k_final: int = 5,
    alpha: float = 0.4,
) -> str:
    """
    Given a training-style question dict (with 'question_type', 'text', 'answer'),
    run retrieval + LLM and return the predicted answer string.
    """
    qtype = q["question_type"]
    question_text = q["text"]

    # Retrieve context articles (you could also experiment with using gold relevant_articles)
    ranked_articles = retrieve_context_for_question(
        question_text,
        top_k_lexical=top_k_lexical,
        top_k_final=top_k_final,
        alpha=alpha,
    )
    context = build_context_from_articles(ranked_articles)

    if qtype == "Đúng/Sai":
        system, user = build_tf_prompt(question_text, context)
        raw = call_llm(system, user)
        return normalise_tf_answer(raw)
    elif qtype == "Trắc nghiệm":
        system, user = build_mc_prompt(question_text, context)
        raw = call_llm(system, user)
        return normalise_mc_answer(raw)
    else:
        # "Tự luận" or anything else
        system, user = build_free_prompt(question_text, context)
        raw = call_llm(system, user)
        return raw.strip()


def evaluate_on_train(
    max_samples: int | None = None,
) -> None:
    """
    Very rough evaluation on the labelled training set, computing accuracy for
    Đúng/Sai và Trắc nghiệm.

    This will call the LLM many times, so you may want to limit max_samples.
    """
    train_data = load_train_data()
    n = len(train_data)
    print(f"Loaded {n} training questions.")

    correct_tf = total_tf = 0
    correct_mc = total_mc = 0

    for idx, q in enumerate(train_data):
        if max_samples is not None and idx >= max_samples:
            break

        qtype = q["question_type"]
        if qtype not in {"Đúng/Sai", "Trắc nghiệm"}:
            continue

        gold = str(q["answer"]).strip()
        pred = answer_question_with_llm(q)

        if qtype == "Đúng/Sai":
            total_tf += 1
            if pred == gold:
                correct_tf += 1
        elif qtype == "Trắc nghiệm":
            total_mc += 1
            if pred == gold:
                correct_mc += 1

        print(f"[{idx+1}] {qtype} | gold={gold!r}, pred={pred!r}")

    if total_tf:
        print(f"\nĐúng/Sai accuracy: {correct_tf}/{total_tf} = {correct_tf/total_tf:.3f}")
    if total_mc:
        print(f"Trắc nghiệm accuracy: {correct_mc}/{total_mc} = {correct_mc/total_mc:.3f}")


def predict_for_train(out_path: Path) -> None:
    """
    Generate a Task 2-style prediction file for the training questions.
    You can adapt this to run on the official Task 2 test set when you have it.
    """
    train_data = load_train_data()
    preds: List[Dict[str, Any]] = []

    for q in train_data:
        answer = answer_question_with_llm(q)
        preds.append(
            {
                "question_id": q["question_id"],
                "answer": answer,
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print("Saved Task 2 predictions to", out_path)


def interactive_cli() -> None:
    print("Interactive legal QA (Task 2-style). Ctrl+C to exit.")
    while True:
        try:
            question = input("\nNhập câu hỏi pháp lý: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        ranked_articles = retrieve_context_for_question(question)
        context = build_context_from_articles(ranked_articles)
        system, user = build_free_prompt(question, context)
        answer = call_llm(system, user)

        print("\n--- Top articles ---")
        for idx, art in enumerate(ranked_articles, start=1):
            preview = art["text"].replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"[{idx}] Luật {art['law_id']} - Điều {art['article_id']}")
            print(f"      {preview}")

        print("\n--- LLM answer ---")
        print(answer)


def main():
    parser = argparse.ArgumentParser(description="Task 2: LLM-based legal question answering.")
    parser.add_argument(
        "--eval-samples",
        type=int,
        help="Run a small evaluation on the training set (limit number of questions).",
    )
    parser.add_argument(
        "--predict-train",
        type=str,
        help="Write Task 2 predictions for all training questions to the given JSON path.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive Q&A mode.",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_cli()
    elif args.eval_samples is not None:
        evaluate_on_train(max_samples=args.eval_samples)
    elif args.predict_train:
        out_path = Path(args.predict_train)
        predict_for_train(out_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
