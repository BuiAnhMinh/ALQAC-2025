import json

def main():
    # Use the same file you used before
    with open("data/zalo_question.json", "r", encoding="utf-8") as f:
        zalo_questions = json.load(f)

    print(f"Total questions: {len(zalo_questions)}")
    print("First 3 question objects and their keys:\n")

    for i, q in enumerate(zalo_questions[:3]):
        print(f"=== Question {i} ===")
        print("Keys:", list(q.keys()))
        print("Raw object:", q)
        print()

if __name__ == "__main__":
    main()
