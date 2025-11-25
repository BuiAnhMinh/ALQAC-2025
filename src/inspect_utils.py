# if __name__ == "__main__":
#     print("=== Basic sanity checks ===")
#     print("Number of law documents:", len(law_documents))
#     print("Example law doc:", law_documents[0]["law_id"], law_documents[0]["article_id"])
#     print()

#     # 1) Test tokenizer
#     sample_text = "Ủy ban nhân dân cấp tỉnh cấp giấy phép phim."
#     print("Sample text:", sample_text)
#     print("Tokenized:", underthesea_tokenizer(sample_text))
#     print()

#     # 2) Take one training question and run BM25 retrieval
#     example_train = train_data[0]
#     print("Question ID:", example_train["question_id"])
#     print("Question text:", example_train["text"])
#     print("Gold relevant articles:", example_train["relevant_articles"])
#     print()

#     top_k = 5
#     bm25_results = bm25_lexical_retrive(example_train["text"], top_k=top_k)

#     print(f"Top {top_k} BM25 results:")
#     for rank, cand in enumerate(bm25_results, start=1):
#         print(f"  Rank {rank}: law={cand['law_id']} article={cand['article_id']}")
#         print(f"    BM25 score = {cand['bm25_score']:.4f}")
#         print("    Text snippet:", cand["text"][:120].replace("\n", " "), "...")
#         print()
        
#     gold_pairs = {(ra["law_id"], ra["article_id"]) for ra in example_train["relevant_articles"]}
#     retrieved_pairs = {(c["law_id"], c["article_id"]) for c in bm25_results}

#     overlap = gold_pairs & retrieved_pairs
#     print("Gold pairs       :", gold_pairs)
#     print("Retrieved pairs  :", retrieved_pairs)
#     print("Overlap (hit)    :", overlap)

# def quick_bm25_smoke_test():
#     assert len(law_documents) > 0, "No law documents loaded!"
#     assert len(corpus_tokens) == len(law_documents), "Token count mismatch!"

#     q = train_data[0]
#     candidates = bm25_lexical_retrive(q["text"], top_k=5)

#     assert len(candidates) == 5, "bm25_lexical_retrive did not return top_k results"
#     for c in candidates:
#         assert "doc_id" in c and "bm25_score" in c, "Missing fields in candidate"

#     print("✅ quick_bm25_smoke_test passed.")

# if __name__ == "__main__":
#     quick_bm25_smoke_test()


