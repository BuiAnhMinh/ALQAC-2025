import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import linear_kernel


train_path = "data/alqac25_train.json"
test_path  = "data/alqac25_private_test_Task_1.json"
law_path = "data/alqac25_law.json"

train_df = pd.read_json(train_path)
test_df  = pd.read_json(test_path)

# check for data

# print(train_df.columns)
# print(train_df.head(2))
# print(test_df.head(2))

# convert [{"law_id": "Luật Du lịch", "article_id": "39"},  {"law_id": "Luật Du lịch", "article_id": "40"}] to "Luật Du lịch|39", "Luật Du lịch|40" string

def article_to_string(relevant_articles):
  return relevant_articles["law_id"] + "|" + relevant_articles["article_id"]

y_labels = train_df["relevant_articles"].apply(
    lambda lst: [article_to_string(relevant_articles) for relevant_articles in lst]
)

# sanity check
# print(y_labels.iloc[10])

mlb = MultiLabelBinarizer()
y_all = mlb.fit_transform(y_labels)

# sanity check no 2
# print("Number of unique article labels:", len(mlb.classes_))

# train / validation split
x_train_text, x_val_text, y_train, y_val = train_test_split(
    train_df["text"],
    y_all,
    test_size=0.2,
    random_state=42
)

# TF-IDF
x_texts = train_df["text"].tolist()

vectorizer = TfidfVectorizer(max_features = 2000, ngram_range=(1, 2)) #limit vocabulary size and bigrams are more important (short phrases that are important in law text)

x_train = vectorizer.fit_transform(x_train_text)
x_val = vectorizer.transform(x_val_text)
#sanity check
# print("x_train shape:", x_train.shape)
# print("x_val shape:",   x_val.shape)

# multi strings classfier (One vs Rest Logistic Regression)

classifier = OneVsRestClassifier(LogisticRegression())
classifier.fit(x_train, y_train)

val_scores = classifier.predict_proba(x_val)

#sanity check
# print("Validation accuracy:", val_scores)

# Macro F Beta (B = 2)

def macro_fbeta_at_threshold(y_true_binary, scores, threshold,  beta=2):
  beta2 = beta ** 2
  y_pred_binary = (scores >= threshold).astype(int)
  n_samples = y_true_binary.shape[0]
  f_list = []

  for i in range(n_samples):
      y_t = y_true_binary[i]
      y_p = y_pred_binary[i]

      tp = np.logical_and(y_t == 1, y_p == 1).sum()
      fp = np.logical_and(y_t == 0, y_p == 1).sum()
      fn = np.logical_and(y_t == 1, y_p == 0).sum()

      # If nothing to evaluate, skip
      if tp == 0 and fp == 0 and fn == 0:
          continue

      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

      if precision == 0 and recall == 0:
          f = 0.0
      else:
          f = (1 + beta2) * precision * recall / (beta2 * precision + recall)

      f_list.append(f)

  return float(np.mean(f_list)) if f_list else 0.0

# grid search for the best threshold (sweet spot)
thresholds = np.linspace(0.01, 0.1, 10)
best_t, best_f2 = None, -1.0

for t in thresholds:
    f2 = macro_fbeta_at_threshold(y_val, val_scores, threshold=t, beta=2.0)
    print(f"threshold={t:.2f}, macro-F2={f2:.4f}")
    if f2 > best_f2:
        best_t, best_f2 = t, f2

#sanity check
# print("\nBest threshold:", best_t, "with macro-F2:", best_f2)

def apply_threshold_model_decides(scores, threshold, top_k = 1):
  n_samples, n_labels = scores.shape
  preds = np.zeros_like(scores, dtype=int)

  for i in range(n_samples):
    row = scores[i]
    mask = row >= threshold

    if mask.sum() > 0:
      # select how many models to keep
      preds[i, mask] = 1
    else:
      # fallback : take only the highest on
      best_idx = np.argmax(row)
      preds[i, best_idx] = 1

  return preds

# retrain on the same settings to ensure strongest model for the final prediction
tfidf_full = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

x_full = tfidf_full.fit_transform(train_df["text"])
y_full = y_all   # same labels

classifier_full = OneVsRestClassifier(
    LogisticRegression(max_iter=200)
)

classifier_full.fit(x_full, y_full)

# predict private test test using the best threshold

x_test = tfidf_full.transform(test_df["text"])
test_scores = classifier_full.predict_proba(x_test)

y_test_pred = apply_threshold_model_decides(
    test_scores,
    threshold=best_t
)

predicted_articles = []
for row in y_test_pred:
    arts = []
    for j, flag in enumerate(row):
        if flag:
            label_str = mlb.classes_[j]              # "law_id|article_id"
            law_id, article_id = label_str.split("|", 1)
            arts.append({"law_id": law_id, "article_id": article_id})
    predicted_articles.append(arts)

#sanity check
# print("Example prediction for first test question:")
# print(predicted_articles[0])



output = []
for qid, ras in zip(test_df["question_id"], predicted_articles):
    output.append({
        "question_id": qid,
        "relevant_articles": ras
    })

out_path = "output/alqac25_task1_predictions.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nSaved predictions to", out_path)