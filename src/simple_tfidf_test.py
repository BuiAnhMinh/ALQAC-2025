from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# ======================================
# 1. Build a simple TF-IDF + LogReg model
# ======================================

def build_tfidf_logreg():
    """
    Returns a sklearn Pipeline: TF-IDF → Logistic Regression
    """
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300)),
    ])
    return model

# ======================================
# 2. Train the model on simple sentences
# ======================================

def train_on_tiny_dataset():
    X = [
        "tôi có con mèo", 
        "con mèo dễ thương", 
        "tôi nuôi mèo",
        "tôi có con chó",
        "con chó rất thông minh",
        "tôi nuôi chó"
    ]

    # Labels: 1 = mèo, 0 = chó
    y = [1,1,1, 0,0,0]

    model = build_tfidf_logreg()
    model.fit(X, y)
    return model

# ======================================
# 3. Test prediction
# ======================================

def test_prediction(model):
    test_sentences = [
        "con mèo ngủ trên bàn", 
        "con chó chạy rất nhanh",
        "tôi muốn nuôi mèo",
        "tôi muốn nuôi chó"
    ]

    preds = model.predict_proba(test_sentences)

    for sentence, prob in zip(test_sentences, preds):
        print("Sentence:", sentence)
        print("→ Probability mèo:", prob[1])
        print("→ Probability chó:", prob[0])
        print("------------------------------")


if __name__ == "__main__":
    model = train_on_tiny_dataset()
    test_prediction(model)
