from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_tfidf(train):
    """Fit TF-IDF and logistic regression on train only. Returns (vec, model)."""
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(train["text"])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train["label"])
    return vec, model


def build_pipeline():
    """Same model as train_tfidf, as one object so it can be saved and served."""
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
