"""TF-IDF + logistic regression baseline (scikit-learn), one model per head."""

from __future__ import annotations

from .data import Example


def train_baseline(train: list[Example]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = [e.text for e in train]
    heads = {}
    for head in ("category", "priority"):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"),
        )
        pipe.fit(texts, [getattr(e, head) for e in train])
        heads[head] = pipe
    return heads
