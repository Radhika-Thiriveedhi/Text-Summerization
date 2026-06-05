"""Summarization helpers with a safe fallback.

Primary path: use `transformers` pipeline when available.
Fallback path: a simple extractive summarizer based on TF-IDF sentence scoring
so the Flask app can run even without `torch` installed.
"""

import math
import re
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HAS_TRANSFORMERS = True
try:
    from transformers import pipeline
except Exception:
    HAS_TRANSFORMERS = False

_summarizer = None


def _get_summarizer():
    global _summarizer, HAS_TRANSFORMERS
    if not HAS_TRANSFORMERS:
        return None
    if _summarizer is None:
        try:
            # prefer a smaller model if available; device=-1 forces CPU
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        except Exception:
            # fallback to extractive summarization if the transformers pipeline is unavailable
            HAS_TRANSFORMERS = False
            return None
    return _summarizer


def _split_sentences(text: str) -> List[str]:
    # very small sentence splitter
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extractive_summary(text: str, max_length: int, min_length: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""

    # compute TF-IDF over sentences
    try:
        vect = TfidfVectorizer(stop_words='english')
        X = vect.fit_transform(sentences)
        scores = np.asarray(X.sum(axis=1)).ravel()
    except Exception:
        # fallback: score by sentence length
        scores = np.array([len(s) for s in sentences], dtype=float)

    # rank sentences by score
    ranked_idx = np.argsort(-scores)

    # select top sentences until reaching approx max_length tokens (words)
    selected = []
    total_words = 0
    for idx in ranked_idx:
        sent = sentences[int(idx)]
        sent_words = len(sent.split())
        if total_words + sent_words <= max_length or not selected:
            selected.append((int(idx), sent))
            total_words += sent_words
        if total_words >= min_length:
            break

    # preserve original order
    selected_sorted = [s for i, s in sorted(selected, key=lambda x: x[0])]
    return " ".join(selected_sorted)


def summarize(text: str, max_length: int = 120, min_length: int = 30) -> str:
    """Generate a summary for `text`.

    If `transformers` is available, uses a model pipeline. Otherwise falls back to
    a lightweight extractive summarizer using TF-IDF sentence scoring.
    The function also appends the input and summary to `data/logs.csv` when possible.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # tiny preprocessing example with pandas/numpy
    try:
        s = pd.Series([text])
        arr = s.values.astype(str)
    except Exception:
        arr = np.array([str(text)])

    # attempt to use transformers pipeline
    summarizer = _get_summarizer()
    summary = ""
    if summarizer is not None:
        try:
            # pipeline expects a string
            result = summarizer(arr[0], max_length=max_length, min_length=min_length)
            summary = result[0].get("summary_text", "").strip()
        except Exception:
            # fall through to extractive
            summary = _extractive_summary(arr[0], max_length, min_length)
    else:
        summary = _extractive_summary(arr[0], max_length, min_length)

    # log the input and summary to CSV (append mode)
    try:
        df = pd.DataFrame({"input": [text], "summary": [summary]})
        df.to_csv("data/logs.csv", mode='a', header=False, index=False)
    except Exception:
        pass

    return summary
