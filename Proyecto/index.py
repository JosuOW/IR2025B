"""
index.py
Funciones para construir índice invertido, tokenización y utilidades.
"""
import re
from collections import defaultdict, Counter
import math

DEFAULT_STOPWORDS = {
    "the","and","is","in","it","of","to","a","an","that","this","for","on","with","as","are","was","were","be","by","from","or","at"
}

def simple_tokenize(text, lowercase=True, remove_punct=True, stopwords=None):
    if text is None:
        return []
    if lowercase:
        text = text.lower()
    if remove_punct:
        # keep internal apostrophes and hyphens
        text = re.sub(r"[^\w'\- ]+", " ", text)
    tokens = [t for t in text.split() if t]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens

def build_inverted_index(docs, tokenizer=None, stopwords=None):
    """
    docs: list of strings (documents)
    returns:
      inverted_index: {term: {doc_id: freq, ...}, ...}
      doc_lengths: {doc_id: num_tokens}
      doc_count: N
    """
    if tokenizer is None:
        tokenizer = simple_tokenize
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    inverted = defaultdict(dict)
    doc_lengths = {}
    for i, doc in enumerate(docs):
        tokens = tokenizer(doc, stopwords=stopwords)
        doc_lengths[i] = len(tokens)
        freqs = Counter(tokens)
        for term, f in freqs.items():
            inverted[term][i] = f
    return dict(inverted), doc_lengths, len(docs)

def docs_from_csv(path, column="Comment", encoding="utf-8"):
    import pandas as pd
    df = pd.read_csv(path, encoding=encoding, dtype=str).fillna('')
    if column not in df.columns:
        raise ValueError(f"CSV must contain column '{column}'")
    return df[column].tolist()
