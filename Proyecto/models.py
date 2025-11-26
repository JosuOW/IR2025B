"""
models.py
Implementación de:
 - Jaccard (vectores binarios)
 - TF-IDF + Coseno (scikit-learn)
 - BM25 (implementación propia)
"""
from collections import defaultdict
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_rank(query, docs, tokenizer):
    qtokens = set(tokenizer(query))
    results = []
    for i, doc in enumerate(docs):
        dtokens = set(tokenizer(doc))
        inter = len(qtokens & dtokens)
        union = len(qtokens | dtokens)
        score = inter/union if union>0 else 0.0
        results.append((i, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

class TFIDFModel:
    def __init__(self, docs, tokenizer=None, max_features=None):
        # sklearn handles tokenization/stopwords; we accept raw docs
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.doc_matrix = self.vectorizer.fit_transform(docs)
    def query(self, query, topk=20):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_matrix).flatten()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:topk]

class BM25:
    # Simple BM25 implementation
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        """
        docs_tokens: list of list of tokens
        """
        self.k1 = k1
        self.b = b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_len)/self.N if self.N>0 else 0.0
        self.term_doc_freqs = {}
        for i, d in enumerate(docs_tokens):
            for t in set(d):
                self.term_doc_freqs.setdefault(t, 0)
                self.term_doc_freqs[t] += 1
        # precompute term frequencies per doc
        self.tf = [defaultdict(int) for _ in range(self.N)]
        for i,d in enumerate(docs_tokens):
            for t in d:
                self.tf[i][t] += 1

    def idf(self, term):
        n_q = self.term_doc_freqs.get(term, 0)
        # add 0.5 smoothing as in some BM25 variants
        return math.log(1 + (self.N - n_q + 0.5)/(n_q + 0.5))

    def score(self, query_tokens, idx):
        score = 0.0
        dl = self.doc_len[idx]
        for q in query_tokens:
            if q not in self.tf[idx]:
                continue
            idf_q = self.idf(q)
            tf = self.tf[idx][q]
            denom = tf + self.k1*(1 - self.b + self.b * dl / self.avgdl)
            score += idf_q * (tf * (self.k1 + 1)) / denom
        return score

    def query(self, query_tokens, topk=20):
        scores = []
        for i in range(self.N):
            s = self.score(query_tokens, i)
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]
