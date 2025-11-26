"""
evaluation.py
Funciones para cargar qrels y calcular precision, recall, AP, MAP.
Qrels format: query_id \t doc_id \t relevance (1 or 0)
"""
from collections import defaultdict
import math

def load_qrels(path):
    qrels = defaultdict(dict)  # qid -> {docid: relevance}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                continue
            qid, docid, rel = parts[0], int(parts[1]), int(parts[2])
            qrels[qid][docid] = rel
    return qrels

def precision_recall_at_k(retrieved, relevant_set, k):
    retrieved_k = [d for d,_ in retrieved[:k]]
    tp = sum(1 for d in retrieved_k if d in relevant_set)
    precision = tp / k if k>0 else 0.0
    recall = tp / len(relevant_set) if len(relevant_set)>0 else 0.0
    return precision, recall

def average_precision(retrieved, relevant_set):
    """
    retrieved: list of tuples (docid, score) ordered
    relevant_set: set of relevant docids
    """
    hits = 0
    sum_prec = 0.0
    for i, (docid, _) in enumerate(retrieved, start=1):
        if docid in relevant_set:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / hits

def mean_average_precision(all_retrieved, qrels):
    """
    all_retrieved: dict qid -> retrieved_list
    qrels: dict qid -> {docid: rel}
    """
    aps = []
    for qid, retrieved in all_retrieved.items():
        rel_docs = {d for d,r in qrels.get(qid, {}).items() if r>0}
        ap = average_precision(retrieved, rel_docs)
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0
