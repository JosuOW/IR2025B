"""
qrels_utils.py
Manejo de archivos qrels para evaluación IR.
Formato esperado (espaciado):
query_id doc_id relevance
"""

def load_qrels(path):
    qrels = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid, docid, rel = parts[0], int(parts[1]), int(parts[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    return qrels


def save_example_qrels(path, queries=5):
    """
    Genera qrels de ejemplo para pruebas.
    query_i → doc_i relevante (1), doc_{i+1} no relevante (0)
    """
    with open(path, 'w', encoding='utf8') as f:
        for i in range(queries):
            f.write(f"q{i} {i} 1\n")
            f.write(f"q{i} {i+1} 0\n")
