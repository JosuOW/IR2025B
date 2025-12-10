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

# -----------------------------------------------------------
# PREPROCESAMIENTO FUERTE
# -----------------------------------------------------------
def clean_text_basic(text):
    """
    Preprocesamiento fuerte:
    - Minúsculas
    - Eliminación de HTML
    - Eliminación de símbolos
    - Reducción de espacios
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                     # Eliminar HTML
    text = re.sub(r"[^a-z0-9\s]", " ", text)               # Dejar solo letras y números
    text = re.sub(r"\s+", " ", text).strip()               # Espacios múltiples
    return text

# -----------------------------------------------------------
# TOKENIZADOR
# -----------------------------------------------------------
def simple_tokenize(text, lowercase=True, remove_punct=True, stopwords=None):
    """
    Tokenizador que aplica preprocesamiento fuerte.
    """
    if text is None:
        return []
    
    # Aplicar preprocesamiento fuerte
    text = clean_text_basic(text)
    tokens = text.split()

    # Remover stopwords
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens

# -----------------------------------------------------------
# ÍNDICE INVERTIDO
# -----------------------------------------------------------
def build_inverted_index(docs, tokenizer=None, stopwords=None):
    """
    Construye índice invertido:
      term -> {doc_id: frecuencia}
    Devuelve:
      inverted_index, doc_lengths, total_docs
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

# -----------------------------------------------------------
# CARGA DE CORPUS CSV
# -----------------------------------------------------------
def docs_from_csv(path, column="Text", encoding="utf-8"):
    """
    Carga documentos desde CSV.
    El corpus Amazon Fine Food Reviews usa la columna 'Text'.
    """
    import pandas as pd
    df = pd.read_csv(path, encoding=encoding, dtype=str).fillna('')
    if column not in df.columns:
        raise ValueError(f"CSV must contain column '{column}'")
    return df[column].tolist()
