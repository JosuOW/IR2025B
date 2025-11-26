# IR Project — Sistema de Recuperación de Información

Este repositorio contiene una implementación simple de un sistema IR solicitado en `proy01.pdf`.
Implementa:
- Índice invertido (index.py)
- Modelos de recuperación:
  - Jaccard (binario)
  - TF-IDF + Coseno (scikit-learn)
  - BM25 (implementación propia)
- Interfaz de línea de comandos (cli.py)
- Evaluación: precision, recall, MAP (evaluation.py)

## Requisitos
- Python 3.8+
- bibliotecas: pandas, scikit-learn, numpy

Instalación rápida:
```bash
pip install pandas scikit-learn numpy
```

## Estructura
- `index.py` — construcción del índice, tokenización.
- `models.py` — Jaccard, TF-IDF wrapper, BM25.
- `evaluation.py` — métricas y carga de qrels.
- `cli.py` — interfaz CLI para ejecutar consultas.
- `main.py` — wrapper para ejecutar `cli.py`.

## Uso
Ejemplo:
```bash
python cli.py --corpus path/to/sentiment_corpus.csv --model bm25 --query "apple pay" --top 10
```

## Notes
- El corpus debe ser un CSV con columna `Comment`.
- Qrels deben tener formato: `query_id doc_id relevance` por línea (relevance 1/0).
- No se incluyen scripts de limpieza de corpus: el usuario debe preprocesar el CSV si lo necesita.
