# IR Project — Sistema de Recuperación de Información

Este repositorio contiene una implementación simple de un sistema RI.
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

## Estructura del proyecto
- index.py — Preprocesamiento + Indexado
- models.py — Modelos Jaccard, TF-IDF, BM25
- evaluation.py — Métricas IR
- qrels_utils.py — Manejo de qrels
- cli.py — Interfaz por línea de comandos
- interface_simple.py — Menú interactivo

## Uso
Ejemplo:
```bash
python cli.py --corpus amazon.csv --model bm25 --query "great flavor"
```

```bash
python interface_simple.py
```
Qrels
```bash
python -c "import qrels_utils as q; q.save_example_qrels('qrels.txt')"

```


