"""
interface_simple.py
Interfaz super simple basada en menú para consultas IR.
"""

from index import docs_from_csv, simple_tokenize
from models import jaccard_rank, TFIDFModel, BM25


def run_interface(corpus_path):
    print("Cargando corpus...")
    docs = docs_from_csv(corpus_path, column="Text")
    tokenizer = lambda text: simple_tokenize(text, stopwords=None)

    print("Construyendo modelos...")
    docs_tokens = [tokenizer(doc) for doc in docs]
    bm25 = BM25(docs_tokens)
    tfidf = TFIDFModel(docs)

    while True:
        print("\n===== MENÚ IR =====")
        print("1) Buscar con Jaccard")
        print("2) Buscar con TF-IDF")
        print("3) Buscar con BM25")
        print("4) Salir")

        opcion = input("Seleccione opción: ").strip()
        if opcion == "4":
            break

        consulta = input("\nIngrese consulta: ")

        if opcion == "1":
            ranking = jaccard_rank(consulta, docs, tokenizer)
        elif opcion == "2":
            ranking = tfidf.query(consulta)
        elif opcion == "3":
            ranking = bm25.query(tokenizer(consulta))
        else:
            print("Opción inválida.")
            continue

        print("\n--- Resultados ---")
        for i, (docid, score) in enumerate(ranking[:10], 1):
            print(f"{i}. Doc {docid} | Score {score:.4f}")
            print("→", docs[docid][:250])
            print()
