"""
cli.py
Interfaz de línea de comandos para el sistema IR.
Ejemplos:
  python cli.py --corpus path/to/corpus.csv --query "apple pay" --model bm25
"""
import argparse
from index import docs_from_csv, build_inverted_index, simple_tokenize
from models import jaccard_rank, TFIDFModel, BM25
import sys
import csv

def load_docs(path):
    return docs_from_csv(path, column="Comment")

def build_models(docs, tokenizer):
    # prepare tokens for BM25
    docs_tokens = [tokenizer(d) for d in docs]
    bm25 = BM25(docs_tokens)
    tfidf = TFIDFModel(docs)
    return bm25, tfidf

def print_ranked(docs, ranked, top=10):
    for i, (docid, score) in enumerate(ranked[:top], start=1):
        print(f"{i:2d}. doc={docid} score={score:.6f}")
        print(f"    {docs[docid][:200]}")
        print()

def main():
    p = argparse.ArgumentParser(description="CLI for simple IR system")
    p.add_argument("--corpus", required=True, help="path to CSV corpus (column 'Comment')")
    p.add_argument("--model", choices=['jaccard','tfidf','bm25'], default='bm25')
    p.add_argument("--query", required=True, help="free-text query")
    p.add_argument("--top", type=int, default=10, help="top K results")
    args = p.parse_args()

    try:
        docs = load_docs(args.corpus)
    except Exception as e:
        print("Error loading corpus:", e, file=sys.stderr)
        sys.exit(1)

    tokenizer = lambda text: simple_tokenize(text, stopwords=None)
    bm25, tfidf = build_models(docs, tokenizer)

    if args.model == 'jaccard':
        ranked = jaccard_rank(args.query, docs, tokenizer)
    elif args.model == 'tfidf':
        ranked = tfidf.query(args.query, topk=args.top)
    else:
        qtokens = tokenizer(args.query)
        ranked = bm25.query(qtokens, topk=args.top)

    print_ranked(docs, ranked, top=args.top)

if __name__ == "__main__":
    main()
