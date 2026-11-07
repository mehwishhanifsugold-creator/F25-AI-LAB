import csv, json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from task2_chunks import make_chunks

# Five-document corpus for evaluation.
docs = {
"late": "Late submission is normally penalized. Illness extensions require medical evidence before the deadline. Contact the course coordinator and keep the request confirmation. The general penalty and illness exception must be read together.",
"password": "Reset a forgotten password using the recovery link on the sign-in page. Never share recovery codes. If the message does not arrive, check spam and contact the IT help desk.",
"assignment": "Upload assignments through the learning portal. Submit the final file as a PDF. Keep the upload confirmation and check that the correct course page was used.",
"contractor": "Contractors can work remotely only if the signed contract permits it. Employee rules do not automatically apply to contractors. Check the contract version before making a decision.",
"printer": "For blank printer pages, check ink levels and run the cleaning utility. Print a test page afterwards. Follow the device manual and confirm that the correct printer is selected."
}

questions = [
("What proof is needed for illness extensions?", "late", "dev"),
("How can I reset my forgotten password?", "password", "dev"),
("Where do I upload my coursework?", "assignment", "dev"),
("Can a contractor work from home?", "contractor", "dev"),
("What should I do for blank printer pages?", "printer", "test"),
("How can I regain access to my account?", "password", "test"),
("How should I hand in my course work?", "assignment", "test"),
("What evidence is needed when I was sick?", "late", "test"),
("What are tomorrow's flight prices?", None, "test"),
("Who won the football match?", None, "test")
]
evidence = {
"late": "Illness extensions require medical evidence before the deadline.",
"password": "Reset a forgotten password using the recovery link on the sign-in page.",
"assignment": "Upload assignments through the learning portal.",
"contractor": "Contractors can work remotely only if the signed contract permits it.",
"printer": "For blank printer pages, check ink levels and run the cleaning utility."
}
items = [{"query": q, "doc": doc,
          "evidence": evidence[doc] if doc else None, "split": split}
         for q, doc, split in questions]

def evaluate(score_rows, chunks, selected_items):
    hits, reciprocal = [], []
    for row, q in zip(score_rows, selected_items):
        if q["doc"] is None:
            continue
        top = np.argsort(-row, kind="stable")[:3]
        good = [rank for rank, i in enumerate(top, 1)
                if chunks[i]["doc"] == q["doc"]
                and q["evidence"] in chunks[i]["text"]]
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    if not hits:
        return None, None
    return float(np.mean(hits)), float(np.mean(reciprocal))

def main():
    with open("corpus.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open("queries.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    load_start = time.perf_counter()
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("Model load seconds:", round(time.perf_counter()-load_start, 3))

    query_texts = [q["query"] for q in items]
    logs = []
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        texts = [c["text"] for c in chunks]
        lengths = [len(model.tokenizer(t, truncation=False)["input_ids"])
                   for t in texts + query_texts]
        if max(lengths) > model.max_seq_length:
            raise ValueError("Model limit exceeded; rechunk or shorten text")

        for method in ["tfidf", "embedding"]:
            start = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform(query_texts)
                encode_time = time.perf_counter() - start
                search_start = time.perf_counter()
                scores = (Q @ C.T).toarray()
            else:
                C = model.encode(texts, normalize_embeddings=True)
                Q = model.encode(query_texts, normalize_embeddings=True)
                encode_time = time.perf_counter() - start
                search_start = time.perf_counter()
                scores = Q @ C.T
            search_time = time.perf_counter() - search_start

            for split in ["dev", "test"]:
                indexes = [i for i, q in enumerate(items)
                           if q["split"] == split]
                result = evaluate(scores[indexes], chunks,
                                  [items[i] for i in indexes])
                print(strategy, method, split,
                      "Hit@3/MRR@3:", result,
                      "chunks:", len(chunks),
                      "encode:", round(encode_time, 4),
                      "search:", round(search_time, 4))

            for row, q in zip(scores, items):
                for rank, i in enumerate(
                        np.argsort(-row, kind="stable")[:3], 1):
                    c = chunks[i]
                    logs.append({
                        "strategy": strategy, "method": method,
                        "query": q["query"], "split": q["split"],
                        "rank": rank, "score": float(row[i]),
                        "chunk_id": c["id"], "doc": c["doc"],
                        "start": c["start"], "end": c["end"],
                        "text": c["text"],
                        "answerable": q["doc"] is not None
                    })
    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    print("Saved retrieval_log.csv")
    print("Unanswerable queries are logged separately; no answer is asserted.")

if __name__ == "__main__":
    main()
