import re
from lab6_search import Retriever

samples = [
    "AI, search!",
    "I can't log in.",
    "help@example.com",
    "CS4944 version 2.1",
    "",
    "یہ ایک مثال ہے"
]

print("TASK 1: TOKENIZATION")
for text in samples:
    whitespace = text.split()
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print("\nOriginal:", repr(text))
    print("Whitespace:", whitespace, "Count:", len(whitespace))
    print("Word-like:", words, "Count:", len(words))

print("\nTwo differences:")
print("1. Whitespace splitting keeps punctuation attached to words.")
print("2. Word-like regex removes punctuation and can split contractions.")
print("Risk: regex tokens are not a complete multilingual tokenizer.")

# Optional model-tokenizer comparison for two English examples.
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for text in samples[:2]:
        print("Model tokens:", text, model.tokenizer.tokenize(text))
except Exception as error:
    print("Model tokenizer not available:", error)

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    {"id": "faq:library", "question": "Where can I find library timings?",
     "answer": "Check the library notice board for current timings.",
     "source": "library.txt"},
    {"id": "faq:email", "question": "How do I contact IT support?",
     "answer": "Send your issue to the university IT help desk.",
     "source": "it_help.txt"}
]

def build_bot():
    records = [dict(row, text=row["question"] + " " + row["answer"])
               for row in FAQ]
    return Retriever(records, "tfidf")

def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    for row in FAQ:
        if query.strip().casefold() == row["question"].casefold():
            return row["answer"] + " [source: " + row["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    return "Possible matches; choose or rephrase your question:\n" + "\n".join(
        row["question"] for row, score in hits)

if __name__ == "__main__":
    engine = build_bot()
    print("\nFAQ assistant. Type exit to stop.")
    for test in ["How do I reset my password?",
                 "password recovery", "", "volcano geology"]:
        print("\nYou:", test)
        print("Bot:", reply(engine, test))
    while True:
        query = input("\nYou: ")
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))
