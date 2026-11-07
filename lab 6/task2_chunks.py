import re
from lab6_search import Retriever, show_hits

def word_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    step = size - overlap
    for i in range(0, len(spans), step):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j-1][1]))
        if j == len(spans):
            break
    return result

def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("size must be positive")
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    output, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                output.append(pending)
            output.extend(windows(segment, size, 0, a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                output.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        output.append(pending)
    return output

def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        if strategy == "sentence":
            spans = sentence_chunks(text, 40)
        else:
            spans = windows(text, 40, int(strategy))
        for i, (start, end) in enumerate(spans):
            result.append({
                "id": f"{doc_id}:{i}", "doc": doc_id,
                "start": start, "end": end, "text": text[start:end]
            })
    return result

# Three synthetic teaching documents. Each is expanded to a longer passage.
docs = {
"handbook": """Late submission is normally penalized under the assessment rules.
Students should upload work through the learning portal before the deadline.
An illness may qualify for an extension when medical evidence is provided
before the deadline. The student should contact the course coordinator.
The evidence must relate to the period of illness. A request is reviewed
by the department and an extension is not automatic. Students should keep
a copy of the submitted work and confirmation message. The portal may show
a pending status while a file is checked. A late request without evidence
may be refused. The assessment office publishes any updated instructions.
Students should read the full rule because a general late penalty has an
exception for documented illness. The exception does not remove the need
to contact the coordinator. If the portal is unavailable, the student
should keep a screenshot of the error and contact support. The final
decision is recorded by the department. Always check the current handbook
version before relying on a rule.""",
"it_support": """A forgotten account password can be reset by using the recovery
link on the sign-in page. The user should enter the registered email address
and follow the message sent to that account. Recovery codes must remain
private and should never be shared with another person. If the email does
not arrive, check the spam folder and confirm that the registered address
is correct. After several unsuccessful attempts, contact the IT help desk.
The support team may ask for basic account information to verify the
request. Do not send a password in an email or include it in a support
ticket. A reset link may expire, so request a new link when necessary.
The system may temporarily delay messages during maintenance. Users
should wait for the message and avoid repeatedly submitting the form.
When access is restored, choose a strong password and sign out of shared
computers. The help desk records the issue number for later follow-up.
These instructions apply to the synthetic student portal used in this lab.""",
"returns": """A damaged parcel should be reported to customer support within
48 hours of delivery. The customer should attach clear photographs of the
item and its packaging. The support team checks the report and explains
the next step. Keep the order number until the case is closed. A change
of mind is different from damage: unused items may be returned within
14 days under the synthetic guide. Do not use the change-of-mind period
for a damaged-item report. The report deadline and the photo requirement
belong together and should be read as one procedure. If the parcel arrived
late but is not damaged, contact support with the delivery reference.
A refund is not automatic and depends on the review. The customer receives
a case number after submitting the photographs. Keep the packaging until
support replies. These rules are sample teaching data and are not an
actual store policy. Always check the correct region and current guide."""
}

if __name__ == "__main__":
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        print("\nStrategy:", strategy, "Chunk count:", len(chunks))
        for chunk in chunks:
            assert chunk["text"] == docs[chunk["doc"]][chunk["start"]:chunk["end"]]
            print(chunk["id"], chunk["start"], chunk["end"],
                  len(chunk["text"].split()), chunk["text"][:90])

    # Required boundary tests
    assert windows("") == []
    assert len(windows("short document")) == 1
    try:
        windows("a b", 2, 2)
    except ValueError:
        print("Invalid overlap test passed.")
    else:
        raise AssertionError("Invalid overlap was accepted")

    long_sentence = " ".join(["word"] * 95) + "."
    assert all(len(long_sentence[a:b].split()) <= 40
               for a, b in sentence_chunks(long_sentence))
    print("Boundary tests passed.")

    # Three paraphrased-query suggestions, TF-IDF and embeddings.
    records = make_chunks(docs, "sentence")
    queries = [
        "I was sick and need more time for my work",
        "I cannot get into my account",
        "My package arrived broken"
    ]
    for backend in ["tfidf", "embedding"]:
        try:
            engine = Retriever(records, backend)
            print("\nBackend:", backend)
            for query in queries:
                print("\nQuery:", query)
                show_hits(engine.search(query, k=3))
        except Exception as error:
            print(backend, "could not run:", error)
