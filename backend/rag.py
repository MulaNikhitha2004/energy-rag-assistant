
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


def load_documents():
    documents = []

    for path in DATA.glob("*.md"):
        content = path.read_text(encoding="utf-8")

        # Split the document into paragraphs.
        chunks = re.split(r"\n\s*\n", content)

        for chunk in chunks:
            chunk = chunk.strip()

            # Ignore empty paragraphs and heading-only paragraphs.
            if not chunk:
                continue

            if all(
                line.strip().startswith("#")
                for line in chunk.splitlines()
            ):
                continue

            documents.append({
                "source": path.name,
                "text": chunk
            })

    return documents


def retrieve(query: str, top_k: int = 3):
    documents = load_documents()

    if not documents:
        print("No Markdown documents were loaded.")
        return []

    print("DOCUMENT COUNT:", len(documents))
    for doc in documents:
        print("SOURCE:", doc["source"])
        print("TEXT:", repr(doc["text"]))

    # Include filenames so document names can help retrieval.
    texts = [
        doc["source"].replace("_", " ").replace("-", " ")
        + " " + doc["text"]
        for doc in documents
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(
        matrix[-1], matrix[:-1]
    ).flatten()

    ranked = scores.argsort()[::-1]
    results = []

    for index in ranked[:top_k]:
        if scores[index] <= 0:
            continue

        results.append({
            "source": documents[index]["source"],
            "text": documents[index]["text"],
            "score": round(float(scores[index]), 4)
        })

    return results
