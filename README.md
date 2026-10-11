 # Energy Operations Document RAG Assistant

An API that searches energy operations Markdown documents and returns relevant passages with source citations and retrieval evidence.

## Features

- Loads Markdown files from `data/` and splits them into passages.
- Retrieves passages using TF-IDF and cosine similarity.
- Returns source filenames, similarity scores, and retrieval evidence.
- Handles questions when no relevant evidence is found.

## Tech Stack

- Python and FastAPI
- Scikit-learn (TF-IDF and cosine similarity)
- Pandas

## Project Structure

```text
energy-rag-assistant/
├── backend/
│   ├── app.py
│   └── rag.py
├── data/                     # Energy operations Markdown documents
├── requirements.txt
└── README.md
```

## Setup

1. Install Python 3.12 or a compatible version.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Start the API from the project root:

	```bash
	python -m uvicorn backend.app:app --reload
	```

4. Open the API documentation at <http://127.0.0.1:8000/docs>.

## API Usage

Endpoint: `POST /api/run`

Example request:

```json
{
  "text": "What safety precautions are mentioned in the hydro safety brief?"
}
```

The response includes a retrieved answer, source citations, retrieval evidence, and limitations.

## Current Limitation

This version retrieves relevant source passages rather than generating new answers with an LLM. Retrieval quality depends on the available documents and query wording.

## Evaluation Results

| Test question | Result | Evidence |
|---|---|---|
| What safety precautions are mentioned in the hydro safety brief? | Passed | Retrieved the hydro safety paragraph with its source citation |
| What should be checked during solar inverter maintenance? | Partially passed | Retrieved the correct solar inverter passage, but also returned unrelated wind turbine passages |

Both test requests returned HTTP 200.

## Limitations

- Uses TF-IDF and cosine similarity for retrieval.
- Returns retrieved source passages rather than generating answers with an LLM.
- Retrieval may include unrelated passages.
- No comprehensive evaluation dataset or accuracy benchmark has been established.
- Railway deployment must be verified separately.