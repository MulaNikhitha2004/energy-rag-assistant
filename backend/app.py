from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from backend.rag import retrieve

app = FastAPI(title='Energy Operations Document RAG Assistant')
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'


class Query(BaseModel):
    text: str = ''


@app.get('/api/demo')
def demo():
    return {
        'case_study': 'Energy Operations Document RAG Assistant',
        'status': 'starter app running',
        'expected_outcome': 'A working RAG-based assistant that answers operational questions from provided documents with citations, retrieval evidence, and hallucination checks.'
    }


@app.post('/api/run')
def run_case(query: Query):
    if not query.text.strip():
        return {
            "input": query.text,
            "result": "Please enter a question.",
            "citations": [],
            "retrieval_evidence": []
        }

    evidence = retrieve(query.text)

    if not evidence:
        return {
            "input": query.text,
            "result": "I could not find relevant information in the provided documents.",
            "citations": [],
            "retrieval_evidence": [],
            "limitations": "No matching evidence was found."
        }

    answer = "\n\n".join(item["text"] for item in evidence)

    return {
        "input": query.text,
        "result": answer,
        "citations": list(dict.fromkeys(
            item["source"] for item in evidence
        )),
        "retrieval_evidence": evidence,
        "limitations": "This version retrieves source passages; it does not generate a new answer with an LLM."
    }