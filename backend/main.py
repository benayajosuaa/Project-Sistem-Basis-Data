from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import search_recipes

app = FastAPI()


class AskPayload(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
def ask(payload: AskPayload):
    result = search_recipes(payload.question, payload.top_k)

    if not result:
        return {"answer": "Tidak ada jawaban ditemukan.", "results": []}

    answer = result[0]["text"]

    return {
        "answer": answer,
        "results": result
    }
