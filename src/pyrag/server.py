import json
import os
import threading

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pyrag.wiring import build_pipeline

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                pipeline, _ = build_pipeline()
                _pipeline = pipeline
    return _pipeline


class ChatRequest(BaseModel):
    question: str


def _serialize_event(event: dict) -> str:
    payload = dict(event)
    if "sources" in payload:
        payload["sources"] = [
            {
                "section_title": rc.chunk.section_title,
                "source_file": rc.chunk.source_file,
                "score": rc.score,
            }
            for rc in payload["sources"]
        ]
    return f"data: {json.dumps(payload)}\n\n"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/chat")
def chat(request: ChatRequest, pipeline=Depends(get_pipeline)):
    def event_stream():
        for event in pipeline.answer_stream(request.question):
            yield _serialize_event(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
