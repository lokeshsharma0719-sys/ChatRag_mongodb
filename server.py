"""
Web UI for the MongoDB chatbot.

Run:   python server.py          (then open http://127.0.0.1:8000)
"""
import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import agent

HERE = Path(__file__).parent
PREVIEW_ROWS = 50

app = FastAPI(title="MongoDB Chatbot")
graph = agent.build_graph()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    """Used by the page header to show connection status."""
    try:
        db = agent.get_db()
        db.client.admin.command("ping")
        cat = agent.get_catalog()
        if cat:
            return {"ok": True, "mode": "catalog", "database": db.name, "collections": len(cat["collections"]),
                    "built_at": cat.get("built_at"), "models": agent._model_names()}
        coll = agent.get_collection()
        return {"ok": True, "mode": "single", "database": db.name, "collection": coll.name,
                "documents": coll.estimated_document_count(), "models": agent._model_names()}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


@app.post("/api/ask")
def ask(req: AskRequest):
    # Plain `def` (not async): FastAPI runs it in a worker thread, so slow LLM/DB calls don't block the server.
    started = time.time()
    try:
        final = graph.invoke(agent.initial_state(req.question))
    except Exception as e:  # LLM overloaded, quota, network, ...
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {str(e)[:300]}")

    docs = final.get("result") or []
    return {
        "answer": final.get("answer", ""),
        "success": final.get("result") is not None,
        "pipeline": final.get("pipeline"),
        "collection": final.get("collection"),
        "keywords": final.get("keywords", []),
        "candidates": final.get("candidates", []),
        "route_reason": final.get("route_reason"),
        "reasoning": final.get("reasoning"),
        "attempts": final.get("attempts", 0),
        "trace": final.get("trace", []),
        "total_docs": len(docs),
        # default=str turns ObjectId/datetime into strings so the browser can display them
        "results": json.loads(json.dumps(docs[:PREVIEW_ROWS], default=str)),
        "seconds": round(time.time() - started, 2),
    }


if __name__ == "__main__":
    # 127.0.0.1 = only reachable from this computer. There is no login, so don't expose it to a network.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)