"""Casino — Data science agent API.

FastAPI server with SSE streaming, artifact REST endpoints, and static file serving.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import get_config
from agent import stream_agent
from artifacts import (
    create_workspace,
    list_workspaces,
    delete_workspace,
    list_datasets,
    list_visualizations,
    list_scripts,
    list_reports,
    list_models,
    get_dataset_preview,
    ensure_workspace,
)

app = FastAPI(title="Casino", version="0.1.0")
cfg = get_config()

# CORS — allow frontend at localhost:3000 and any origin for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure workspace root exists
os.makedirs(cfg.workspace_root, exist_ok=True)


# ─── Health ───────────────────────────────────────────────────────────────────


@app.get("/")
async def health():
    return {
        "name": "casino",
        "version": "0.1.0",
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
        "sandbox": cfg.sandbox,
    }


# ─── Agent streaming ─────────────────────────────────────────────────────────


@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    workspace_id = body.get("workspaceId", "")
    history = body.get("history", [])

    if not prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    if not workspace_id:
        return JSONResponse({"error": "workspaceId is required"}, status_code=400)

    # Ensure workspace exists
    ensure_workspace(cfg.workspace_root, workspace_id)

    if not cfg.llm.api_key and cfg.llm.provider not in ("ollama",):
        return JSONResponse(
            {"error": f"LLM_API_KEY not set for provider '{cfg.llm.provider}'"},
            status_code=500,
        )

    async def event_stream():
        async for chunk in stream_agent(
            prompt=prompt,
            workspace_id=workspace_id,
            workspace_root=cfg.workspace_root,
            config=cfg.llm,
            message_history=history,
        ):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Workspaces ───────────────────────────────────────────────────────────────


@app.get("/workspaces")
async def get_workspaces():
    return {"workspaces": list_workspaces(cfg.workspace_root)}


@app.post("/workspaces")
async def post_workspace(request: Request):
    body = await request.json()
    name = body.get("name", f"workspace-{int(__import__('time').time())}")
    ws = create_workspace(cfg.workspace_root, name)
    return {"workspace": ws}


@app.delete("/workspaces/{workspace_id}")
async def del_workspace(workspace_id: str):
    ok = delete_workspace(cfg.workspace_root, workspace_id)
    return {"deleted": ok}


# ─── Datasets ─────────────────────────────────────────────────────────────────


@app.get("/workspaces/{workspace_id}/datasets")
async def get_datasets(workspace_id: str):
    return {"datasets": list_datasets(cfg.workspace_root, workspace_id)}


@app.get("/workspaces/{workspace_id}/datasets/{name}/preview")
async def get_preview(workspace_id: str, name: str, limit: int = 10):
    preview = get_dataset_preview(cfg.workspace_root, workspace_id, name, limit)
    if preview is None:
        return JSONResponse({"error": "Dataset not found"}, status_code=404)
    return preview


@app.post("/workspaces/{workspace_id}/datasets/upload")
async def upload_dataset(workspace_id: str, file: UploadFile = File(...)):
    """Upload a CSV/TSV/JSON file to the workspace datasets directory."""
    allowed = (".csv", ".tsv", ".json", ".parquet", ".xlsx")
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed:
        return JSONResponse(
            {"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"},
            status_code=400,
        )

    ds_dir = Path(cfg.workspace_root) / workspace_id / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    dest = ds_dir / file.filename

    contents = await file.read()
    dest.write_bytes(contents)

    # Generate metadata for CSV files
    meta = {"name": file.filename, "size": len(contents)}
    if ext == ".csv":
        import csv, io

        reader = csv.reader(io.StringIO(contents.decode("utf-8", errors="replace")))
        headers = next(reader, [])
        row_count = sum(1 for _ in reader)
        from artifacts import save_dataset_meta

        save_dataset_meta(
            cfg.workspace_root, workspace_id, file.filename,
            rows=row_count, columns=len(headers), column_names=headers,
        )
        meta.update({"rows": row_count, "columns": len(headers)})

    return {"uploaded": meta}


# ─── Visualizations ──────────────────────────────────────────────────────────


@app.get("/workspaces/{workspace_id}/visualizations")
async def get_visualizations(workspace_id: str):
    return {"visualizations": list_visualizations(cfg.workspace_root, workspace_id)}


# ─── Scripts ──────────────────────────────────────────────────────────────────


@app.get("/workspaces/{workspace_id}/scripts")
async def get_scripts(workspace_id: str):
    return {"scripts": list_scripts(cfg.workspace_root, workspace_id)}


# ─── Reports ──────────────────────────────────────────────────────────────────


@app.get("/workspaces/{workspace_id}/reports")
async def get_reports(workspace_id: str):
    return {"reports": list_reports(cfg.workspace_root, workspace_id)}


# ─── Models ───────────────────────────────────────────────────────────────────


@app.get("/workspaces/{workspace_id}/models")
async def get_models(workspace_id: str):
    return {"models": list_models(cfg.workspace_root, workspace_id)}


# ─── Static file serving for workspace artifacts ─────────────────────────────


@app.get("/workspace/{workspace_id}/{category}/{filename}")
async def serve_artifact(workspace_id: str, category: str, filename: str):
    """Serve workspace files (visualizations, reports, models, etc.)."""
    if category not in ("datasets", "visualizations", "scripts", "reports", "models"):
        return JSONResponse({"error": "Invalid category"}, status_code=400)

    file_path = Path(cfg.workspace_root) / workspace_id / category / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Prevent path traversal
    try:
        file_path.resolve().relative_to(Path(cfg.workspace_root).resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=403)

    return FileResponse(file_path)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=True,
        reload_excludes=["workspace/*", "*.pyc"],
    )
