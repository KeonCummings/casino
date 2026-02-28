"""Filesystem artifact management for workspace data."""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Any

ARTIFACT_DIRS = ["datasets", "visualizations", "scripts", "reports", "models"]


def workspace_path(workspace_root: str, workspace_id: str) -> Path:
    return Path(workspace_root) / workspace_id


def ensure_workspace(workspace_root: str, workspace_id: str) -> Path:
    root = workspace_path(workspace_root, workspace_id)
    for d in ARTIFACT_DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


def list_datasets(workspace_root: str, workspace_id: str) -> list[dict[str, Any]]:
    ds_dir = workspace_path(workspace_root, workspace_id) / "datasets"
    if not ds_dir.exists():
        return []

    datasets = []
    for f in sorted(ds_dir.iterdir()):
        if f.suffix == ".json":
            # Metadata file
            continue
        meta_path = ds_dir / f"{f.stem}.meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

        datasets.append(
            {
                "id": f.stem,
                "name": f.name,
                "rows": meta.get("rows", 0),
                "columns": meta.get("columns", 0),
                "schema": meta.get("schema"),
                "columnNames": meta.get("columnNames"),
                "createdAt": meta.get(
                    "createdAt", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_ctime))
                ),
            }
        )
    return datasets


def list_visualizations(
    workspace_root: str, workspace_id: str
) -> list[dict[str, Any]]:
    viz_dir = workspace_path(workspace_root, workspace_id) / "visualizations"
    if not viz_dir.exists():
        return []

    vizs = []
    for f in sorted(viz_dir.iterdir()):
        if f.suffix in (".png", ".jpg", ".jpeg", ".svg"):
            vizs.append(
                {
                    "name": f.stem,
                    "format": f.suffix.lstrip("."),
                    "imageUrl": f"/workspace/{workspace_id}/visualizations/{f.name}",
                    "createdAt": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_ctime)
                    ),
                }
            )
    return vizs


def list_scripts(workspace_root: str, workspace_id: str) -> list[dict[str, Any]]:
    scripts_dir = workspace_path(workspace_root, workspace_id) / "scripts"
    if not scripts_dir.exists():
        return []

    scripts = []
    for f in sorted(scripts_dir.iterdir()):
        content = None
        try:
            content = f.read_text()
        except Exception:
            pass

        scripts.append(
            {
                "name": f.stem,
                "extension": f.suffix.lstrip("."),
                "size_bytes": f.stat().st_size,
                "content": content,
                "created_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_ctime)
                ),
            }
        )
    return scripts


def list_reports(workspace_root: str, workspace_id: str) -> list[dict[str, Any]]:
    reports_dir = workspace_path(workspace_root, workspace_id) / "reports"
    if not reports_dir.exists():
        return []

    reports = []
    for f in sorted(reports_dir.iterdir()):
        reports.append(
            {
                "id": f.stem,
                "name": f.name,
                "format": f.suffix.lstrip("."),
                "url": f"/workspace/{workspace_id}/reports/{f.name}",
                "size": f.stat().st_size,
                "createdAt": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_ctime)
                ),
            }
        )
    return reports


def list_models(workspace_root: str, workspace_id: str) -> list[dict[str, Any]]:
    models_dir = workspace_path(workspace_root, workspace_id) / "models"
    if not models_dir.exists():
        return []

    models = []
    for f in sorted(models_dir.iterdir()):
        models.append(
            {
                "id": f.stem,
                "name": f.name,
                "format": f.suffix.lstrip("."),
                "size_bytes": f.stat().st_size,
                "last_modified": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime)
                ),
                "download_url": f"/workspace/{workspace_id}/models/{f.name}",
            }
        )
    return models


def save_dataset_meta(
    workspace_root: str,
    workspace_id: str,
    filename: str,
    rows: int,
    columns: int,
    schema: dict | None = None,
    column_names: list[str] | None = None,
) -> None:
    ds_dir = workspace_path(workspace_root, workspace_id) / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    meta = {
        "rows": rows,
        "columns": columns,
        "schema": schema,
        "columnNames": column_names,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ds_dir / f"{stem}.meta.json").write_text(json.dumps(meta))


def get_dataset_preview(
    workspace_root: str, workspace_id: str, name: str, limit: int = 10
) -> dict[str, Any] | None:
    ds_dir = workspace_path(workspace_root, workspace_id) / "datasets"

    # Find file by name (with or without extension)
    target = None
    for f in ds_dir.iterdir():
        if f.name == name or f.stem == name:
            if f.suffix != ".json":
                target = f
                break

    if not target:
        return None

    try:
        import csv

        with open(target, "r") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
            rows = []
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(row)

        return {"columns": headers, "rows": rows, "totalRows": len(rows)}
    except Exception:
        return None


def create_workspace(workspace_root: str, name: str) -> dict[str, Any]:
    workspace_id = f"ws_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    ensure_workspace(workspace_root, workspace_id)
    meta = {
        "workspaceId": workspace_id,
        "name": name,
        "datasetCount": 0,
        "visualizationCount": 0,
        "modelCount": 0,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lastAccessed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = workspace_path(workspace_root, workspace_id) / "workspace.json"
    meta_path.write_text(json.dumps(meta))
    return meta


def list_workspaces(workspace_root: str) -> list[dict[str, Any]]:
    root = Path(workspace_root)
    if not root.exists():
        return []

    workspaces = []
    for d in sorted(root.iterdir()):
        meta_path = d / "workspace.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            workspaces.append(meta)
    return workspaces


def delete_workspace(workspace_root: str, workspace_id: str) -> bool:
    ws = workspace_path(workspace_root, workspace_id)
    if ws.exists():
        shutil.rmtree(ws)
        return True
    return False
