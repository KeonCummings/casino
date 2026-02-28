"""Python code execution via subprocess with artifact capture."""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    artifacts: list[dict]  # files created during execution


def execute_python(
    code: str,
    workspace_dir: str,
    timeout: int = 30,
) -> ExecutionResult:
    """Execute Python code in a subprocess, capture output and created files."""

    workspace_dir = os.path.abspath(workspace_dir)
    datasets_dir = os.path.join(workspace_dir, "datasets")
    viz_dir = os.path.join(workspace_dir, "visualizations")
    scripts_dir = os.path.join(workspace_dir, "scripts")
    reports_dir = os.path.join(workspace_dir, "reports")
    models_dir = os.path.join(workspace_dir, "models")

    for d in [datasets_dir, viz_dir, scripts_dir, reports_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    # Snapshot existing files to detect new ones
    def snapshot_dir(d: str) -> set[str]:
        if not os.path.exists(d):
            return set()
        return set(os.listdir(d))

    before = {
        "datasets": snapshot_dir(datasets_dir),
        "visualizations": snapshot_dir(viz_dir),
        "scripts": snapshot_dir(scripts_dir),
        "reports": snapshot_dir(reports_dir),
        "models": snapshot_dir(models_dir),
    }

    # Write code to temp file, execute with cwd = workspace
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=workspace_dir, delete=False
    ) as f:
        # Inject workspace paths so code can save artifacts to the right places
        preamble = f"""
import os, sys
WORKSPACE_DIR = {repr(workspace_dir)}
DATASETS_DIR = {repr(datasets_dir)}
VISUALIZATIONS_DIR = {repr(viz_dir)}
SCRIPTS_DIR = {repr(scripts_dir)}
REPORTS_DIR = {repr(reports_dir)}
MODELS_DIR = {repr(models_dir)}

# Configure matplotlib for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Override plt.show to save instead
_original_show = plt.show
import time as _time
_plot_counter = [0]
def _save_show(*args, **kwargs):
    _plot_counter[0] += 1
    fig = plt.gcf()
    name = f"plot_{{int(_time.time()*1000)}}_{{_plot_counter[0]}}.png"
    path = os.path.join(VISUALIZATIONS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close(fig)
plt.show = _save_show

# Override plt.savefig to also save to viz dir
_original_savefig = plt.savefig
def _capture_savefig(fname, *args, **kwargs):
    _original_savefig(fname, *args, **kwargs)
    # Copy to visualizations dir if not already there
    from pathlib import Path
    p = Path(fname)
    if p.suffix in ('.png', '.jpg', '.jpeg', '.svg'):
        dest = Path(VISUALIZATIONS_DIR) / p.name
        if str(p.resolve()) != str(dest.resolve()):
            import shutil
            shutil.copy2(str(p), str(dest))
plt.savefig = _capture_savefig
"""
        f.write(preamble + "\n" + code)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_dir,
            env={
                **os.environ,
                "MPLBACKEND": "Agg",
            },
        )
        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = f"Execution timed out after {timeout}s"
        returncode = -1
    except Exception as e:
        stdout = ""
        stderr = str(e)
        returncode = -1
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    # Scan for new files after execution (also check cwd for stray files)
    _capture_stray_files(workspace_dir, viz_dir, datasets_dir)

    after = {
        "datasets": snapshot_dir(datasets_dir),
        "visualizations": snapshot_dir(viz_dir),
        "scripts": snapshot_dir(scripts_dir),
        "reports": snapshot_dir(reports_dir),
        "models": snapshot_dir(models_dir),
    }

    artifacts = []
    for category in ARTIFACT_CATEGORIES:
        new_files = after[category] - before[category]
        for fname in new_files:
            artifacts.append({"category": category, "name": fname})

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        artifacts=artifacts,
    )


ARTIFACT_CATEGORIES = ["datasets", "visualizations", "scripts", "reports", "models"]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg"}
DATA_EXTS = {".csv", ".tsv", ".parquet", ".json", ".xlsx"}
SCRIPT_EXTS = {".py", ".r", ".sql", ".sh"}
REPORT_EXTS = {".md", ".txt", ".html", ".pdf"}
MODEL_EXTS = {".pkl", ".joblib", ".h5", ".pt", ".onnx"}


def _capture_stray_files(
    workspace_dir: str, viz_dir: str, datasets_dir: str
) -> None:
    """Move files created in workspace root to appropriate artifact dirs."""
    for fname in os.listdir(workspace_dir):
        fpath = os.path.join(workspace_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()

        if ext in IMAGE_EXTS:
            dest = os.path.join(viz_dir, fname)
            if not os.path.exists(dest):
                shutil.move(fpath, dest)
        elif ext in DATA_EXTS:
            dest = os.path.join(datasets_dir, fname)
            if not os.path.exists(dest):
                shutil.move(fpath, dest)
