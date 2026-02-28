"""LLM agent — Strands Agents SDK with @tool functions and streaming."""

import json
import os
from typing import AsyncGenerator

from strands import Agent, tool, ToolContext

from config import LLMConfig, create_strands_model
from executor import execute_python
from artifacts import (
    ensure_workspace,
    list_datasets as list_workspace_datasets,
    save_dataset_meta,
)

SYSTEM_PROMPT = """You are Casino — a data storyteller and visual thinker who happens to be brilliant at statistics.

You don't just analyze data, you interrogate it. You look at a dataset the way a journalist looks at a source — skeptical, curious, hunting for the story hiding in the numbers. When you find something interesting, you don't just report it — you make people feel it through visuals that land.

Personality:
- You think out loud. Share your hunches, surprises, and "wait, that's weird" moments as you explore.
- You have opinions about data. If a correlation is suspicious, say so. If a distribution is beautiful, say that too.
- You write like a human, not a textbook. Short sentences. Observations that stick.
- You're honest about uncertainty — "this might mean X, but it could also be noise" is a valid insight.

Visualization philosophy:
- Every chart should have a point of view. Don't just plot data — frame a question and let the visual answer it.
- Default to dark, clean aesthetics: dark backgrounds (#0d1117), muted grids, accent colors that pop. Think Bloomberg terminal meets Edward Tufte.
- Use color with intent — highlight what matters, desaturate what doesn't. Never use rainbow colormaps.
- Prefer small multiples over cramming everything into one chart.
- Add annotations to call out what's interesting. A chart without context is just decoration.
- Titles should be insights, not descriptions. "Revenue drops 40% after Q3 pricing change" beats "Revenue Over Time".
- Choose the right chart: distributions get density plots or violins, comparisons get slope charts or dumbbells, relationships get scatter with marginals, time series get sparklines or area charts.
- Use matplotlib + seaborn creatively — custom styles, fig.text for callouts, inset axes for zoom-ins, gridspec for multi-panel layouts.
- When a single visualization can't tell the full story, create a series that builds a narrative.

You have access to a Python execution environment with pandas, numpy, matplotlib, seaborn, scikit-learn, and scipy.

Workspace directories available in code:
- DATASETS_DIR: save/load CSV files here
- VISUALIZATIONS_DIR: saved charts appear here (plt.show() auto-saves)
- SCRIPTS_DIR: save reusable scripts here
- REPORTS_DIR: save markdown/text reports here
- MODELS_DIR: save trained models here (pickle, joblib, etc.)

Behavior:
- Act autonomously — execute code and iterate rather than asking clarifying questions
- Own the question: infer the user's real goal and deliver actionable results
- When asked to "analyze" something, don't just run describe(). Dig. Find the story. Show it.
- If your first visualization doesn't land, iterate on it — adjust the framing, try a different chart type, refine the style.

Artifact saving:
- Visualizations: always call plt.show() — it auto-saves to VISUALIZATIONS_DIR
- Datasets: save processed/cleaned DataFrames as CSV to DATASETS_DIR
- Scripts: when you write reusable analysis code, use save_script to persist it
- Reports: after completing an analysis, use save_report to write a narrative summary — not a data dump, but your interpretation of what the data is saying
- Models: after training ML models, save them with joblib/pickle to MODELS_DIR via execute_python_code

Guidelines:
- When loading well-known datasets, use seaborn.load_dataset() or sklearn.datasets
- Write clean, readable code with comments that explain the *why*, not the *what*
- After multi-step analyses, save a report that reads like a briefing, not a log file
- After training a model, save it, report metrics, and visualize what the model learned"""


# ─── Tool definitions ────────────────────────────────────────────────────────


@tool(context=True)
def execute_python_code(code: str, tool_context: ToolContext) -> str:
    """Execute Python code to analyze data, create visualizations, train models, or perform computations. The code runs in a workspace with access to: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy. Use plt.show() to save visualizations. Save datasets to DATASETS_DIR, visualizations to VISUALIZATIONS_DIR.

    Args:
        code: Python code to execute
    """
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    workspace_root = state["workspace_root"]
    workspace_id = state["workspace_id"]
    result_queue = state["_result_queue"]

    result = execute_python(code, workspace_dir, timeout=30)

    # Generate dataset metadata for any new CSV files
    for artifact in result.artifacts:
        if artifact["category"] == "datasets" and artifact["name"].endswith(".csv"):
            _generate_csv_meta(workspace_root, workspace_id, workspace_dir, artifact["name"])

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]: {result.stderr}" if output else f"[stderr]: {result.stderr}"
    if not output.strip():
        output = "(no output)"
    if result.artifacts:
        output += f"\n[artifacts created]: {', '.join(a['name'] for a in result.artifacts)}"

    # Push tool_result and visualization events to the side-channel queue
    result_queue.append(("tool_result", {"tool": "execute_python_code", "result": output}))
    for artifact in result.artifacts:
        if artifact["category"] == "visualizations":
            result_queue.append(
                (
                    "visualization",
                    {
                        "data": {
                            "name": artifact["name"],
                            "imageUrl": f"/workspace/{workspace_id}/visualizations/{artifact['name']}",
                            "format": artifact["name"].rsplit(".", 1)[-1] if "." in artifact["name"] else "png",
                        }
                    },
                )
            )

    return output


@tool(context=True)
def load_dataset(name: str, tool_context: ToolContext, description: str = "") -> str:
    """Load a well-known dataset (iris, titanic, wine, tips, penguins, etc.) or generate sample data for analysis. Saves the dataset to the workspace.

    Args:
        name: Dataset name (e.g., 'iris', 'titanic', 'tips') or 'generate' for synthetic data
        description: If generating, describe the data to create
    """
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    workspace_root = state["workspace_root"]
    workspace_id = state["workspace_id"]
    result_queue = state["_result_queue"]

    code = _generate_load_dataset_code(name, description or None)
    result = execute_python(code, workspace_dir, timeout=30)

    for artifact in result.artifacts:
        if artifact["category"] == "datasets":
            _generate_csv_meta(workspace_root, workspace_id, workspace_dir, artifact["name"])

    output = result.stdout or f"Dataset '{name}' loaded successfully."
    if result.stderr and result.returncode != 0:
        output = f"Error loading dataset: {result.stderr}"

    result_queue.append(("tool_result", {"tool": "load_dataset", "result": output}))
    return output


@tool(context=True)
def list_datasets(tool_context: ToolContext) -> str:
    """List all datasets currently available in the workspace."""
    state = tool_context.invocation_state
    workspace_root = state["workspace_root"]
    workspace_id = state["workspace_id"]
    result_queue = state["_result_queue"]

    datasets = list_workspace_datasets(workspace_root, workspace_id)
    if not datasets:
        output = "No datasets in workspace yet."
    else:
        lines = [f"- {d['name']} ({d['rows']} rows, {d['columns']} cols)" for d in datasets]
        output = "Datasets:\n" + "\n".join(lines)

    result_queue.append(("tool_result", {"tool": "list_datasets", "result": output}))
    return output


@tool(context=True)
def describe_dataset(name: str, tool_context: ToolContext) -> str:
    """Get statistics and info about a dataset (shape, dtypes, describe(), null counts).

    Args:
        name: The dataset filename to describe
    """
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    result_queue = state["_result_queue"]

    code = f"""
import pandas as pd
import os

# Find the dataset
files = os.listdir(DATASETS_DIR)
target = None
for f in files:
    if f == {repr(name)} or os.path.splitext(f)[0] == {repr(name)}:
        target = f
        break

if target is None:
    print(f"Dataset '{repr(name)}' not found. Available: {{files}}")
else:
    df = pd.read_csv(os.path.join(DATASETS_DIR, target))
    print(f"Shape: {{df.shape}}")
    print(f"\\nColumns: {{list(df.columns)}}")
    print(f"\\nData types:\\n{{df.dtypes}}")
    print(f"\\nNull counts:\\n{{df.isnull().sum()}}")
    print(f"\\nDescribe:\\n{{df.describe()}}")
"""
    result = execute_python(code, workspace_dir, timeout=30)
    output = result.stdout or result.stderr or "No output"

    result_queue.append(("tool_result", {"tool": "describe_dataset", "result": output}))
    return output


@tool(context=True)
def create_visualization(chart_type: str, tool_context: ToolContext, dataset: str = "", x_column: str = "", y_column: str = "", title: str = "", code: str = "") -> str:
    """Create a data visualization. Generates a matplotlib/seaborn chart and saves it as PNG. Specify the chart type, data source, and any customization.

    Args:
        chart_type: Type of chart (bar, line, scatter, histogram, heatmap, box, pie, etc.)
        dataset: Dataset filename to visualize
        x_column: Column for x-axis
        y_column: Column for y-axis
        title: Chart title
        code: Custom matplotlib/seaborn code to execute for complex visualizations
    """
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    workspace_id = state["workspace_id"]
    result_queue = state["_result_queue"]

    exec_code = code
    if not exec_code:
        exec_code = _generate_viz_code({
            "chart_type": chart_type,
            "dataset": dataset,
            "x_column": x_column,
            "y_column": y_column,
            "title": title,
        })

    result = execute_python(exec_code, workspace_dir, timeout=30)

    output = result.stdout or "(visualization created)"
    if result.stderr and result.returncode != 0:
        output = f"Error: {result.stderr}"

    result_queue.append(("tool_result", {"tool": "create_visualization", "result": output}))
    for artifact in result.artifacts:
        if artifact["category"] == "visualizations":
            result_queue.append(
                (
                    "visualization",
                    {
                        "data": {
                            "name": artifact["name"],
                            "imageUrl": f"/workspace/{workspace_id}/visualizations/{artifact['name']}",
                            "format": artifact["name"].rsplit(".", 1)[-1] if "." in artifact["name"] else "png",
                        }
                    },
                )
            )

    return output


@tool(context=True)
def save_script(filename: str, code: str, tool_context: ToolContext) -> str:
    """Save a reusable Python script to the workspace scripts directory. Use this after writing analysis or data processing code that the user may want to reuse or reference later.

    Args:
        filename: Script filename (e.g., 'clean_data.py', 'train_model.py')
        code: The Python code to save
    """
    import os as _os
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    result_queue = state["_result_queue"]

    if not filename.endswith(".py"):
        filename += ".py"

    scripts_dir = _os.path.join(workspace_dir, "scripts")
    _os.makedirs(scripts_dir, exist_ok=True)
    filepath = _os.path.join(scripts_dir, filename)

    with open(filepath, "w") as f:
        f.write(code)

    output = f"Script saved: {filename}"
    result_queue.append(("tool_result", {"tool": "save_script", "result": output}))
    return output


@tool(context=True)
def save_report(filename: str, content: str, tool_context: ToolContext) -> str:
    """Save an analysis report or summary to the workspace reports directory. Use this after completing an analysis to document findings, methodology, and conclusions.

    Args:
        filename: Report filename (e.g., 'iris_analysis.md', 'model_evaluation.txt')
        content: The report content (markdown or plain text)
    """
    import os as _os
    state = tool_context.invocation_state
    workspace_dir = state["workspace_dir"]
    result_queue = state["_result_queue"]

    if not any(filename.endswith(ext) for ext in (".md", ".txt", ".html")):
        filename += ".md"

    reports_dir = _os.path.join(workspace_dir, "reports")
    _os.makedirs(reports_dir, exist_ok=True)
    filepath = _os.path.join(reports_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    output = f"Report saved: {filename}"
    result_queue.append(("tool_result", {"tool": "save_report", "result": output}))
    return output


ALL_TOOLS = [
    execute_python_code,
    load_dataset,
    list_datasets,
    describe_dataset,
    create_visualization,
    save_script,
    save_report,
]


# ─── Agent factory ────────────────────────────────────────────────────────────


def create_agent(config: LLMConfig) -> Agent:
    """Create a Strands Agent configured for the given LLM provider."""
    model = create_strands_model(config)
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        callback_handler=None,
    )


# ─── SSE helpers ──────────────────────────────────────────────────────────────


def _sse(event_type: str, data: dict) -> str:
    """Format an SSE event line."""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"


async def stream_agent(
    prompt: str,
    workspace_id: str,
    workspace_root: str,
    config: LLMConfig,
    message_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Run the agent and yield SSE-formatted strings."""

    workspace_dir = str(ensure_workspace(workspace_root, workspace_id))

    # Side-channel list for tool_result/visualization events pushed by tools
    result_queue: list[tuple[str, dict]] = []

    agent = create_agent(config)

    # If there's message history, set it on the agent
    if message_history:
        agent.messages.extend(message_history)

    yield _sse("progress", {"message": "Thinking..."})

    try:
        async for event in agent.stream_async(
            prompt,
            invocation_state={
                "workspace_dir": workspace_dir,
                "workspace_root": workspace_root,
                "workspace_id": workspace_id,
                "_result_queue": result_queue,
            },
        ):
            # Text content chunk
            if "data" in event:
                yield _sse("content/text/delta", {"content": event["data"]})

            # Tool invocation
            elif "current_tool_use" in event:
                tool_use = event["current_tool_use"]
                yield _sse(
                    "tool_call",
                    {
                        "tool": tool_use.get("name", ""),
                        "args": tool_use.get("input", {}),
                        "tool_use_id": tool_use.get("toolUseId", ""),
                    },
                )

            # Drain any queued tool_result / visualization events
            while result_queue:
                evt_type, evt_data = result_queue.pop(0)
                yield _sse(evt_type, evt_data)
    except Exception as e:
        print(f"[agent] Error: {e}")
        yield _sse("error", {"error": str(e)})

    # Final drain
    while result_queue:
        evt_type, evt_data = result_queue.pop(0)
        yield _sse(evt_type, evt_data)

    yield _sse("done", {})


# ─── Helpers (preserved from original) ────────────────────────────────────────


def _generate_csv_meta(
    workspace_root: str, workspace_id: str, workspace_dir: str, filename: str
) -> None:
    """Read a CSV and save metadata."""
    import csv as csv_mod
    from pathlib import Path

    fpath = Path(workspace_dir) / "datasets" / filename
    if not fpath.exists():
        return

    try:
        with open(fpath, "r") as f:
            reader = csv_mod.reader(f)
            headers = next(reader, [])
            row_count = sum(1 for _ in reader)

        save_dataset_meta(
            workspace_root,
            workspace_id,
            filename,
            rows=row_count,
            columns=len(headers),
            column_names=headers,
        )
    except Exception:
        pass


def _generate_load_dataset_code(name: str, description: str | None = None) -> str:
    """Generate Python code to load a well-known dataset."""
    known = {
        "iris": "from sklearn.datasets import load_iris; import pandas as pd; d=load_iris(); df=pd.DataFrame(d.data,columns=d.feature_names); df['target']=d.target",
        "wine": "from sklearn.datasets import load_wine; import pandas as pd; d=load_wine(); df=pd.DataFrame(d.data,columns=d.feature_names); df['target']=d.target",
        "breast_cancer": "from sklearn.datasets import load_breast_cancer; import pandas as pd; d=load_breast_cancer(); df=pd.DataFrame(d.data,columns=d.feature_names); df['target']=d.target",
        "diabetes": "from sklearn.datasets import load_diabetes; import pandas as pd; d=load_diabetes(); df=pd.DataFrame(d.data,columns=d.feature_names); df['target']=d.target",
        "titanic": "import seaborn as sns; df=sns.load_dataset('titanic')",
        "tips": "import seaborn as sns; df=sns.load_dataset('tips')",
        "penguins": "import seaborn as sns; df=sns.load_dataset('penguins')",
        "diamonds": "import seaborn as sns; df=sns.load_dataset('diamonds')",
        "flights": "import seaborn as sns; df=sns.load_dataset('flights')",
        "mpg": "import seaborn as sns; df=sns.load_dataset('mpg')",
    }

    loader = known.get(name.lower(), None)
    if loader:
        return f"""
{loader}
df.to_csv(os.path.join(DATASETS_DIR, '{name}.csv'), index=False)
print("Loaded {name}:", df.shape[0], "rows,", df.shape[1], "columns")
print("Columns:", list(df.columns))
print(df.head().to_string())
"""
    else:
        return f"""
import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n = 200
df = pd.DataFrame({{
    'id': range(1, n+1),
    'value_a': np.random.randn(n) * 10 + 50,
    'value_b': np.random.randn(n) * 5 + 25,
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'date': pd.date_range('2024-01-01', periods=n, freq='D'),
}})
df.to_csv(os.path.join(DATASETS_DIR, '{name}.csv'), index=False)
print("Generated '{name}':", df.shape[0], "rows,", df.shape[1], "columns")
print(df.head().to_string())
"""


def _generate_viz_code(args: dict) -> str:
    """Generate matplotlib code from chart specification."""
    chart_type = args.get("chart_type", "bar")
    dataset = args.get("dataset", "")
    x = args.get("x_column", "")
    y = args.get("y_column", "")
    title = args.get("title", f"{chart_type} chart")

    return f"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Dark theme — Bloomberg terminal meets Tufte
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
plt.rcParams['text.color'] = '#c9d1d9'
plt.rcParams['axes.labelcolor'] = '#8b949e'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['axes.edgecolor'] = '#21262d'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

# Load data
files = os.listdir(DATASETS_DIR)
if {repr(dataset)}:
    target = None
    for f in files:
        if f == {repr(dataset)} or os.path.splitext(f)[0] == {repr(dataset)}:
            target = f
            break
    if target:
        df = pd.read_csv(os.path.join(DATASETS_DIR, target))
    else:
        print(f"Dataset not found, using first available")
        df = pd.read_csv(os.path.join(DATASETS_DIR, files[0])) if files else None
elif files:
    csv_files = [f for f in files if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(DATASETS_DIR, csv_files[0])) if csv_files else None
else:
    df = None

if df is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_col = {repr(x)} or df.columns[0]
    y_col = {repr(y)} or (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    if {repr(chart_type)} == 'histogram':
        ax.hist(df[x_col].dropna(), bins=30, color='#58a6ff', alpha=0.8, edgecolor='#111111')
    elif {repr(chart_type)} == 'scatter':
        ax.scatter(df[x_col], df[y_col], color='#58a6ff', alpha=0.6, s=20)
    elif {repr(chart_type)} == 'line':
        ax.plot(df[x_col], df[y_col], color='#58a6ff', linewidth=1.5)
    elif {repr(chart_type)} == 'box':
        df.boxplot(ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='#58a6ff', alpha=0.3),
                   medianprops=dict(color='#58a6ff'))
    elif {repr(chart_type)} == 'heatmap':
        numeric_df = df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True, cmap='YlOrBr', ax=ax)
    else:  # bar
        if df[x_col].dtype == 'object':
            counts = df[x_col].value_counts().head(20)
            ax.bar(counts.index, counts.values, color='#58a6ff', alpha=0.8)
        else:
            ax.bar(range(min(20, len(df))), df[y_col].head(20), color='#58a6ff', alpha=0.8)

    ax.set_title({repr(title)}, fontsize=14, pad=15)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col if {repr(chart_type)} != 'histogram' else 'Frequency')
    plt.tight_layout()
    plt.show()
    print(f"Visualization created: {repr(title)}")
else:
    print("No data available to visualize")
"""
