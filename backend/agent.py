"""LLM agent loop — call model with tools, stream SSE events, handle tool_use cycles."""

import json
import os
import time
from typing import AsyncGenerator

from config import LLMConfig
from executor import execute_python
from artifacts import (
    ensure_workspace,
    list_datasets as list_workspace_datasets,
    list_visualizations as list_workspace_vizs,
    save_dataset_meta,
)
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a data science assistant. You help users analyze data, create visualizations, build models, and generate insights.

You have access to a Python execution environment with pandas, numpy, matplotlib, seaborn, scikit-learn, and scipy.

Key workspace directories available in code:
- DATASETS_DIR: save/load CSV files here
- VISUALIZATIONS_DIR: saved charts appear here (plt.show() auto-saves)
- SCRIPTS_DIR: save reusable scripts here
- REPORTS_DIR: save markdown/text reports here
- MODELS_DIR: save trained models here (pickle, joblib, etc.)

Guidelines:
- Always use plt.show() after creating plots — it auto-saves to the visualizations directory
- When loading well-known datasets, use seaborn.load_dataset() or sklearn.datasets
- Save processed data as CSV to DATASETS_DIR so users can see it in the Data tab
- Write clear, commented code
- Provide concise explanations of your findings
- When asked to analyze data, start with descriptive statistics, then visualize"""


async def run_agent(
    prompt: str,
    workspace_id: str,
    workspace_root: str,
    config: LLMConfig,
    message_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """Run the agent loop, yielding SSE-formatted strings."""

    workspace_dir = str(ensure_workspace(workspace_root, workspace_id))

    messages = []
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": prompt})

    # Convert tool definitions to provider format
    tools = _format_tools_for_provider(config.provider)

    while True:
        # Call LLM
        yield _sse("progress", {"message": "Thinking..."})

        response = await _call_llm(messages, tools, config)

        if response is None:
            yield _sse("error", {"error": "LLM call failed"})
            break

        # Process response content blocks
        assistant_content = []
        has_tool_use = False

        for block in response.get("content", []):
            if block["type"] == "text":
                text = block["text"]
                assistant_content.append(block)
                yield _sse("content/text/delta", {"content": text})

            elif block["type"] == "tool_use":
                has_tool_use = True
                assistant_content.append(block)

                tool_name = block["name"]
                tool_input = block["input"]
                tool_use_id = block["id"]

                yield _sse(
                    "tool_call",
                    {"tool": tool_name, "args": tool_input, "tool_use_id": tool_use_id},
                )

                # Execute the tool
                result = await _execute_tool(
                    tool_name, tool_input, workspace_dir, workspace_root, workspace_id, config
                )

                yield _sse(
                    "tool_result",
                    {"tool": tool_name, "result": result["output"]},
                )

                # Emit visualizations if any were created
                for artifact in result.get("artifacts", []):
                    if artifact["category"] == "visualizations":
                        yield _sse(
                            "visualization",
                            {
                                "data": {
                                    "name": artifact["name"],
                                    "imageUrl": f"/workspace/{workspace_id}/visualizations/{artifact['name']}",
                                    "format": artifact["name"].rsplit(".", 1)[-1] if "." in artifact["name"] else "png",
                                }
                            },
                        )

                # Add assistant message + tool result to conversation
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": _truncate(json.dumps(result["output"]) if not isinstance(result["output"], str) else result["output"], 8000),
                            }
                        ],
                    }
                )
                assistant_content = []
                break  # Re-enter the loop for next LLM call

        if not has_tool_use:
            # No tool use — agent is done
            break

    yield _sse("done", {})


async def _call_llm(
    messages: list[dict],
    tools: list[dict],
    config: LLMConfig,
) -> dict | None:
    """Call the LLM provider and return the response."""

    if config.provider == "anthropic":
        return await _call_anthropic(messages, tools, config)
    elif config.provider == "openai":
        return await _call_openai(messages, tools, config)
    elif config.provider == "ollama":
        return await _call_ollama(messages, tools, config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


async def _call_anthropic(
    messages: list[dict],
    tools: list[dict],
    config: LLMConfig,
) -> dict | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.api_key)

        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        return {
            "content": [_block_to_dict(b) for b in response.content],
            "stop_reason": response.stop_reason,
        }
    except Exception as e:
        print(f"[agent] Anthropic error: {e}")
        return None


async def _call_openai(
    messages: list[dict],
    tools: list[dict],
    config: LLMConfig,
) -> dict | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.api_key)

        # Convert messages to OpenAI format
        oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            oai_messages.append(_to_openai_message(msg))

        # Convert tools to OpenAI format
        oai_tools = [_to_openai_tool(t) for t in tools]

        response = client.chat.completions.create(
            model=config.model,
            messages=oai_messages,
            tools=oai_tools if oai_tools else None,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        choice = response.choices[0]
        content_blocks = []

        if choice.message.content:
            content_blocks.append({"type": "text", "text": choice.message.content})

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                )

        return {
            "content": content_blocks,
            "stop_reason": choice.finish_reason,
        }
    except Exception as e:
        print(f"[agent] OpenAI error: {e}")
        return None


async def _call_ollama(
    messages: list[dict],
    tools: list[dict],
    config: LLMConfig,
) -> dict | None:
    try:
        import httpx

        base_url = config.base_url or "http://localhost:11434"

        # Convert to Ollama format (OpenAI-compatible)
        ollama_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            ollama_messages.append(_to_openai_message(msg))

        payload = {
            "model": config.model,
            "messages": ollama_messages,
            "stream": False,
        }

        # Ollama tool support is model-dependent
        oai_tools = [_to_openai_tool(t) for t in tools]
        if oai_tools:
            payload["tools"] = oai_tools

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        content_blocks = []

        if msg.get("content"):
            content_blocks.append({"type": "text", "text": msg["content"]})

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"tool_{int(time.time()*1000)}",
                        "name": tc["function"]["name"],
                        "input": tc["function"]["arguments"]
                        if isinstance(tc["function"]["arguments"], dict)
                        else json.loads(tc["function"]["arguments"]),
                    }
                )

        return {
            "content": content_blocks,
            "stop_reason": data.get("done_reason", "stop"),
        }
    except Exception as e:
        print(f"[agent] Ollama error: {e}")
        return None


def _block_to_dict(block) -> dict:
    """Convert Anthropic content block to dict."""
    if hasattr(block, "text"):
        return {"type": "text", "text": block.text}
    elif hasattr(block, "name"):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": "text", "text": str(block)}


async def _execute_tool(
    name: str,
    args: dict,
    workspace_dir: str,
    workspace_root: str,
    workspace_id: str,
    config: LLMConfig,
) -> dict:
    """Execute a tool and return its result."""

    if name == "execute_python":
        code = args.get("code", "")
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

        return {"output": output, "artifacts": result.artifacts}

    elif name == "load_dataset":
        dataset_name = args.get("name", "iris")
        code = _generate_load_dataset_code(dataset_name, args.get("description"))
        result = execute_python(code, workspace_dir, timeout=30)

        for artifact in result.artifacts:
            if artifact["category"] == "datasets":
                _generate_csv_meta(workspace_root, workspace_id, workspace_dir, artifact["name"])

        output = result.stdout or f"Dataset '{dataset_name}' loaded successfully."
        if result.stderr and result.returncode != 0:
            output = f"Error loading dataset: {result.stderr}"

        return {"output": output, "artifacts": result.artifacts}

    elif name == "list_datasets":
        datasets = list_workspace_datasets(workspace_root, workspace_id)
        if not datasets:
            return {"output": "No datasets in workspace yet.", "artifacts": []}
        lines = [f"- {d['name']} ({d['rows']} rows, {d['columns']} cols)" for d in datasets]
        return {"output": "Datasets:\n" + "\n".join(lines), "artifacts": []}

    elif name == "describe_dataset":
        ds_name = args.get("name", "")
        code = f"""
import pandas as pd
import os

# Find the dataset
files = os.listdir(DATASETS_DIR)
target = None
for f in files:
    if f == {repr(ds_name)} or os.path.splitext(f)[0] == {repr(ds_name)}:
        target = f
        break

if target is None:
    print(f"Dataset '{repr(ds_name)}' not found. Available: {{files}}")
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
        return {"output": output, "artifacts": []}

    elif name == "create_visualization":
        code = args.get("code", "")
        if not code:
            code = _generate_viz_code(args)
        result = execute_python(code, workspace_dir, timeout=30)

        output = result.stdout or "(visualization created)"
        if result.stderr and result.returncode != 0:
            output = f"Error: {result.stderr}"

        return {"output": output, "artifacts": result.artifacts}

    else:
        return {"output": f"Unknown tool: {name}", "artifacts": []}


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

# Dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#111111'
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['text.color'] = '#f5f0e8'
plt.rcParams['axes.labelcolor'] = '#c4bdb3'
plt.rcParams['xtick.color'] = '#c4bdb3'
plt.rcParams['ytick.color'] = '#c4bdb3'

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
        ax.hist(df[x_col].dropna(), bins=30, color='#d4a843', alpha=0.8, edgecolor='#111111')
    elif {repr(chart_type)} == 'scatter':
        ax.scatter(df[x_col], df[y_col], color='#d4a843', alpha=0.6, s=20)
    elif {repr(chart_type)} == 'line':
        ax.plot(df[x_col], df[y_col], color='#d4a843', linewidth=1.5)
    elif {repr(chart_type)} == 'box':
        df.boxplot(ax=ax, patch_artist=True,
                   boxprops=dict(facecolor='#d4a843', alpha=0.3),
                   medianprops=dict(color='#d4a843'))
    elif {repr(chart_type)} == 'heatmap':
        numeric_df = df.select_dtypes(include='number')
        sns.heatmap(numeric_df.corr(), annot=True, cmap='YlOrBr', ax=ax)
    else:  # bar
        if df[x_col].dtype == 'object':
            counts = df[x_col].value_counts().head(20)
            ax.bar(counts.index, counts.values, color='#d4a843', alpha=0.8)
        else:
            ax.bar(range(min(20, len(df))), df[y_col].head(20), color='#d4a843', alpha=0.8)

    ax.set_title({repr(title)}, fontsize=14, pad=15)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col if {repr(chart_type)} != 'histogram' else 'Frequency')
    plt.tight_layout()
    plt.show()
    print(f"Visualization created: {repr(title)}")
else:
    print("No data available to visualize")
"""


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


def _format_tools_for_provider(provider: str) -> list[dict]:
    """Format tool definitions for the LLM provider."""
    if provider == "anthropic":
        return ALL_TOOLS
    elif provider in ("openai", "ollama"):
        return [_to_openai_tool(t) for t in ALL_TOOLS]
    return ALL_TOOLS


def _to_openai_tool(tool: dict) -> dict:
    """Convert Anthropic tool format to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }


def _to_openai_message(msg: dict) -> dict:
    """Convert message to OpenAI format."""
    if isinstance(msg.get("content"), list):
        # Tool result or multi-part content
        parts = msg["content"]
        for part in parts:
            if part.get("type") == "tool_result":
                return {
                    "role": "tool",
                    "tool_call_id": part.get("tool_use_id", ""),
                    "content": part.get("content", ""),
                }
        # Multi-part text
        text = " ".join(p.get("text", str(p.get("content", ""))) for p in parts)
        return {"role": msg["role"], "content": text}
    return {"role": msg["role"], "content": msg.get("content", "")}


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 100] + f"\n... (truncated, {len(s)} total chars)"


def _sse(event_type: str, data: dict) -> str:
    """Format an SSE event line."""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload)}\n\n"
