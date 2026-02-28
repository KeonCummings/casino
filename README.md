# Ca$ino

A data storyteller that thinks out loud. Ca$ino is an AI agent that doesn't just analyze data — it interrogates it, finds the story hiding in the numbers, and makes you feel it through visuals that land.

Powered by [Strands Agents SDK](https://github.com/strands-agents/sdk-python). Bring your own LLM key. Everything runs on your machine.

## Quick start

```bash
git clone https://github.com/KeonCummings/casino.git
cd casino
```

Create a `.env` file:

```bash
echo 'LLM_PROVIDER=anthropic' > .env
echo 'LLM_API_KEY=your-key-here' >> .env
```

**With Docker:**

```bash
docker compose up --build
```

**With Python (3.10+):**

```bash
cd backend
pip install -r requirements.txt
export $(cat ../.env | xargs)
python main.py
```

The API runs at `http://localhost:8000`.

## What it does

Ask Ca$ino to analyze data and it will execute code, create visualizations, train models, and save artifacts — all autonomously. It thinks like a data journalist: skeptical, curious, hunting for what's interesting.

```bash
# Create a workspace
curl -s -X POST localhost:8000/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name":"demo"}' | python3 -m json.tool

# Ask it something
curl -N -X POST localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"load the iris dataset and show me something interesting","workspaceId":"<id>"}'
```

## Configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `LLM_API_KEY` | — | Your API key (not needed for Ollama) |
| `LLM_MODEL` | auto | Model name (auto-selects per provider) |
| `LLM_BASE_URL` | — | Custom endpoint (required for Ollama if not localhost) |
| `CASINO_EXEC_TIMEOUT` | `30` | Code execution timeout (seconds) |

**Provider defaults:**

| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-20250514` |
| `openai` | `gpt-4o` |
| `ollama` | `llama3.1` |

## Tools

Ca$ino has 7 tools, all defined as Strands `@tool` decorators with auto-generated schemas:

| Tool | What it does |
|---|---|
| `execute_python_code` | Run Python with pandas, numpy, matplotlib, seaborn, sklearn, scipy |
| `load_dataset` | Load iris, titanic, wine, tips, penguins, etc. or generate synthetic data |
| `list_datasets` | List datasets in the workspace |
| `describe_dataset` | Get shape, dtypes, describe(), null counts |
| `create_visualization` | Generate charts from a dataset |
| `save_script` | Save reusable Python code to the scripts directory |
| `save_report` | Save analysis reports and summaries as markdown |

**Automatic artifact capture:**
- `plt.show()` saves PNGs to visualizations with unique timestamped filenames
- CSVs saved to `DATASETS_DIR` get metadata indexed automatically
- Models saved via `joblib.dump()` to `MODELS_DIR` appear in the models list
- Stray image/data files in the workspace root are moved to the right directory

## API

### Stream a query

```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "load the iris dataset and create a scatter plot", "workspaceId": "ws_1"}'
```

Returns SSE events:

```
data: {"type": "progress", "message": "Thinking..."}
data: {"type": "content/text/delta", "content": "Let me look at this..."}
data: {"type": "tool_call", "tool": "execute_python_code", "args": {"code": "..."}}
data: {"type": "tool_result", "tool": "execute_python_code", "result": "..."}
data: {"type": "visualization", "data": {"name": "plot_1719000000_1.png", ...}}
data: {"type": "done"}
```

### Workspaces

```bash
# Create
curl -X POST localhost:8000/workspaces -H "Content-Type: application/json" -d '{"name": "my-project"}'

# List
curl localhost:8000/workspaces

# Delete
curl -X DELETE localhost:8000/workspaces/{id}
```

### Artifacts

```bash
curl localhost:8000/workspaces/{id}/datasets
curl localhost:8000/workspaces/{id}/datasets/{name}/preview
curl localhost:8000/workspaces/{id}/visualizations
curl localhost:8000/workspaces/{id}/scripts
curl localhost:8000/workspaces/{id}/reports
curl localhost:8000/workspaces/{id}/models

# Serve files directly
curl localhost:8000/workspace/{id}/visualizations/plot.png --output plot.png
```

## Architecture

```
backend/
├── main.py           # FastAPI — SSE streaming, REST endpoints
├── agent.py          # Strands Agent — @tool functions, persona, streaming
├── executor.py       # Python subprocess execution + artifact capture
├── artifacts.py      # Filesystem CRUD for workspace data
├── config.py         # Provider config + Strands model factory
├── Dockerfile
└── requirements.txt
```

The agent is stateless per request — a fresh `Agent` instance is created for each `/stream` call. Tools communicate side-channel SSE events (tool results, visualizations) via a shared queue in `invocation_state`.

## Security

Code execution runs LLM-generated Python directly on your machine via subprocess with a 30-second timeout. There is no sandbox. Only run this locally with API keys you control.

## License

MIT
