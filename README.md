# Casino

A local data-science agent that executes Python code, creates visualizations, and manages datasets — all through a streaming API.

Bring your own LLM key. Everything runs on your machine.

## Quick start

```bash
git clone https://github.com/keon/casino.git
cd casino
cp .env.example .env   # add your API key
```

**With Docker:**

```bash
docker compose up
```

**Without Docker:**

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The API runs at `http://localhost:8000`.

## Configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `LLM_API_KEY` | — | Your API key (not needed for Ollama) |
| `LLM_MODEL` | auto | Model name (auto-selects per provider) |
| `CASINO_SANDBOX` | `subprocess` | `subprocess` or `docker` |
| `CASINO_EXEC_TIMEOUT` | `30` | Code execution timeout (seconds) |

## API

### Stream a query

```bash
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "load the iris dataset and create a scatter plot", "workspaceId": "ws_1"}'
```

Returns SSE events:

```
data: {"type": "progress", "message": "Thinking..."}
data: {"type": "content/text/delta", "content": "I'll load the iris dataset..."}
data: {"type": "tool_call", "tool": "execute_python", "args": {"code": "..."}}
data: {"type": "tool_result", "tool": "execute_python", "result": "..."}
data: {"type": "visualization", "data": {"name": "plot_1", "imageUrl": "/workspace/ws_1/visualizations/plot_1.png"}}
data: {"type": "done"}
```

### Workspaces

```bash
# Create
curl -X POST http://localhost:8000/workspaces -H "Content-Type: application/json" -d '{"name": "my-project"}'

# List
curl http://localhost:8000/workspaces

# Delete
curl -X DELETE http://localhost:8000/workspaces/ws_123
```

### Artifacts

```bash
# Datasets
curl http://localhost:8000/workspaces/ws_1/datasets
curl http://localhost:8000/workspaces/ws_1/datasets/iris/preview

# Visualizations
curl http://localhost:8000/workspaces/ws_1/visualizations

# Scripts / Reports / Models
curl http://localhost:8000/workspaces/ws_1/scripts
curl http://localhost:8000/workspaces/ws_1/reports
curl http://localhost:8000/workspaces/ws_1/models

# Serve files directly
curl http://localhost:8000/workspace/ws_1/visualizations/plot_1.png --output plot.png
```

## Tools

The agent has these tools:

| Tool | What it does |
|---|---|
| `execute_python` | Run arbitrary Python with pandas, numpy, matplotlib, seaborn, sklearn, scipy |
| `load_dataset` | Load iris, titanic, wine, tips, penguins, etc. or generate synthetic data |
| `list_datasets` | List datasets in the workspace |
| `describe_dataset` | Get shape, dtypes, describe(), null counts |
| `create_visualization` | Generate charts from a dataset |

Code execution auto-captures artifacts:
- `plt.show()` saves PNGs to the visualizations directory
- CSVs saved to `DATASETS_DIR` appear in datasets
- Any image files in the workspace root are moved to visualizations

## Security

**Default mode (`subprocess`)**: The agent executes LLM-generated Python code directly on your machine via subprocess. There is a 30-second timeout but no sandbox. Only run this locally with API keys you control.

## Architecture

```
backend/
├── main.py           # FastAPI — SSE streaming, REST endpoints, static files
├── agent.py          # LLM loop — call model → tools → stream events
├── executor.py       # Python subprocess execution + artifact capture
├── artifacts.py      # Filesystem CRUD for workspace data
├── config.py         # Provider config (Anthropic/OpenAI/Ollama)
└── tools/
    ├── code_interpreter.py
    ├── dataset_tools.py
    └── visualization_tools.py
```

## License

MIT
