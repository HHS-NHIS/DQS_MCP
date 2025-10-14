[README_PUBLIC.md](https://github.com/user-attachments/files/22907718/README_PUBLIC.md)

# DQS Health Data Tools (Core API, Web Widget & MCP Tool)

Welcome! This repository provides three pieces that make it easy to **ask plain‑language questions** about CDC/NCHS Data Query System (DQS) topics and get **concise answers** (estimates with confidence intervals when available):

- **Core API (`core/`)** — a FastAPI service that interprets a natural‑language question and returns a structured answer.
- **Web Widget (`widget/`)** — a tiny web app + proxy you can drop into any site; it sends your question to Core and renders a friendly reply.
- **MCP Tool (`mcp/`)** — a small JSON‑RPC process that lets MCP‑compatible clients call the Core API as a “tool.”

> If you’re here to deploy (GitHub + Railway), see **README_DEPLOY.md** in this repo.

---

## Quick Start (no installation, using Docker)

You’ll need **Docker** and **docker compose**.

```bash
# From the repo root
docker compose up --build
```

- **Widget UI** → open **http://localhost:8130/** and type a question, e.g.  
  _“Current asthma in adults by sex in the latest year available.”_
- **Core health check** → **http://localhost:8123/health**

> Stopping: press `Ctrl + C` in the terminal.

---

## What kinds of questions can I ask?

The system is tuned for DQS topic queries like:

- “**Current asthma in adults** by **sex** in the **latest year available**.”
- “What **percentage** of adults **received an influenza vaccination** in **2024**?”
- “**Current asthma in children** by **health insurance** in the **last available year**.”

When possible, the answer includes **estimate, 95% CI**, and a short **explanation** of year choices or fallbacks.

---

## Using the Web Widget

1. Run the widget (via Docker compose or your own hosting).
2. Open `http://<your-widget-host>/` and type a question.
3. The widget calls **`POST /ask`** on itself, which proxies to Core. No extra configuration is needed when using docker compose.

**Embedding elsewhere?** Host `widget/` as a small Node service and point it to your Core API via the environment variable `CORE_URL` (see “Configuration” below).

---

## Using the Core HTTP API (directly)

**Endpoint**

```
POST /v1/widget_text
Content-Type: application/json
{ "query": "Current asthma in adults by sex in the latest year available" }
```

**Minimal example with `curl`**

```bash
curl -s -X POST http://localhost:8123/v1/widget_text   -H "Content-Type: application/json"   -d '{"query":"Current asthma in adults by sex in the latest year available"}' | jq .
```

**Response shape (example)**

```json
{
  "engine_version": "x.y.z",
  "ms_elapsed": 123,
  "answer_text": "Current asthma in adults — 2024: Female 9.7% (95% CI 9.0–10.4), Male 6.7% (6.2–7.3)…",
  "topic": "Current asthma in adults",
  "dataset_id": "abcd-efgh",
  "group": "Sex",
  "years": ["2024"],
  "warnings": []
}
```

---

## Using the MCP Tool (for MCP‑compatible clients)

Run the tool service (Docker or Node). It exposes a JSON‑RPC method that relays to Core.

**Tools**
- `tools/list` → lists available tools
- `tools/call` with `name: "dqs.ask.core"` and `arguments: { "query": "..." }`

**Minimal request**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": { "name": "dqs.ask.core", "arguments": { "query": "flu shot among adults latest year" } }
}
```

**Result** contains the same fields as the Core API response.

---

## Configuration & Data Files

The Core API looks for JSON configuration files in the container’s working directory (by default `/app`). If you deploy with Docker or Railway using the provided Dockerfile, you **don’t need** to set extra variables as long as the files are kept in `core/`.

Common files:
- `standardized_dqs_full.json` (required main data table)
- `keyword_mappings2.json` (optional — for aliasing and demographic hints)
- `disambiguation_rules.json`
- `group_label_map.json`
- `dataset_map.json`
- `cdc_topic_links.json`
- `dataset_overrides.json`

**Optional environment variables** (Core):
- `DATA_PATH=/app/standardized_dqs_full.json` — set only if you move the main table.
- `KEYWORD_MAPPINGS_PATH=/app/keyword_mappings2.json` — set only if you want to pin the alias file path.

**Widget** environment variables:
- `CORE_URL` — where the widget proxies questions (defaults to the Core service under docker compose).
- `CORE_HEALTH` — optional health probe target.

> When running via docker compose, these are pre‑wired for you.

---

## Running without Docker (developers)

### Core
```bash
cd core
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Widget
```bash
cd widget
npm install
# If Core is not on the default compose URL, set:
#   set CORE_URL=http://localhost:8000/v1/widget_text    (Windows)
#   export CORE_URL=http://localhost:8000/v1/widget_text (macOS/Linux)
node server.js
```

### MCP
```bash
cd mcp
npm install
node server-http.js
```

---

## Data sources & caveats

- Answers come from **CDC/NCHS Data Query System (DQS)** datasets. Some topics can appear in **multiple datasets** (e.g., measured vs self‑reported), and the tool explains when it chooses a specific source or the “latest year available.”
- If a requested subgroup or multi‑group breakdown isn’t available in a dataset, the tool falls back to what is available and states why.
- Reliability flags/suppression rules depend on what’s provided by the published DQS aggregates; not all datasets expose the same metadata.

---

## Troubleshooting

- **Widget returns 502** → Check `CORE_URL` points to your Core API.
- **Core returns no data** → Confirm your `standardized_dqs_full.json` is present and current.
- **Slow responses** → Large queries can take a moment on cold start; scale the Core service if hosting publicly.

---

## Contributing

Issues and pull requests are welcome! Please describe your environment, reproduction steps, and include sample queries.

---

## License

See `LICENSE` in the repository.

---

## Deployment Guide

If you want to deploy to GitHub + Railway (one project, three services), see **README_DEPLOY.md** in this repo.
