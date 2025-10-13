# DQS15 — GitHub & Railway Deployment (Core + Widget + MCP)

This repo is a monorepo with **three services**:

- **core/** — Python FastAPI (`uvicorn`) with the deterministic query engine
- **widget/** — Node/Express HTTP server serving the UI and proxying `/ask` → Core
- **mcp/** — Minimal JSON‑RPC tool process that calls Core (optionally exposed over HTTP for smoke tests)

It already includes Dockerfiles for each service. Below are **copy‑paste** steps to push to GitHub and deploy each service to Railway.

---

## 0) Clean up & prepare locally

```bash
# from repo root
cd /mnt/data/DQS15_2/DQS15

# remove vendored node_modules from git history (kept locally)
echo "node_modules/" >> .gitignore
git rm -r --cached widget/node_modules mcp/node_modules 2>/dev/null || true
```

> Tip: Keep `data/standardized_dqs_full.json` in `core/` (already included) so `core` can build standalone. The `.gitignore` ignores large JSON in `data/` at the top level to keep the repo lean.

---

## 1) Push to GitHub

**Option A – using GitHub CLI**

```bash
git init
git add -A
git commit -m "Initial commit: core+widget+mcp with Dockerfiles"
git branch -M main
gh repo create DQS15 --private --source . --remote origin --push
```

**Option B – manual remote**

```bash
git init
git add -A
git commit -m "Initial commit: core+widget+mcp with Dockerfiles"
git branch -M main
# create an empty GitHub repo named DQS15, then:
git remote add origin https://github.com/<YOUR_ORG_OR_USER>/DQS15.git
git push -u origin main
```

---

## 2) Deploy to Railway (monorepo)

We’ll create **one Railway project** with **three services** built from subfolders via Docker.

### A) Create project & empty services

1. In Railway, create a **New → Empty Project**.  
2. Click **Create → Empty Service** three times; rename them to: `core`, `widget`, `mcp`.

### B) Point each service at the correct subfolder

Open each service → **Settings** → **Root Directory** and set:

- `core` → `/core`
- `widget` → `/widget`
- `mcp` → `/mcp`

> Railway auto‑detects the Dockerfile at each service root. You can also set the `RAILWAY_DOCKERFILE_PATH` variable if you rename the file.

### C) Connect GitHub

In each service → **Settings** → connect this GitHub repo and branch `main`.

### D) Domains

- Generate a **public domain** for **widget** (you’ll paste this in a browser).  
- `core` can be **private‑only** (use the *private* domain internally), or you can also generate a public domain for quick curl tests.
- `mcp` typically doesn’t need a public domain.

### E) Variables (wire services together)

Use **Reference Variables** so Railway keeps URLs up to date after redeploys.

**Widget** → Variables:

```
CORE_URL=https://${core.RAILWAY_PUBLIC_DOMAIN}/v1/widget_text
CORE_HEALTH=https://${core.RAILWAY_PUBLIC_DOMAIN}/health
```

> If you prefer **private networking**, use:
>
> `CORE_URL=http://${core.RAILWAY_PRIVATE_DOMAIN}:8000/v1/widget_text`  
> `CORE_HEALTH=http://${core.RAILWAY_PRIVATE_DOMAIN}:8000/health`
>
> (Private DNS ends in `.railway.internal`, and you include the container port.)

**MCP** → Variables:

```
CORE_URL=http://${core.RAILWAY_PRIVATE_DOMAIN}:8000/v1/widget_text
```

> You can also point MCP at the public domain:
>
> `CORE_URL=https://${core.RAILWAY_PUBLIC_DOMAIN}/v1/widget_text`

### F) Deploy

Click **Deploy** on each service. Railway will log:
```
==========================
Using detected Dockerfile!
==========================
```
when it builds from Docker.

### G) Smoke tests

- **Core**: visit `https://<core-public-domain>/health` → should return `ok` JSON.
- **Widget**: open `https://<widget-public-domain>/` → type a question and submit.
- **MCP** (optional HTTP probe): `curl https://<mcp-public-domain>/` should return a tiny JSON (or run via stdio in your toolchain).

---

## 3) Local dev with Docker Compose

```bash
docker compose up --build
# Core → http://localhost:8123/health
# Widget → http://localhost:8130/
```

Compose services:
- `core` on port 8123→8000
- `widget` on port 8130→8099 (proxies `/ask` to core)
- `mcp` (no public port by default)

---

## 4) Files you might tweak

- `core/Dockerfile` runs uvicorn and now binds to `$PORT` (Railway‑friendly).  
- `widget/server.js` already respects `process.env.PORT`.
- `mcp/server-http.js` uses `process.env.PORT`.

---

## 5) Troubleshooting

- **Widget 502 / “proxy_failed”** → Check `CORE_URL` variable. If using private networking, include `:8000`. If using public domain, use `https://...` without a port.
- **Core not healthy** → Open **Logs**; ensure `standardized_dqs_full.json` is present (bundled in `core/`), or set `DATA_PATH` to an in‑image location.
- **Dockerfile ignored** → Ensure file name is **`Dockerfile`** (capital D) at the service root.
- **Huge pushes** → Run `git rm -r --cached widget/node_modules mcp/node_modules` and commit again.

---

## Appendix: Why these settings?

- Railway monorepo deployment relies on the **Root Directory** per service.  
- Using Docker keeps local and prod parity.  
- Internal communication can use Railway’s **private networking** to avoid egress and improve performance.

Happy shipping!
