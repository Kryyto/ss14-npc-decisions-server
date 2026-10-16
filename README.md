# SS14 NPC Decisions — server

Local REST API around [THX-01](https://huggingface.co/doofz/THX-01) (Apache-2.0, calibrated
non-autoregressive typed decisions), meant to run on the same machine as a Space Station 14 server so
androids can classify what players say to them.

The SS14 side (`AndroidConversationSystem`) sends each addressed message to `POST /v1/decide` at
`conversationEndpoint` (`http://localhost:8095` by default), with the job context as `state` and two
questions: an intent `choice` and a tone `score`. Intents, replies and in-game effects all live on the
SS14 side; this service only answers typed questions. If it is down or slower than the SS14 timeout,
androids fall back to their local rules, so the game never depends on it.

## Run

Requirements: Python 3.10+, about 650 MB of disk for the model (downloaded once into the Hugging Face
cache on first start) and roughly 1.5–2 GB of RAM. A GPU is optional.

```powershell
git clone https://github.com/Kryyto/ss14-npc-decisions-server.git
cd ss14-npc-decisions-server
.\run_local.ps1            # add -Cuda on a machine with an NVIDIA GPU
```

```bash
git clone https://github.com/Kryyto/ss14-npc-decisions-server.git
cd ss14-npc-decisions-server
./run_local.sh             # ./run_local.sh --cuda on a machine with an NVIDIA GPU
```

The first run creates `.venv`, installs torch (CPU wheel by default) and `requirements.txt`, copies
`.env.example` to `.env`, then starts the server. Later runs reuse the environment and only reinstall
when `requirements.txt` or the torch flavour changes.

Check it:

```bash
curl http://localhost:8095/health
```

Options: `.\run_local.ps1 -ListenHost 127.0.0.1 -Port 8095 -Threads 2 [-Cuda]`, or
`./run_local.sh [--cuda] 127.0.0.1 8095 2`. Keep `127.0.0.1` unless the SS14 server runs on another
machine; in that case set `THX01_API_KEY` in `.env` and the same value in the SS14 server's
`THX01_API_KEY` environment variable.

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `THX01_MODEL` | `doofz/THX-01` | Hugging Face id or local checkpoint path. |
| `THX01_DEVICE` | auto | `cuda` or `cpu`; empty picks CUDA when available. |
| `THX01_API_KEY` | empty | When set, `/v1/*` requires `Authorization: Bearer <key>`. |
| `HOST`, `PORT` | `127.0.0.1`, `8095` | Overridden by the script arguments. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `2` (script) | Keeps CPU inference from starving the game server. |

## API

- `GET /health` → `{"status": "ok" | "loading", "device": ...}`
- `POST /v1/decide` with `{"state": ..., "questions": {...}}` → `{"answers": {...}, "detected": {...}, "latency_ms": ...}`
- `POST /v1/decide/batch`, `GET /v1/models`, `GET /v1/presets`; Swagger UI on `/docs`.

Question types: `choice`, `noul`, `score`, `number`, `excerpt` (`"cite": true` on any), plus the server-side
extensions `"internal": true` (asked but hidden from the answer) and `"type": "rules"` (decision derived from
other answers). See `app/main.py`.
