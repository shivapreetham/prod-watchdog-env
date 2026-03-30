# ProdWatchdog — Architecture & Project Guide

## What Are We Building?

**ProdWatchdog** is a simulated production incident response environment for AI agents.

The problem: a real on-call SRE engineer gets paged at 3am. They look at alerts, query logs, check metrics, identify the root cause, take the right remediation action, and declare the incident resolved. Can an AI agent do this?

We simulate that exact workflow — a microservice cluster with real-sounding symptoms, logs, red-herring alerts, cascading failures — and let an LLM agent navigate it. Graders measure if it diagnosed correctly and fixed efficiently.

This is the submission for the **Meta × Hugging Face OpenEnv Hackathon, Round 1 (deadline: April 8, 2026)**.

---

## Project File Structure

```
prod-watchdog-env/
│
├── models.py              ← Pydantic data models (Action, Observation)
├── inference.py           ← LLM agent + fallback policy (runs against live server)
├── openenv.yaml           ← OpenEnv manifest (tasks, action/obs schema, metadata)
├── pyproject.toml         ← Python package config
├── Dockerfile             ← Container definition for HF Spaces deployment
├── ARCHITECTURE.md        ← This file
├── README.md              ← Public-facing docs for judges
│
└── server/
    ├── app.py             ← FastAPI server (all HTTP endpoints)
    ├── environment.py     ← Core simulation logic + graders
    ├── requirements.txt   ← Server dependencies
    └── Dockerfile         ← Alternative server-only Dockerfile
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     inference.py                        │
│  (LLM Agent using OpenAI-compatible client)             │
│                                                         │
│  1. POST /reset?task_id=task1                           │
│  2. loop: format_obs → LLM → parse_action → POST /step  │
│  3. POST /grader → get score                            │
│  4. Repeat for task2, task3                             │
│  5. POST /baseline → trigger all 3 automatically        │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (localhost:7860 or HF Space URL)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   server/app.py                         │
│              FastAPI (port 7860)                        │
│                                                         │
│  OpenEnv standard endpoints:                            │
│    POST /reset   → env.reset(task_id)                   │
│    POST /step    → env.step(action)                     │
│    GET  /state   → env.state                            │
│    GET  /health  → 200 OK                               │
│                                                         │
│  Hackathon-required custom endpoints:                   │
│    GET  /tasks   → list of tasks + action schema        │
│    POST /grader  → score for completed episode          │
│    POST /baseline→ run expert policy, return all scores │
└────────────────────────┬────────────────────────────────┘
                         │ calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│              server/environment.py                      │
│         ProdWatchdogEnvironment class                   │
│                                                         │
│  reset(task_id) → loads SCENARIO → initializes state   │
│  step(action)   → _process_action() → reward + obs     │
│  state          → episode_id + step_count              │
│                                                         │
│  Module-level _EPISODE_STATE (shared across requests):  │
│    task_id, step_count, service_health,                 │
│    episode_history, circuit_breakers, resolved          │
│                                                         │
│  Graders:                                               │
│    grader_task1(history) → 0.0–1.0                      │
│    grader_task2(history) → 0.0–1.0                      │
│    grader_task3(history) → 0.0–1.0                      │
└─────────────────────────────────────────────────────────┘
```

---

## The Service Topology

The simulated cluster has 6 microservices with real dependency relationships:

```
                    ┌─────────────┐
                    │ api-gateway │ ← entry point
                    └──────┬──────┘
                    ┌──────┴───────┐
                    ▼              ▼
             ┌─────────────┐  ┌───────────────┐
             │auth-service │  │ order-service │
             └─────────────┘  └───────┬───────┘
                                 ┌────┴────┐
                                 ▼         ▼
                        ┌─────────────┐  ┌──────────────────┐
                        │  payment-   │  │inventory-service │
                        │  service    │  └──────────────────┘
                        └──────┬──────┘
                               ▼
                      ┌──────────────────┐
                      │notification-     │
                      │service           │
                      └──────────────────┘
```

Failures propagate upward. When payment-service goes down, order-service degrades. When auth-service saturates, api-gateway degrades. Circuit breakers break the propagation chain.

---

## The 3 Tasks

### Task 1 — Easy: Single Service Outage
| | |
|---|---|
| **What broke** | `order-service` bad deployment (v2.1.3 has missing env var) |
| **Symptoms** | 87% 502 rate on order-service, api-gateway slightly degraded |
| **Red herrings** | None — clean scenario |
| **Correct fix** | `rollback_deploy` on `order-service` |
| **Max steps** | 10 |
| **Grader** | investigate(0.3) + correct fix(0.3) + resolved(0.4) |

### Task 2 — Medium: Cascading Failure + Red Herring
| | |
|---|---|
| **What broke** | `auth-service` CPU spike (99%) from promo campaign → api-gateway degraded |
| **Symptoms** | auth-service down, api-gateway 23% error rate |
| **Red herring** | notification-service showing high latency — it's just campaign emails, unrelated |
| **Correct fix** | `scale_up` on `auth-service` |
| **Max steps** | 15 |
| **Grader** | root investigated(0.4) + correct fix(0.3) + cascade resolved(0.3) |

### Task 3 — Hard: Multi-Service Cascade + 2 Red Herrings
| | |
|---|---|
| **What broke** | `payment-service` DB connection leak → notification-service down → order-service degraded |
| **Symptoms** | notification-service down, order-service degraded, payment-service DB pool at 97% |
| **Red herrings** | auth-service OAuth 401s (provider-side, unrelated) + inventory-service slow queries (scheduled analytics job) |
| **Correct fix** | `enable_circuit_breaker(notification-service)` THEN `restart_service(payment-service)` |
| **Max steps** | 20 |
| **Grader** | root investigated(0.35) + circuit breaker(0.25) + correct fix(0.25) + efficiency(0.15) |

The efficiency component in Task 3 penalizes wasted steps: score 1.0 if ≤8 steps, linearly drops to 0.0 at 20 steps.

---

## Action Space

The agent can take 7 action types, each targeting a specific service:

| Action | Effect | Reward if correct | Reward if wrong |
|--------|--------|-------------------|-----------------|
| `query_logs` | Returns recent logs for service | +0.20 (root service) | -0.05 |
| `check_metrics` | Returns CPU/memory/error_rate/latency | +0.15 (root service) | -0.05 |
| `rollback_deploy` | Reverts to last stable version | +0.50 | 0.0 |
| `scale_up` | Adds instances, reduces CPU load | +0.50 | -0.10 (if healthy) |
| `restart_service` | Clears DB leaks, memory OOM | +0.50 | -0.10 (if healthy) |
| `enable_circuit_breaker` | Isolates service, stops cascade | +0.30 (correct svc) | +0.05 |
| `declare_resolved` | Ends episode | +0.50 (if truly fixed) | -0.10 (if wrong) |

---

## Observation Space

Every step, the agent sees:

```json
{
  "alerts": ["CRITICAL [order-service] Health check failed..."],
  "service_health": {
    "api-gateway": "healthy",
    "auth-service": "healthy",
    "order-service": "down",
    "payment-service": "healthy",
    "inventory-service": "healthy",
    "notification-service": "healthy"
  },
  "last_action_result": "Logs for order-service: [ERROR] startup failed...",
  "step_count": 2,
  "available_actions": ["query_logs", "check_metrics", ...],
  "done": false,
  "reward": 0.2
}
```

---

## How inference.py Works

```
inference.py
│
├── Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_BASE_URL from env vars
│
├── create_llm_client()        → OpenAI-compatible client pointing to HF API
│
├── format_observation(obs)    → converts JSON obs to human-readable prompt
│
├── parse_action(llm_response) → extracts JSON action from LLM text
│   └── fallback: {"action_type": "query_logs", "service": "api-gateway"}
│
├── run_task(task_id, client)
│   ├── POST /reset?task_id=task1
│   ├── loop (max 20 steps):
│   │   ├── format observation into system+user prompt
│   │   ├── call LLM → get action JSON
│   │   ├── POST /step with action
│   │   └── if done: break
│   └── POST /grader → return score
│
├── _run_fallback_task(task_id) → deterministic expert policy
│   ├── Task1: query_logs → check_metrics → rollback_deploy → declare_resolved
│   ├── Task2: query_logs → check_metrics → scale_up → declare_resolved
│   └── Task3: query_logs → check_metrics → enable_circuit_breaker → restart_service → declare_resolved
│
└── run_all_tasks()            → runs all 3, prints scores, never crashes
```

**Baseline scores (deterministic fallback):**
- Task 1: 1.0000
- Task 2: 1.0000
- Task 3: 1.0000
- Average: 1.0000

---

## HTTP API Reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/reset` | `{"task_id": "task1"}` | Initial observation |
| POST | `/step` | `{"action_type": "query_logs", "service": "order-service"}` | Observation + reward + done |
| GET | `/state` | — | `{episode_id, step_count}` |
| GET | `/health` | — | `{"status": "ok"}` |
| GET | `/tasks` | — | Task list + action schema |
| POST | `/grader` | `{"task_id": "task1"}` (optional) | `{score, steps_taken, episode_id}` |
| POST | `/baseline` | — | `{task1, task2, task3, average}` |
| GET | `/schema` | — | JSON schemas |
| WS | `/ws` | — | WebSocket stateful session |

---

## What's Already Built ✅

- [x] Full environment simulation (`server/environment.py`) — 3 scenarios, all actions, reward shaping, service healing, circuit breakers
- [x] Pydantic models (`models.py`) — typed Action and Observation
- [x] FastAPI server (`server/app.py`) — all standard + custom hackathon endpoints
- [x] `openenv.yaml` — valid manifest with task definitions and schemas
- [x] `inference.py` — LLM agent with deterministic fallback, reads from env vars
- [x] `Dockerfile` — containerized, port 7860, health check
- [x] `pyproject.toml` — proper Python package structure
- [x] `README.md` — environment docs, action/obs space, task descriptions, baseline scores
- [x] Server imports verified working locally

---

## What Still Needs To Be Done

### Critical (must before submission)
- [ ] **Docker Desktop** — install it, test `docker build && docker run` locally
- [ ] **HF Space** — create Docker Space, tag it `openenv`, push code
- [ ] **Set HF Space secrets** — `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] **Run openenv validate** — `openenv validate openenv.yaml`
- [ ] **Live endpoint test** — hit `/reset`, `/step`, `/tasks`, `/grader`, `/baseline` on HF Space URL
- [ ] **Submit** — GitHub repo + HF Space URL on dashboard

### Quality improvements (will improve score)
- [ ] **More realistic log messages** — make Task 3 logs harder to distinguish signal from noise
- [ ] **Seed support** — add deterministic seeding so episodes are reproducible
- [ ] **Step penalties** — currently max_steps just truncates; penalize approaching limit
- [ ] **/docs endpoint** — FastAPI auto-docs should be accessible for judges
- [ ] **README baseline scores table** — add actual numbers from a real run

### Stretch goals (creativity score +10%)
- [ ] **Task 4** — database migration failure scenario (schema mismatch, rollback required)
- [ ] **Random scenario variants** — slight log/metric randomization per seed (makes it harder to memorize)
- [ ] **Multi-step cascade scoring** — reward partial cascade resolution per service healed

---

## System Readiness

| Tool | Status | Version |
|------|--------|---------|
| Python | ✅ Ready | 3.10.19 |
| openenv-core | ✅ Installed | 0.2.2 |
| FastAPI | ✅ Installed | 0.128.7 |
| uvicorn | ✅ Installed | 0.40.0 |
| openai (client) | ✅ Installed | 2.30.0 |
| pydantic | ✅ Installed | 2.12.5 |
| httpx | ✅ Installed | 0.28.1 |
| git | ✅ Ready | 2.51.2 |
| huggingface_hub | ✅ Installed + Logged in | 0.36.2 |
| server imports | ✅ Pass | — |
| **Docker** | ❌ NOT FOUND | Install Docker Desktop |

**Docker is the only blocker.** Everything else is ready.

---

## How to Run Locally Right Now

```bash
# Start the server
cd e:/meta-hackathon/prod-watchdog-env
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal — test it
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task1"}'

# Run inference agent (set env vars first)
set API_BASE_URL=https://api-inference.huggingface.co/v1
set MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
set HF_TOKEN=hf_your_token_here
python inference.py
```

## How to Deploy to HF Spaces

```bash
# 1. Create a new Space on huggingface.co
#    - Type: Docker
#    - Tag: openenv

# 2. Push code
cd e:/meta-hackathon/prod-watchdog-env
git remote add hf https://huggingface.co/spaces/shivapreetham/prod-watchdog-env
git push hf master

# 3. Set secrets in Space settings:
#    API_BASE_URL, MODEL_NAME, HF_TOKEN
```
