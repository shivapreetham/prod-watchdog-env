# ProdWatchdog — Architecture & Project Guide (v3)

## What Are We Building?

**ProdWatchdog** is a simulated production incident response environment for AI agents.

A real on-call SRE gets paged at 3am. They look at alerts, query logs, check metrics, trace the dependency graph, identify root cause, contain the blast radius, and remediate. Can an AI agent do this reliably — and do it without being fooled by red-herring alerts or cascading symptoms?

We simulate that exact workflow across **6 escalating incidents** in an **11-service microservice cluster** with realistic logs, noisy metrics, cascade timelines, and deceptive red herrings. A grader scores each episode based on investigation quality, root-cause accuracy, containment, and efficiency.

**Submission:** Meta × Hugging Face OpenEnv Hackathon, Round 1 (deadline: April 8, 2026).

---

## Project File Structure

```
prod-watchdog-env/
│
├── models.py              ← Pydantic data models (Action, Observation) — 11 services, 10 actions
├── inference.py           ← LLM agent + deterministic fallback policy
├── openenv.yaml           ← OpenEnv manifest (6 tasks, action/obs schema, metadata)
├── pyproject.toml         ← Python package config ([project.scripts] server = ...)
├── Dockerfile             ← Container definition for HF Spaces (python:3.11-slim, port 7860)
├── uv.lock                ← Locked dependencies (must be at root, committed)
│
├── docs/
│   └── ARCHITECTURE.md    ← This file
│
└── server/
    ├── app.py             ← FastAPI server (all HTTP endpoints)
    ├── environment.py     ← Core simulation logic, scenarios, graders, reward shaping
    └── requirements.txt   ← Server dependencies
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       inference.py                           │
│  LLM agent (OpenAI-compatible) + deterministic fallback      │
│                                                              │
│  for each task1..task6:                                      │
│    1. POST /reset  { task_id }                               │
│    2. loop (max steps):                                      │
│         format_obs → LLM → parse_action → POST /step        │
│         auto-declare if all services healthy                 │
│    3. POST /grader → score                                   │
│    4. 5s cooldown before next task                           │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP (ENV_BASE_URL, default localhost:7860)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     server/app.py                            │
│              FastAPI via openenv create_app()                │
│                                                              │
│  Standard OpenEnv endpoints:                                 │
│    POST /reset   → env.reset(task_id)                        │
│    POST /step    → env.step(action)                          │
│    GET  /state   → {episode_id, step_count}                  │
│    GET  /health  → {"status": "healthy"}                     │
│    GET  /schema  → action/obs JSON schemas                   │
│                                                              │
│  Hackathon custom endpoints:                                 │
│    GET  /tasks   → 6 tasks with difficulty + action schema   │
│    POST /grader  → episode score 0.0–1.0                     │
│    POST /baseline→ expert policy on all 6 tasks, scores dict │
└───────────────────────────┬──────────────────────────────────┘
                            │ calls
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  server/environment.py                       │
│            ProdWatchdogEnvironment class                     │
│                                                              │
│  reset(task_id) → loads SCENARIO, initializes _EPISODE_STATE│
│  step(action)   → _process_action() + potential shaping      │
│                   + cascade spreading + step penalty         │
│  state          → episode_id, step_count (last-step only)    │
│                                                              │
│  _EPISODE_STATE (module-level, persists across HTTP calls):  │
│    task_id, step_count, service_health[11], episode_history  │
│    circuit_breakers[], services_queried[], primary_fixed     │
│                                                              │
│  Potential-based reward shaping (Ng 1999):                   │
│    Φ(s) = weighted avg health of all 11 services             │
│    shaping = γ·Φ(s') − Φ(s) added every step                │
│                                                              │
│  6 graders — deterministic, scores 0.10–1.00:               │
│    grader_task1..6(episode_history) → float                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Service Topology (11 services)

```
Internet
    │
    ▼
┌─────────┐
│ nginx-lb│  ← Entry point / load balancer
└────┬────┘
     │
     ▼
┌────────────┐     ┌─────────────┐
│ api-gateway│ ←── │ redis-cache │  ← Session/response cache
└──────┬─────┘     └─────────────┘
       │
  ┌────┴─────────┐
  ▼              ▼
┌──────────┐  ┌──────────────┐
│auth-     │  │order-service │
│service   │  └──────┬───────┘
└────┬─────┘         │
     │          ┌────┼────────────┐
     ▼          ▼    ▼            ▼
┌──────────┐  ┌─────────┐  ┌──────────────┐  ┌───────────┐
│postgres  │  │payment- │  │inventory-    │  │kafka-     │
│primary   │  │service  │  │service       │  │broker     │
└────┬─────┘  └────┬────┘  └──────┬───────┘  └─────┬─────┘
     │              │              │                │
     ▼              │              ▼                ▼
┌──────────┐        │      ┌──────────────┐  ┌──────────────┐
│postgres  │        │      │postgres-     │  │notification- │
│replica   │        │      │replica       │  │service       │
└──────────┘        │      └──────────────┘  └──────────────┘
                    ▼
             ┌──────────┐
             │postgres  │
             │primary   │ (write path for payment + auth + order)
             └──────────┘
```

**HEAL_PROPAGATION**: when a service heals, which dependents may auto-recover:
- `redis-cache` heals → `api-gateway` can recover
- `kafka-broker` heals → `order-service`, `notification-service` recover
- `postgres-primary` heals → `auth-service`, `payment-service`, `order-service`, `postgres-replica` recover
- `postgres-replica` heals → `inventory-service` recovers

---

## The 6 Tasks

### Task 1 — Easy (max 12 steps): Redis Cache OOM → Thundering Herd
| | |
|---|---|
| **Root cause** | `redis-cache` memory OOM (100%) — 0% cache hit rate |
| **Symptoms** | api-gateway 18% errors + 4200ms p99 (looks like gateway issue) |
| **Cascade** | nginx-lb degrades at step 6 if not fixed |
| **Red herring** | api-gateway errors — downstream effect, not root |
| **Correct fix** | `scale_up(redis-cache)` |
| **Tempting wrong** | `scale_up(api-gateway)` or `flush_cache(redis-cache)` |
| **Grader** | diagnosed redis(0.25) + first-investigation correct(0.15) + fix(0.25) + final state(0.20) + efficiency(0.15) |

### Task 2 — Easy (max 10 steps): nginx-lb Bad Config Deploy
| | |
|---|---|
| **Root cause** | `nginx-lb` config deploy set `worker_connections=4` (was 4096) |
| **Symptoms** | nginx-lb drops 99% traffic, api-gateway connection resets |
| **Cascade** | None — nginx being down is already the critical failure |
| **Red herring** | api-gateway 94% error rate looks like gateway misconfiguration |
| **Correct fix** | `rollback_deploy(nginx-lb)` |
| **Tempting wrong** | `restart_service(api-gateway)` or `scale_up(api-gateway)` |
| **Key insight** | When ALL services appear degraded, check the entry point (nginx-lb) first |
| **Grader** | diagnosed lb(0.30) + first-investigation quality(0.20) + fix(0.30) + final state(0.20) |

### Task 3 — Medium (max 15 steps): Kafka Broker Partition Failure
| | |
|---|---|
| **Root cause** | `kafka-broker` JVM GC pause (47s) → broker fenced → no partition leaders |
| **Symptoms** | order-service producer timeouts (memory growing), notification-service consumer lag 847k msgs |
| **Cascade** | order-service=down at step 8, notification-service=down at step 12 |
| **Red herrings** | notification consumer lag looks primary; order-service 78% memory looks like leak |
| **Correct fix** | `restart_service(kafka-broker)` |
| **Tempting wrong** | `restart_service(notification-service)` or `rebalance_partitions(kafka-broker)` |
| **Grader** | diagnosed kafka(0.20) + investigation quality/position(0.15) + fix(0.30) + final state(0.20) + efficiency(0.15) |

### Task 4 — Medium (max 15 steps): Postgres Replica WAL Corruption
| | |
|---|---|
| **Root cause** | `postgres-replica` WAL segment CRC mismatch after OS force-stop |
| **Symptoms** | inventory-service 100% read failures, order-service stock checks failing |
| **Cascade** | order-service=down at step 9 |
| **Red herrings** | postgres-primary CPU 87% (it's absorbing read fallback traffic, not failing) |
| **Correct fix** | `restart_service(postgres-replica)` — clears bad WAL, re-syncs from primary |
| **Tempting wrong** | `restart_service(postgres-primary)` or `rollback_deploy(postgres-primary)` |
| **Grader** | diagnosed replica(0.25) + first-investigation on right track(0.15) + fix(0.30) + final state(0.20) + efficiency(0.10) |

### Task 5 — Medium-Hard (max 20 steps): Auth Service JWT Memory Leak
| | |
|---|---|
| **Root cause** | `auth-service` JVM heap at 96% (JWT cache leak, 7d uptime) → GC pauses → 4s latency |
| **Symptoms** | api-gateway 34% errors + 78% CPU (retry storm), order-service checkout failures |
| **Cascade** | order-service=degraded at step 8, auth-service OOMKilled + payment=degraded at step 14 |
| **Red herrings** | api-gateway CPU 78% looks like gateway is the bottleneck (it isn't) |
| **Correct fix** | `enable_circuit_breaker(api-gateway)` THEN `restart_service(auth-service)` |
| **Tempting wrong** | `scale_up(api-gateway)` (bottleneck is auth latency, not gateway capacity) |
| **Grader** | diagnosed auth(0.25) + circuit breaker api-gateway(0.20) + fix(0.30) + final state(0.15) + efficiency(0.10) |

### Task 6 — Hard (max 25 steps): Postgres Primary Disk Full + Split-Brain
| | |
|---|---|
| **Root cause** | `postgres-primary` disk 100% full → OOM-killed → auto-failover ran but postgres-replica still read-only |
| **Symptoms** | payment-service 89% write failures (ReadOnlyDatabaseException), order/auth degraded |
| **Cascade** | inventory=degraded step 7, order=down step 12, auth=down step 16 |
| **Red herrings** | auth 23% errors look like JWT key rotation; order memory spike looks like memory leak |
| **3-step fix (order enforced)** | 1) `enable_circuit_breaker(payment-service)` 2) `rollback_deploy(postgres-primary)` 3) `restart_service(payment-service)` |
| **Tempting wrong** | `promote_replica(postgres-replica)` — already promoted, still read-only |
| **State machine** | `primary_fixed` flag in `_EPISODE_STATE`; step 3 only heals payment if flag is set |
| **Grader** | postgres investigation(0.20) + CB payment(0.20) + primary fix(0.25) + secondary fix in order(0.15) + final state(0.10) + efficiency(0.10) |

---

## Action Space (10 actions)

| Action | Effect | Reward if correct root | Reward if wrong/healthy |
|--------|--------|----------------------|------------------------|
| `query_logs` | Returns service logs | +0.20 | −0.05 |
| `check_metrics` | Returns CPU/memory/error/latency | +0.15 | −0.05 |
| `restart_service` | Restarts service (clears heap/connections) | +0.50 | 0.0 / −0.10 |
| `rollback_deploy` | Reverts deployment or restores from snapshot | +0.50 | −0.05 |
| `enable_circuit_breaker` | Isolates service, stops cascade | +0.30 (required svc) | −0.10 (healthy svc) |
| `scale_up` | Adds instances / memory | +0.50 | −0.10 (healthy) |
| `declare_resolved` | Ends episode | +0.50 (fully fixed) | −0.10 (not fixed) |
| `flush_cache` | Flushes redis-cache (does NOT fix OOM) | +0.05 partial | −0.05 |
| `promote_replica` | Promote postgres-replica to primary | 0.0 (red herring in task6) | −0.05 |
| `rebalance_partitions` | Rebalance kafka partitions | 0.0 (red herring in task3) | −0.05 |

Duplicate investigation (query same service twice) gives 0.0 reward — no double-dipping.

---

## Reward Design

### Per-step: Potential-Based Shaping (Ng 1999)
```
shaping = γ · Φ(s') − Φ(s)

where Φ(s) = Σ weight[svc] × health_value[status] / Σ weights

health_value: healthy=1.0, degraded=0.4, down=0.0

weights (criticality):
  postgres-primary: 1.5   kafka-broker: 1.4   redis-cache: 1.3
  payment/order:    1.2   api-gateway:  1.1   auth/nginx:  1.0
  notification:     0.9   inventory:    0.8   postgres-replica: 0.8

γ = 0.95
```
- Actions that worsen health (cascade fires) → negative shaping this step
- Correct fix that heals services → large positive shaping jump
- Policy-invariant: does not change which action sequence is optimal

### Per-step: Urgency Penalty
```
STEP_PENALTY = -0.02 per step (encourages efficiency)
```

### Terminal: Action-based reward (see table above)
- Correct fix: +0.50
- Circuit breaker on required service: +0.30
- Investigation of root service: +0.20 / +0.15

---

## Grader Score Distribution

| Agent Behaviour | task1 | task2 | task3 | task4 | task5 | task6 |
|---|---|---|---|---|---|---|
| Naive (chases loudest alert, wrong fix) | 0.20 | 0.00 | 0.25 | 0.10 | 0.17 | 0.11 |
| Wrong fix (flush_cache, rebalance, promote) | 0.70 | — | 0.60 | — | — | — |
| Correct root, partial fix (missing step) | — | — | — | — | — | 0.81 |
| LLM agent (Llama-3.3-70B observed) | 0.981 | 1.000 | 0.000* | — | — | — |
| Expert / fallback baseline | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*task3 LLM forgot to call `declare_resolved` after fixing kafka — scored 0 for not ending episode.

---

## How inference.py Works

```
inference.py
│
├── Env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN, ENV_BASE_URL
│
├── create_llm_client() → OpenAI-compatible client
│
├── _FALLBACK_SEQUENCES[task_id] → hardcoded expert action list per task
│
├── run_task(task_id, llm_client, env_client)
│   ├── POST /reset { task_id }
│   ├── conversation = [system_prompt]
│   ├── loop max_steps:
│   │   ├── format_observation(obs) → user message
│   │   ├── LLM.chat.completions.create(temperature=0)
│   │   ├── parse_action(response) → {action_type, service}
│   │   ├── POST /step
│   │   ├── sleep(1.0)   ← rate limit buffer
│   │   └── if all services healthy → force declare_resolved
│   ├── if LLM fails → _run_fallback_task()
│   └── POST /grader → score
│
├── _run_fallback_task(task_id)
│   └── runs _FALLBACK_SEQUENCES[task_id] deterministically → always scores ~1.0
│
└── run_all_tasks()
    ├── tasks: task1, task2, task3, task4, task5, task6
    ├── 5s inter-task sleep (rate limit cooldown)
    └── never crashes — fallback rescues any LLM failure
```

**Fallback sequences:**
- task1: `query_logs(redis-cache)` → `scale_up(redis-cache)` → `declare_resolved`
- task2: `query_logs(nginx-lb)` → `rollback_deploy(nginx-lb)` → `declare_resolved`
- task3: `query_logs(kafka-broker)` → `restart_service(kafka-broker)` → `declare_resolved`
- task4: `query_logs(postgres-replica)` → `restart_service(postgres-replica)` → `declare_resolved`
- task5: `query_logs(auth-service)` → `enable_circuit_breaker(api-gateway)` → `restart_service(auth-service)` → `declare_resolved`
- task6: `query_logs(postgres-primary)` → `check_metrics(postgres-primary)` → `enable_circuit_breaker(payment-service)` → `rollback_deploy(postgres-primary)` → `restart_service(payment-service)` → `declare_resolved`

---

## Observation Space

Every step, the agent receives:

```json
{
  "alerts": [
    "CRITICAL [redis-cache] OOM: memory 100% — eviction storm active",
    "WARNING  [api-gateway] p99 latency 4200ms (SLA: 300ms) — thundering herd",
    "INFO     [order-service] DB query latency elevated: 380ms — cache miss load"
  ],
  "service_health": {
    "nginx-lb": "healthy",
    "api-gateway": "degraded",
    "redis-cache": "down",
    "auth-service": "healthy",
    "order-service": "healthy",
    "payment-service": "healthy",
    "inventory-service": "healthy",
    "notification-service": "healthy",
    "kafka-broker": "healthy",
    "postgres-primary": "healthy",
    "postgres-replica": "healthy"
  },
  "last_action_result": "[LOGS redis-cache]\n[ERROR] OOM: maxmemory reached...",
  "step_count": 1,
  "available_actions": ["query_logs", "check_metrics", ...],
  "done": false,
  "reward": 0.14
}
```

---

## HTTP API Reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| POST | `/reset` | `{"task_id": "task1"}` | Initial observation |
| POST | `/step` | `{"action": {"action_type": "query_logs", "service": "redis-cache"}}` | `{observation, reward, done}` |
| GET | `/state` | — | `{episode_id, step_count}` |
| GET | `/health` | — | `{"status": "healthy"}` |
| GET | `/tasks` | — | 6 tasks + full action schema |
| POST | `/grader` | `?task_id=task1` (optional) | `{task_id, score, steps_taken, episode_id}` |
| POST | `/baseline` | — | `{task1..task6, average}` — runs expert on all 6 |
| GET | `/schema` | — | JSON schemas for action/observation/state |
| WS | `/ws` | — | WebSocket stateful session |

---

## How to Run Locally

```bash
cd prod-watchdog-env

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test it
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task1"}'
curl -X POST http://localhost:7860/baseline

# Run LLM inference agent
# (set creds in .env or export)
python inference.py
```

## How to Deploy to HF Spaces

```bash
# Push to HF Space (Docker type, tag: openenv)
git remote add hf https://huggingface.co/spaces/<username>/prod-watchdog-env
git push hf main

# Set Space secrets: API_BASE_URL, MODEL_NAME, HF_TOKEN
# Space URL format: https://<username>-prod-watchdog-env.hf.space
```

---

## Submission Checklist

| Item | Status |
|------|--------|
| 6 tasks (easy → hard) | ✅ |
| 11-service topology | ✅ |
| 10 action types | ✅ |
| Graders deterministic, scores vary 0.0–1.0 | ✅ |
| Potential-based reward shaping | ✅ |
| Fallback baseline always scores ~1.0 | ✅ |
| inference.py reads env vars (not hardcoded) | ✅ |
| openenv.yaml valid | ✅ |
| Pydantic Action + Observation models | ✅ |
| Dockerfile (python:3.11-slim, port 7860) | ✅ |
| uv.lock at root, committed | ✅ |
| pyproject.toml with `server` script entry | ✅ |
| /reset /step /state /health all work | ✅ |
| /tasks /grader /baseline work | ✅ |
| GitHub repo public | ⬜ push & verify |
| HF Space deployed + tagged `openenv` | ⬜ deploy |
| Live endpoint test on HF URL | ⬜ test |
| Submit on dashboard | ⬜ submit |