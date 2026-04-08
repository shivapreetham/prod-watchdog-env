---
title: prod-watchdog-env
emoji: 🚨
colorFrom: red
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
short_description: Production incident response SRE environment for OpenEnv
---

# ProdWatchdog — Production Incident Response Environment

An OpenEnv environment that simulates real on-call SRE (Site Reliability Engineering) work.
An AI agent receives production alerts, investigates microservice failures using logs and metrics,
contains blast radius using circuit breakers, and remediates the root cause.

Built for the **Meta × Hugging Face OpenEnv Hackathon, Round 1 (2026)** by team **labwale**.

---

## Environment Description

### Motivation
Production incidents are one of the highest-stakes, time-critical tasks engineers perform.
This environment tests whether an agent can reason under uncertainty, resist noisy red-herring
signals, follow causal chains through a realistic service dependency graph, and take targeted
corrective actions — all skills that matter enormously for real-world AI reliability tooling.

### Service Dependency Graph (11 services)

```
nginx-lb ──────────────────────────────────────→ api-gateway
redis-cache ────────────────────────────────────→ api-gateway
                                                        │
                          ┌─────────────────────────────┤
                          │                             │
                    auth-service                  order-service
                          │                        /         \
                   postgres-primary       payment-service   inventory-service
                                               │                  │
                                       notification-service  postgres-replica
                                               │
                                         kafka-broker
```

Failures cascade downstream: if `kafka-broker` goes down, `order-service` and
`notification-service` will eventually crash unless the root cause is fixed or circuit
breakers are applied. Agents must identify the root cause, not just treat symptoms.

---

## Action Space

| `action_type`              | `service` required | Description                                           |
|----------------------------|--------------------|-------------------------------------------------------|
| `query_logs`               | yes                | Read recent logs for a service                        |
| `check_metrics`            | yes                | Check CPU/memory/error_rate/latency metrics           |
| `restart_service`          | yes                | Restart a service (fixes DB leaks, memory OOM, JVM)  |
| `rollback_deploy`          | yes                | Roll back last deployment or restore from snapshot    |
| `enable_circuit_breaker`   | yes                | Isolate service to stop cascade propagation           |
| `scale_up`                 | yes                | Add instances/memory (fixes OOM, CPU spikes)          |
| `flush_cache`              | yes                | Flush cache contents (partial fix for cache issues)   |
| `promote_replica`          | yes                | Promote DB replica to primary                         |
| `rebalance_partitions`     | yes                | Rebalance Kafka partitions                            |
| `declare_resolved`         | optional           | End episode and claim incident is resolved            |

**Action JSON format:**
```json
{"action_type": "query_logs", "service": "kafka-broker"}
```

---

## Observation Space

| Field                | Type            | Description                                               |
|----------------------|-----------------|-----------------------------------------------------------|
| `alerts`             | `List[str]`     | Active PagerDuty-style alert messages                     |
| `service_health`     | `Dict[str, str]`| Health of each service: `healthy` / `degraded` / `down`  |
| `last_action_result` | `str`           | Output of last action (logs, metrics, command result)     |
| `step_count`         | `int`           | Steps taken in current episode                            |
| `available_actions`  | `List[str]`     | Valid action types for this step                          |
| `done`               | `bool`          | Whether the episode has ended                             |
| `reward`             | `float`         | Per-step reward signal                                    |

---

## Tasks

### Task 1 — Easy: Redis Cache OOM — Thundering Herd
- **Scenario:** `redis-cache` has hit memory OOM (100% utilization), causing 100% cache miss rate. Every request hits the DB directly. `api-gateway` shows high error rate — a downstream symptom. If not fixed by step 6, `nginx-lb` starts dropping connections.
- **Fix:** `scale_up(redis-cache)`
- **Max steps:** 12
- **Grader:** `diagnosed_redis(0.25) + first_inv_correct(0.15) + fix(0.25) + final_state(0.20) + efficiency(0.15)`

### Task 2 — Easy: nginx-lb Bad Config Deploy
- **Scenario:** `nginx-lb` received a bad config deploy setting `worker_connections` to 4 (was 4096). `nginx-lb` is dropping 99% of traffic. All backend services are actually healthy. Key insight: when ALL services appear degraded, check the entry point first.
- **Fix:** `rollback_deploy(nginx-lb)`
- **Max steps:** 10
- **Grader:** `diagnosed_lb(0.30) + first_inv_quality(0.20) + fix(0.30) + final_state(0.20)`

### Task 3 — Medium: Kafka Broker Partition Failure
- **Scenario:** `kafka-broker` partition leader election failed after a 47s JVM GC pause. `order-service` producer times out silently (orders queuing in memory). `notification-service` consumer group is stuck with 847k undelivered messages. Red herrings: notification consumer lag looks primary, order memory spike looks like a leak. Cascade: order crashes at step 8, notification crashes at step 12.
- **Fix:** `restart_service(kafka-broker)`
- **Max steps:** 15
- **Grader:** `diagnosed_kafka(0.20) + inv_speed(0.15) + fix(0.30) + final_state(0.20) + efficiency(0.15)`

### Task 4 — Medium: Postgres Replica WAL Corruption
- **Scenario:** `postgres-replica` has a corrupted WAL segment after an OS-level force stop. `inventory-service` reads are all failing. `order-service` cannot verify stock availability. Red herring: `order-service` error rate looks like the root.
- **Fix:** `restart_service(postgres-replica)`
- **Max steps:** 15
- **Grader:** `diagnosed_replica(0.25) + right_track_first(0.15) + fix(0.30) + final_state(0.20) + efficiency(0.10)`

### Task 5 — Medium-Hard: Auth Service JWT Memory Leak
- **Scenario:** `auth-service` JWT validation cache has a 7-day memory leak, causing GC pauses and latency spikes. `api-gateway`, `order-service`, and `payment-service` are all showing auth failures. Red herrings: `redis-cache` latency looks suspicious, `payment-service` error rate looks like the root.
- **Fix:** `enable_circuit_breaker(api-gateway)` → `restart_service(auth-service)`
- **Max steps:** 20
- **Grader:** `diagnosed_auth(0.25) + circuit_breaker_gw(0.20) + fix(0.30) + final_state(0.15) + efficiency(0.10)`

### Task 6 — Hard: Postgres Primary Disk Full + Split-Brain (3-step fix)
- **Scenario:** `postgres-primary` failed (disk 100% full). Auto-failover promoted `postgres-replica` but its `postgresql.conf` still has `read-only` mode (split-brain). `payment-service` write transactions failing with `ReadOnlyDatabaseException`. 3-step fix required in order. Red herrings: `auth` shows 23% errors (JWT key issue), `order` memory looks like a leak. Cascade: inventory degrades step 7, order crashes step 12, auth crashes step 16.
- **Fix (ordered):** `enable_circuit_breaker(payment-service)` → `rollback_deploy(postgres-primary)` → `restart_service(payment-service)`
- **Max steps:** 25
- **Grader:** `diagnosed_pg(0.20) + circuit_breaker_payment(0.20) + primary_fix(0.25) + secondary_fix(0.15) + final_state(0.10) + efficiency(0.10)`

---

## Reward Function

Rewards use **potential-based shaping** (Ng 1999) for dense signal every step:

```
step_reward = action_reward + γ × Φ(s') − Φ(s) − 0.02
```

where `Φ(s)` is a criticality-weighted average of service health scores.

| Event                                        | Reward     |
|----------------------------------------------|------------|
| Query logs on root cause service (first time) | +0.20      |
| Check metrics on root cause service           | +0.15      |
| Query/metrics on non-root service             | −0.05      |
| Repeat query (already seen)                   | 0.00       |
| Correct remediation action                    | +0.30–0.60 |
| Correct circuit breaker                       | +0.20–0.30 |
| Correct `declare_resolved`                    | +0.50      |
| Wrong action on healthy service               | −0.10      |
| Per-step penalty (efficiency)                 | −0.02      |

Episode score (0.0–1.0) is computed by the task-specific grader after episode completion.
Scores are capped at 0.99 for tasks that require complex multi-step reasoning.

---

## Baseline Scores (Expert Policy)

The expert policy investigates the root cause service, applies the correct fix sequence, and declares resolved:

| Task   | Difficulty    | Score  | Steps |
|--------|---------------|--------|-------|
| task1  | easy          | 0.9900 | 3     |
| task2  | easy          | 0.9900 | 3     |
| task3  | medium        | 0.9900 | 3     |
| task4  | medium        | 0.9900 | 3     |
| task5  | medium-hard   | 0.9900 | 4     |
| task6  | hard          | 0.9900 | 6     |

---

## API Endpoints

| Method | Path         | Description                                                        |
|--------|--------------|--------------------------------------------------------------------|
| POST   | `/reset`     | Reset environment. Body: `{"task_id": "task1"}`                   |
| POST   | `/step`      | Execute action. Body: `{"action": {"action_type": ..., "service": ...}}` |
| GET    | `/state`     | Current state: episode_id, step_count, observation, reward, done  |
| GET    | `/health`    | Health check → `{"status": "healthy"}`                            |
| GET    | `/schema`    | Action/observation/state JSON schemas                             |
| POST   | `/mcp`       | MCP JSON-RPC endpoint                                             |
| WS     | `/ws`        | WebSocket for stateful sessions                                   |
| GET    | `/tasks`     | List all 6 tasks with descriptions and action schema              |
| POST   | `/grader`    | Score completed episode. Param: `?task_id=task1`                  |
| POST   | `/baseline`  | Run expert baseline on all 6 tasks, return scores                 |

---

## Setup & Usage

### Local Development

```bash
git clone <repo-url>
cd prod-watchdog-env

# Install dependencies
pip install -r server/requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
cd prod-watchdog-env

# Build
docker build -t prod-watchdog-env .

# Run
docker run -p 7860:7860 prod-watchdog-env
```

### Run Inference Agent

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=<your_groq_api_key>

# Make sure server is running, then:
python inference.py
```

The inference script runs an LLM agent with two fallback layers:
1. Primary: LLM via `API_BASE_URL` (HF router or any OpenAI-compatible API)
2. Secondary: Groq (`GROQ_API_KEY` env var), if primary is rate-limited
3. Final fallback: Deterministic rule-based expert policy (never crashes, always scores)

### Quick Test

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset for task1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "query_logs", "service": "redis-cache"}}'

# Get grader score after episode
curl -X POST "http://localhost:7860/grader?task_id=task1"

# Run expert baseline on all 6 tasks
curl -X POST http://localhost:7860/baseline
```

---

## Project Structure

```
prod-watchdog-env/
├── inference.py         # LLM agent (primary + Groq fallback + rule-based fallback)
├── client.py            # HTTP client for environment API
├── models.py            # Pydantic Action + Observation types
├── openenv.yaml         # OpenEnv manifest (6 tasks, action/obs space)
├── pyproject.toml       # Package definition ([project.scripts] server = ...)
├── Dockerfile           # Single Dockerfile for HF Spaces deployment
├── uv.lock              # Locked dependencies
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI app + /tasks /grader /baseline endpoints
    ├── environment.py   # Core env logic, graders, scenarios, reward shaping
    └── requirements.txt
```

---

## Team

**Team:** labwale  
Built for the Meta × Hugging Face OpenEnv Hackathon, Round 1 (March–April 2026).
