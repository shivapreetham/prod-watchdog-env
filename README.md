---
title: prod-watchdog-env
emoji: 🚨
colorFrom: red
colorTo: orange
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

---

## Environment Description

### Motivation
Production incidents are one of the highest-stakes, time-critical tasks engineers perform.
This environment tests whether an agent can reason under uncertainty, resist noisy red-herring
signals, follow causal chains through service dependencies, and take targeted corrective actions —
all skills that matter enormously for real-world AI reliability tooling.

### Service Dependency Graph
```
api-gateway ──→ auth-service
             └→ order-service ──→ payment-service ──→ notification-service
                              └→ inventory-service
```

Failures cascade downstream: if `order-service` is down, `payment-service` and `inventory-service`
will eventually degrade unless the root cause is fixed or circuit breakers are applied.

---

## Action Space

| `action_type`          | `service` required | Description                                      |
|------------------------|--------------------|--------------------------------------------------|
| `query_logs`           | yes                | Read recent logs for a service                   |
| `check_metrics`        | yes                | Check CPU/memory/error_rate/latency metrics      |
| `restart_service`      | yes                | Restart a service (fixes DB leaks, memory OOM)  |
| `rollback_deploy`      | yes                | Roll back last deployment (fixes bad deploys)    |
| `enable_circuit_breaker` | yes              | Isolate service to stop cascade propagation      |
| `scale_up`             | yes                | Add instances (fixes CPU spikes, high load)      |
| `declare_resolved`     | optional           | End episode and claim incident is resolved       |

**Action JSON format:**
```json
{"action_type": "query_logs", "service": "order-service"}
```

---

## Observation Space

| Field                | Type           | Description                                               |
|----------------------|----------------|-----------------------------------------------------------|
| `alerts`             | `List[str]`    | Active PagerDuty-style alert messages                     |
| `service_health`     | `Dict[str,str]`| Health of each service: `healthy` / `degraded` / `down`  |
| `last_action_result` | `str`          | Output of last action (logs, metrics, command result)     |
| `step_count`         | `int`          | Steps taken in current episode                            |
| `available_actions`  | `List[str]`    | Valid action types for this step                          |
| `done`               | `bool`         | Whether the episode has ended                             |
| `reward`             | `float`        | Per-step reward signal                                    |

---

## Tasks

### Task 1 — Easy: Single Service Outage
- **Scenario:** `order-service` is down due to a bad deployment (version 2.1.3)
- **Fix:** `rollback_deploy` on `order-service`
- **Max steps:** 10
- **Grader:** `investigated_correct_service(0.3) + correct_fix(0.3) + resolved(0.4)`

### Task 2 — Medium: Cascading Failure Diagnosis
- **Scenario:** `auth-service` CPU spike caused `api-gateway` degradation. Red herring: `notification-service` shows high latency (unrelated campaign traffic)
- **Fix:** `scale_up` on `auth-service`
- **Max steps:** 15
- **Grader:** `root_investigated(0.4) + correct_fix(0.3) + resolved(0.3)`

### Task 3 — Hard: Multi-Service Cascade with Noise
- **Scenario:** `payment-service` DB connection leak → `notification-service` down + `order-service` degraded. Two red-herring alerts: `auth-service` (OAuth provider issue) and `inventory-service` (scheduled analytics job). Agent must use circuit breaker to contain blast radius before fixing root.
- **Fix:** `enable_circuit_breaker(notification-service)` + `restart_service(payment-service)`
- **Max steps:** 20
- **Grader:** `root_investigated(0.35) + circuit_breaker_used(0.25) + correct_fix(0.25) + efficiency(0.15)`

---

## Reward Function

Per-step rewards guide the agent toward efficient, correct investigation:

| Action                            | Reward   |
|-----------------------------------|----------|
| Query logs/metrics on root service | +0.15–0.20 |
| Correct remediation action         | +0.50    |
| Correct circuit breaker             | +0.30    |
| Correct `declare_resolved`         | +0.50    |
| Query healthy service (wasted step)| −0.05    |
| Restart a healthy service          | −0.10    |
| Unknown/invalid action             | −0.10    |

Episode score (0.0–1.0) is computed by the task-specific grader after episode completion.

---

## API Endpoints

| Method | Path         | Description                                           |
|--------|--------------|-------------------------------------------------------|
| POST   | `/reset`     | Reset environment. Body: `{"task_id": "task1"}`      |
| POST   | `/step`      | Execute action. Body: `{"action": {...}}`             |
| GET    | `/state`     | Get current environment state                         |
| GET    | `/health`    | Health check → `{"status": "healthy"}`               |
| GET    | `/schema`    | Action/observation/state JSON schemas                 |
| POST   | `/mcp`       | MCP JSON-RPC endpoint                                 |
| WS     | `/ws`        | WebSocket for stateful sessions                       |
| GET    | `/tasks`     | List tasks with descriptions and action schema        |
| POST   | `/grader`    | Score completed episode. Param: `?task_id=task1`     |
| POST   | `/baseline`  | Run expert baseline on all tasks, return scores       |

---

## Baseline Scores (Expert Policy)

The expert policy knows the correct investigation and fix sequence:

| Task   | Score  | Steps |
|--------|--------|-------|
| task1  | 1.0000 | 5     |
| task2  | 1.0000 | 5     |
| task3  | 1.0000 | 6     |

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

# Build (from project root, one level up)
docker build -f server/Dockerfile -t prod-watchdog-env .

# Run
docker run -p 7860:7860 prod-watchdog-env
```

### Run Inference Agent

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here

# Make sure server is running, then:
python inference.py
```

### Quick Test

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset for task1
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Take an action
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "query_logs", "service": "order-service"}}'

# Get grader score
curl -X POST "http://localhost:7860/grader?task_id=task1"

# Run expert baseline
curl -X POST http://localhost:7860/baseline
```

### Validate with OpenEnv

```bash
# Local structure validation
openenv validate .

# Runtime validation (with server running)
openenv validate --url http://localhost:7860
```

---

## Project Structure

```
prod-watchdog-env/
├── models.py            # Pydantic Action + Observation types
├── inference.py         # LLM agent using OpenAI client
├── openenv.yaml         # OpenEnv manifest
├── pyproject.toml       # Package definition
├── README.md
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI app + custom endpoints
    ├── environment.py   # Core env logic + graders + scenarios
    ├── requirements.txt
    └── Dockerfile
```

---

## Team

**Team:** labwale
Built for the Meta × Hugging Face OpenEnv Hackathon, Round 1 (March–April 2026).
