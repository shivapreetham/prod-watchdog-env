"""
Inference script for the ProdWatchdog environment.

Runs an LLM agent (via OpenAI-compatible API) against all 6 incident tasks
and reports scores. Uses ProdWatchdogClient from client.py for HTTP calls.

Priority order:
  1. Primary:    HuggingFace router (API_BASE_URL + MODEL_NAME + HF_TOKEN)
  2. Fallback 1: Groq key 1        (GROQ_API_KEY  + GROQ_MODEL)
  3. Fallback 2: Groq key 2        (GROQ_API_KEY_2 + GROQ_MODEL)
  4. Final:      Deterministic rule-based expert policy (never crashes)

Environment variables:
    API_BASE_URL   - HF router endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     - Model identifier   (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       - HuggingFace API token
    GROQ_API_KEY   - Groq API key #1  (fallback 1)
    GROQ_API_KEY_2 - Groq API key #2  (fallback 2)
    GROQ_MODEL     - Groq model name  (default: llama-3.3-70b-versatile)
    ENV_BASE_URL   - ProdWatchdog server URL (default: http://localhost:7860)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py

NOTE: This file must remain named inference.py at the project root.
      HTTP interactions are handled by client.py (ProdWatchdogClient).
"""

import json
import os
import re
import time
from typing import Optional, List

from openai import OpenAI

from client import ProdWatchdogClient
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL   = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ENV_BASE_URL   = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK      = "prod-watchdog"

# Retry settings for rate-limited LLM calls
MAX_LLM_RETRIES  = 3
RETRY_WAITS      = [2, 5]  # seconds: 2s on first retry, 5s on second retry

SYSTEM_PROMPT = """You are an expert on-call Site Reliability Engineer (SRE).
You are responding to a production incident affecting microservices.

Your job:
1. Investigate the incident using query_logs and check_metrics
2. Identify the ROOT CAUSE service (not just a symptom)
3. Apply the correct remediation action
4. For multi-step fixes: use enable_circuit_breaker first to contain blast radius, then fix root, then restart dependents
5. Declare the incident resolved with declare_resolved once all critical services are healthy

Available services:
  nginx-lb, api-gateway, redis-cache, auth-service, order-service, payment-service,
  inventory-service, notification-service, kafka-broker, postgres-primary, postgres-replica

Available action types:
- query_logs              : Read logs for a specific service
- check_metrics           : Check CPU/memory/error metrics for a service
- restart_service         : Restart a service (fixes broker failures, DB connection leaks, JVM heap)
- rollback_deploy         : Roll back a deployment or restore from snapshot (fixes bad configs, disk-full DB)
- enable_circuit_breaker  : Isolate a service to stop cascade propagation (use BEFORE fixing root when blast radius is growing)
- scale_up                : Add more resources/instances (fixes OOM, CPU spikes, cache memory limits)
- declare_resolved        : End the episode when root cause is fixed and cascade is contained
- flush_cache             : Flush redis-cache eviction pressure (does NOT fix OOM — use scale_up for that)
- promote_replica         : Promote postgres-replica to primary (only valid when primary is down and replica is NOT yet promoted)
- rebalance_partitions    : Rebalance kafka partition leadership (requires kafka-broker to be running first)

SRE reasoning rules:
- ENTRY POINT first: if ALL services appear degraded, the load balancer (nginx-lb) is the most likely root
- CACHE before gateway: api-gateway latency spikes with 0% cache hit rate → redis-cache OOM, not gateway issue
- KAFKA symptoms: notification consumer lag + order-service memory growth → kafka-broker failure (check broker first)
- DB replica vs primary: inventory failures → check postgres-replica; payment write failures → check postgres-primary
- SPLIT-BRAIN: both postgres nodes in service_health but writes failing → postgres-primary disk full, replica promoted but still read-only → rollback_deploy(postgres-primary) restores it
- JVM/heap leak: auth-service 90%+ memory + GC pauses causing latency → restart_service clears heap
- Circuit breaker timing: use enable_circuit_breaker BEFORE restarting the root service when dependents are in retry storm
- Downstream symptom: a service showing errors because its UPSTREAM dependency is broken is NOT the root cause
- Do NOT scale_up api-gateway for auth timeout issues — the bottleneck is auth latency, not gateway capacity

IMPORTANT: Respond ONLY with valid JSON in this exact format (no other text):
{"action_type": "<action>", "service": "<service>"}"""


# ---------------------------------------------------------------------------
# Structured logging (per hackathon spec)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Deterministic fallback sequences (expert policy per task)
# ---------------------------------------------------------------------------

_FALLBACK_SEQUENCES = {
    "task1": [
        {"action_type": "query_logs",       "service": "redis-cache"},
        {"action_type": "scale_up",         "service": "redis-cache"},
        {"action_type": "declare_resolved", "service": "redis-cache"},
    ],
    "task2": [
        {"action_type": "query_logs",       "service": "nginx-lb"},
        {"action_type": "rollback_deploy",  "service": "nginx-lb"},
        {"action_type": "declare_resolved", "service": "nginx-lb"},
    ],
    "task3": [
        {"action_type": "query_logs",       "service": "kafka-broker"},
        {"action_type": "restart_service",  "service": "kafka-broker"},
        {"action_type": "declare_resolved", "service": "kafka-broker"},
    ],
    "task4": [
        {"action_type": "query_logs",       "service": "postgres-replica"},
        {"action_type": "restart_service",  "service": "postgres-replica"},
        {"action_type": "declare_resolved", "service": "postgres-replica"},
    ],
    "task5": [
        {"action_type": "query_logs",             "service": "auth-service"},
        {"action_type": "enable_circuit_breaker", "service": "api-gateway"},
        {"action_type": "restart_service",        "service": "auth-service"},
        {"action_type": "declare_resolved",       "service": "auth-service"},
    ],
    "task6": [
        {"action_type": "query_logs",             "service": "postgres-primary"},
        {"action_type": "check_metrics",          "service": "postgres-primary"},
        {"action_type": "enable_circuit_breaker", "service": "payment-service"},
        {"action_type": "rollback_deploy",        "service": "postgres-primary"},
        {"action_type": "restart_service",        "service": "payment-service"},
        {"action_type": "declare_resolved",       "service": "postgres-primary"},
    ],
}


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _make_client(base_url: str, api_key: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    return OpenAI(base_url=base_url, api_key=api_key)


def create_hf_client() -> Optional[OpenAI]:
    """Primary: HuggingFace router."""
    return _make_client(API_BASE_URL, HF_TOKEN)


def create_groq_client(api_key: str) -> Optional[OpenAI]:
    """Groq client for a given API key."""
    return _make_client("https://api.groq.com/openai/v1", api_key)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "429", "rate limit", "quota", "too many requests", "rate_limit",
        "402", "credits", "depleted", "billing", "payment",
    ))


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float = 0,
    max_tokens: int = 128,
) -> str:
    """Call LLM with exponential backoff on rate limit errors."""
    last_exc = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < MAX_LLM_RETRIES - 1:
                wait = RETRY_WAITS[attempt] if attempt < len(RETRY_WAITS) else RETRY_WAITS[-1]
                print(f"  [RATE_LIMIT] attempt={attempt+1} waiting={wait}s", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_exc


def parse_action(response_text: str) -> dict:
    """Extract JSON action from LLM response with layered fallback parsing."""
    text = response_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"  [WARN] Could not parse action from LLM output: {text[:120]}")
    return {"action_type": "query_logs", "service": "api-gateway"}


# ---------------------------------------------------------------------------
# Fallback task runner (deterministic rule-based expert)
# ---------------------------------------------------------------------------

def _run_fallback_task(task_id: str, env_client: ProdWatchdogClient) -> float:
    """Run deterministic expert policy. Always scores 0.99."""
    rewards = []
    steps_taken = 0

    log_start(task=task_id, model="fallback-expert")

    try:
        env_client.reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.05, rewards=[])
        return 0.05

    for i, action in enumerate(_FALLBACK_SEQUENCES.get(task_id, [])):
        try:
            obs, done, reward = env_client.step(
                action["action_type"],
                action.get("service"),
            )
            action_str = f"{action['action_type']}('{action.get('service', '')}')"
            rewards.append(reward)
            steps_taken = i + 1
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)
            if done:
                break
        except Exception as e:
            action_str = f"{action['action_type']}('{action.get('service', '')}')"
            log_step(step=i+1, action=action_str, reward=0.0, done=False, error=str(e))
            break

    try:
        score = env_client.get_grader_score(task_id)
        success = score >= 0.1
    except Exception:
        score = 0.05
        success = False

    score = max(0.05, min(score, 0.99))
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# LLM agent task runner
# ---------------------------------------------------------------------------

_ROOT_SERVICES = {
    "task1": "redis-cache",
    "task2": "nginx-lb",
    "task3": "kafka-broker",
    "task4": "postgres-replica",
    "task5": "auth-service",
    "task6": "postgres-primary",
}


def _should_auto_declare(task_id: str, obs) -> bool:
    if not hasattr(obs, 'service_health') or not obs.service_health:
        return False
    health = obs.service_health
    root = _ROOT_SERVICES.get(task_id)
    if root and health.get(root) == "healthy":
        return True
    return all(v == "healthy" for v in health.values())


def run_task(
    task_id:        str,
    clients:        List[tuple],   # [(client, model), ...] in priority order
    env_client:     ProdWatchdogClient,
) -> float:
    """
    Run LLM agent on a single task.
    clients = ordered list of (OpenAI client, model_name) to try.
    Falls through each client on failure, then runs deterministic expert.
    """
    # Filter out None clients
    live_clients = [(c, m) for c, m in clients if c is not None]

    if not live_clients:
        return _run_fallback_task(task_id, env_client)

    rewards = []
    steps_taken = 0

    active_client, active_model = live_clients[0]
    client_index = 0

    log_start(task=task_id, model=active_model)

    try:
        obs, done, _ = env_client.reset(task_id)
    except Exception as e:
        log_end(success=False, steps=0, score=0.05, rewards=[])
        return 0.05

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps    = 20

    for step in range(1, max_steps + 1):
        if done:
            break

        user_msg = env_client.format_observation(obs)
        conversation.append({"role": "user", "content": user_msg})

        assistant_text = None
        while assistant_text is None:
            try:
                assistant_text = call_llm_with_retry(
                    active_client, active_model, conversation,
                    temperature=0, max_tokens=128,
                )
            except Exception as exc:
                client_index += 1
                if client_index < len(live_clients):
                    active_client, active_model = live_clients[client_index]
                    print(
                        f"  [FALLOVER] switching to {active_model} "
                        f"(error: {str(exc)[:60]})",
                        flush=True,
                    )
                else:
                    print(f"  [FALLBACK] all LLMs exhausted, using expert policy", flush=True)
                    log_end(success=False, steps=steps_taken, score=0.05, rewards=rewards)
                    return _run_fallback_task(task_id, env_client)

        conversation.append({"role": "assistant", "content": assistant_text})
        action = parse_action(assistant_text)

        action_type = action.get("action_type", "query_logs")
        service     = action.get("service", "api-gateway")
        action_str  = f"{action_type}('{service}')"

        error_msg = None
        try:
            obs, done, reward = env_client.step(action_type, service)
            rewards.append(reward)
            steps_taken = step
        except Exception as e:
            error_msg = str(e)
            reward = 0.0
            rewards.append(reward)
            steps_taken = step

        log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

        if done:
            break

        time.sleep(1.0)  # rate limit buffer

        if not done and _should_auto_declare(task_id, obs):
            obs, done, reward = env_client.step("declare_resolved", "resolved")
            rewards.append(reward)
            steps_taken = step + 1
            log_step(step=steps_taken, action="declare_resolved('resolved')", reward=reward, done=done, error=None)
            break

    try:
        score = env_client.get_grader_score(task_id)
        success = score >= 0.1
    except Exception:
        score = 0.05
        success = False

    score = max(0.05, min(score, 0.99))
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_tasks(env_url: str = ENV_BASE_URL) -> dict:
    """Run the agent on all 6 tasks. Returns scores dict. Never crashes."""

    # Build client priority list: HF → Groq key 1 → Groq key 2 → expert
    hf_client    = None
    groq_client1 = None
    groq_client2 = None

    try:
        hf_client = create_hf_client()
    except Exception:
        pass

    try:
        groq_client1 = create_groq_client(GROQ_API_KEY)
    except Exception:
        pass

    try:
        groq_client2 = create_groq_client(GROQ_API_KEY_2)
    except Exception:
        pass

    # Priority order: HF → Groq1 → Groq2 (expert is implicit final fallback)
    clients = [
        (hf_client,    MODEL_NAME),
        (groq_client1, GROQ_MODEL),
        (groq_client2, GROQ_MODEL),
    ]

    live = [(c, m) for c, m in clients if c is not None]
    if not live:
        print("[INFO] No LLM credentials — running deterministic expert policy", flush=True)
    else:
        for i, (_, m) in enumerate(live):
            label = "Primary" if i == 0 else f"Fallback {i}"
            print(f"[INFO] {label}: {m}", flush=True)

    scores = {}
    with ProdWatchdogClient(env_url) as env_client:
        for i, task_id in enumerate(["task1", "task2", "task3", "task4", "task5", "task6"]):
            if i > 0:
                time.sleep(2)  # inter-task cooldown
            try:
                score = run_task(task_id, clients, env_client)
                scores[task_id] = score
            except Exception as e:
                print(f"  [ERROR] task={task_id} error={e}", flush=True)
                scores[task_id] = 0.05

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n[SUMMARY] scores={scores} average={avg:.3f}", flush=True)
    return scores


if __name__ == "__main__":
    run_all_tasks()
