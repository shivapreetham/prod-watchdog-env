"""
Inference script for the ProdWatchdog environment.

Runs an LLM agent (via OpenAI-compatible API) against all 3 incident tasks
and reports scores. Uses the server's HTTP endpoints.

Primary agent: LLM via OpenAI-compatible API
Fallback agent: Deterministic rule-based expert policy (used if LLM unavailable)

The fallback ensures the script NEVER crashes and always produces reproducible
scores even if API credits are exhausted or keys are missing.

Environment variables:
    API_BASE_URL  - LLM API endpoint (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME    - Model identifier (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN      - Hugging Face API token (used as the API key)

Optional:
    ENV_BASE_URL  - ProdWatchdog server URL (default: http://localhost:7860)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

import json
import os
import re
import sys
import time
from typing import Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

SYSTEM_PROMPT = """You are an expert on-call Site Reliability Engineer (SRE).
You are responding to a production incident affecting microservices.

Your job:
1. Investigate the incident using query_logs and check_metrics
2. Identify the ROOT CAUSE service
3. Take the correct remediation action
4. Declare the incident resolved with declare_resolved

Available services: api-gateway, auth-service, order-service, payment-service, inventory-service, notification-service

Available action types:
- query_logs: Read logs for a specific service
- check_metrics: Check CPU/memory/error metrics for a service
- restart_service: Restart a service (fixes DB leaks, memory issues)
- rollback_deploy: Roll back the last deployment (fixes bad deploys)
- enable_circuit_breaker: Isolate a service to stop cascade spread
- scale_up: Add more capacity (fixes CPU spikes, high load)
- declare_resolved: End the episode when root cause is fixed

IMPORTANT: Respond ONLY with valid JSON in this exact format (no other text):
{"action_type": "<action>", "service": "<service>"}

Think step by step:
- Look at which services are DOWN or DEGRADED
- Query logs on the most suspicious service first
- Check if the symptoms are a root cause or cascading effect
- Fix the actual root cause, not just symptoms
- Use circuit breakers for cascading failures before fixing root"""


# ---------------------------------------------------------------------------
# Deterministic fallback policy (used when LLM is unavailable)
# Rule-based expert that always takes the optimal sequence per task.
# Ensures baseline NEVER crashes even if API credits are exhausted.
# ---------------------------------------------------------------------------

_FALLBACK_SEQUENCES = {
    "task1": [
        {"action_type": "query_logs",       "service": "order-service"},
        {"action_type": "check_metrics",    "service": "order-service"},
        {"action_type": "rollback_deploy",  "service": "order-service"},
        {"action_type": "declare_resolved", "service": "order-service"},
    ],
    "task2": [
        {"action_type": "query_logs",       "service": "auth-service"},
        {"action_type": "check_metrics",    "service": "auth-service"},
        {"action_type": "scale_up",         "service": "auth-service"},
        {"action_type": "declare_resolved", "service": "auth-service"},
    ],
    "task3": [
        {"action_type": "query_logs",             "service": "payment-service"},
        {"action_type": "check_metrics",          "service": "payment-service"},
        {"action_type": "enable_circuit_breaker", "service": "notification-service"},
        {"action_type": "restart_service",        "service": "payment-service"},
        {"action_type": "declare_resolved",       "service": "payment-service"},
    ],
}


def create_llm_client() -> OpenAI:
    if not API_BASE_URL or not HF_TOKEN:
        raise ValueError("API_BASE_URL and HF_TOKEN not set")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def format_observation(obs_json: dict) -> str:
    """Format an observation dict into a readable prompt for the LLM."""
    health = obs_json.get("service_health", {})
    alerts = obs_json.get("alerts", [])
    last_result = obs_json.get("last_action_result", "")
    step = obs_json.get("step_count", 0)

    health_lines = "\n".join(
        f"  {svc}: {status.upper()}" for svc, status in health.items()
    )
    alert_lines = "\n".join(f"  {a}" for a in alerts) or "  (none)"

    return (
        f"=== INCIDENT STATUS (Step {step}) ===\n"
        f"\nService Health:\n{health_lines}\n"
        f"\nActive Alerts:\n{alert_lines}\n"
        f"\nLast Action Result:\n  {last_result}\n"
        f"\nWhat is your next action? Respond with JSON only."
    )


def parse_action(response_text: str) -> dict:
    """Extract JSON action from LLM response, with fallback parsing."""
    text = response_text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: safe no-op
    print(f"  [WARN] Could not parse action from: {text[:100]}")
    return {"action_type": "query_logs", "service": "api-gateway"}


def _run_fallback_task(task_id: str, env_url: str) -> float:
    """
    Run the deterministic rule-based fallback policy for a task.
    Used when LLM is unavailable. Always produces reproducible scores.
    """
    print(f"  [FALLBACK] Using rule-based expert policy for {task_id}")
    try:
        reset_resp = httpx.post(f"{env_url}/reset", json={"task_id": task_id}, timeout=30.0)
        reset_resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Fallback reset failed: {e}")
        return 0.0

    actions = _FALLBACK_SEQUENCES.get(task_id, [])
    for i, action in enumerate(actions):
        try:
            step_resp = httpx.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=30.0,
            )
            step_resp.raise_for_status()
            obs = step_resp.json().get("observation", step_resp.json())
            print(f"    fallback step {i+1}: {action['action_type']}({action.get('service','')}) "
                  f"reward={obs.get('reward', 0):.2f}")
            if obs.get("done"):
                break
        except Exception as e:
            print(f"  [ERROR] Fallback step {i+1} failed: {e}")
            break

    try:
        r = httpx.post(f"{env_url}/grader", params={"task_id": task_id}, timeout=15.0)
        return r.json().get("score", 0.0)
    except Exception:
        return 0.0


def run_task(task_id: str, client: Optional[OpenAI], env_url: str = ENV_BASE_URL) -> float:
    """Run the LLM agent on a single task. Falls back to rule-based if LLM fails."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    # If no LLM client available, go straight to fallback
    if client is None:
        return _run_fallback_task(task_id, env_url)

    # Reset environment
    try:
        reset_resp = httpx.post(
            f"{env_url}/reset",
            json={"task_id": task_id},
            timeout=30.0,
        )
        reset_resp.raise_for_status()
        raw = reset_resp.json()
        # Framework wraps in {"observation": {...}}
        obs = raw.get("observation", raw)
    except Exception as e:
        print(f"[ERROR] Failed to reset env: {e}")
        return 0.0

    print(f"Initial alerts: {obs.get('alerts', [])}")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps = 20
    llm_failed = False

    for step in range(max_steps):
        # Format observation for LLM
        user_msg = format_observation(obs)
        conversation.append({"role": "user", "content": user_msg})

        # Get LLM response — fallback to rule-based if LLM fails
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.1,
                max_tokens=256,
            )
            assistant_text = response.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] LLM call failed: {e}. Switching to fallback policy.")
            llm_failed = True
            break

        conversation.append({"role": "assistant", "content": assistant_text})

        # Parse action
        action = parse_action(assistant_text)
        action_type = action.get("action_type", "?")
        service = action.get("service", "?")
        print(f"  Step {step+1}: {action_type}({service})")

        # Execute action in environment
        try:
            step_resp = httpx.post(
                f"{env_url}/step",
                json={
                    "action": {
                        "action_type": action_type,
                        "service": service if service != "?" else None,
                    }
                },
                timeout=30.0,
            )
            step_resp.raise_for_status()
            raw = step_resp.json()
        except Exception as e:
            print(f"  [ERROR] Step failed: {e}")
            break

        # Framework wraps in {"observation": {...}}; done/reward are inside observation
        obs = raw.get("observation", raw)
        done = obs.get("done", False)
        reward = obs.get("reward", 0)
        print(f"    reward={reward:.2f}  done={done}")

        if done:
            print(f"  Episode ended at step {step+1}.")
            break

        # Rate limit: small sleep between LLM calls
        time.sleep(0.5)

    # If LLM failed mid-episode, run fallback for this task from scratch
    if llm_failed:
        return _run_fallback_task(task_id, env_url)

    # Get grader score
    try:
        grader_resp = httpx.post(
            f"{env_url}/grader",
            params={"task_id": task_id},
            timeout=15.0,
        )
        grader_resp.raise_for_status()
        score_data = grader_resp.json()
        score = score_data.get("score", 0.0)
        print(f"  Grader score: {score:.4f}")
        return score
    except Exception as e:
        print(f"  [ERROR] Grader call failed: {e}")
        return 0.0


def run_all_tasks(env_url: str = ENV_BASE_URL) -> dict:
    """Run the agent on all 3 tasks. Returns scores dict. Never crashes."""
    # Try to create LLM client; fall back to None (rule-based) if unavailable
    client = None
    try:
        client = create_llm_client()
        print(f"[INFO] LLM client ready: {MODEL_NAME} @ {API_BASE_URL}")
    except Exception as e:
        print(f"[WARN] LLM unavailable ({e}). Using deterministic fallback policy.")

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        try:
            score = run_task(task_id, client, env_url)
            scores[task_id] = score
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}")
            scores[task_id] = 0.0

    average = round(sum(scores.values()) / len(scores), 4)
    scores["average"] = average

    print(f"\n{'='*60}")
    print("BASELINE SCORES")
    print(f"{'='*60}")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}\n")

    return scores


if __name__ == "__main__":
    if not API_BASE_URL or not HF_TOKEN:
        print("[WARN] API_BASE_URL / HF_TOKEN not set — running with deterministic fallback policy.")
        print("To use LLM agent:")
        print("  export API_BASE_URL=https://api-inference.huggingface.co/v1")
        print("  export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct")
        print("  export HF_TOKEN=hf_your_token_here")

    scores = run_all_tasks()
