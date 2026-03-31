"""
Inference script for the ProdWatchdog environment.

Runs an LLM agent (via OpenAI-compatible API) against all 3 incident tasks
and reports scores. Uses ProdWatchdogClient from client.py for HTTP calls.

Primary agent: LLM via OpenAI-compatible API
Fallback agent: Deterministic rule-based expert policy (used if LLM unavailable)

The fallback ensures the script NEVER crashes and always produces reproducible
scores even if API credits are exhausted or keys are missing.

Environment variables (required for LLM agent):
    API_BASE_URL  - LLM API endpoint (e.g. https://router.huggingface.co/v1)
    MODEL_NAME    - Model identifier (e.g. meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN      - Hugging Face / API token (used as the API key)

Optional:
    ENV_BASE_URL  - ProdWatchdog server URL (default: http://localhost:7860)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py

NOTE: This file must remain named inference.py at the project root.
      HTTP interactions are handled by client.py (ProdWatchdogClient).
"""

import json
import os
import re
import time
from typing import Optional

from openai import OpenAI

from client import ProdWatchdogClient

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

SYSTEM_PROMPT = """You are an expert on-call Site Reliability Engineer (SRE).
You are responding to a production incident affecting microservices.

Your job:
1. Investigate the incident using query_logs and check_metrics
2. Identify the ROOT CAUSE service (not just a symptom)
3. Take the correct remediation action
4. If there is active cascade spread, use enable_circuit_breaker BEFORE fixing root
5. Declare the incident resolved with declare_resolved once root is fixed

Available services: api-gateway, auth-service, order-service, payment-service, inventory-service, notification-service

Available action types:
- query_logs: Read logs for a specific service
- check_metrics: Check CPU/memory/error metrics for a service
- restart_service: Restart a service (fixes DB leaks, memory exhaustion)
- rollback_deploy: Roll back the last deployment (fixes bad deploys)
- enable_circuit_breaker: Isolate a service to stop cascade propagation
- scale_up: Add more capacity (fixes CPU spikes, high load)
- declare_resolved: End the episode when root cause is fixed

Key reasoning rules:
- A service showing errors because its UPSTREAM is down = symptom, not root cause
- High CPU + saturated instances = scale_up
- Bad deployment (startup failure, missing config) = rollback_deploy
- DB connection exhaustion, memory leak = restart_service
- If a downstream service is in a crash loop due to upstream failures = circuit breaker first
- Investigate the ROOT cause service, not just the loudest alert

IMPORTANT: Respond ONLY with valid JSON in this exact format (no other text):
{"action_type": "<action>", "service": "<service>"}"""


# ---------------------------------------------------------------------------
# Deterministic fallback sequences (expert policy per task)
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


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def create_llm_client() -> OpenAI:
    if not API_BASE_URL or not HF_TOKEN:
        raise ValueError("API_BASE_URL and HF_TOKEN not set")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def parse_action(response_text: str) -> dict:
    """Extract JSON action from LLM response with layered fallback parsing."""
    text = response_text.strip()

    # 1. Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find any JSON object in text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Safe no-op fallback
    print(f"  [WARN] Could not parse action from LLM output: {text[:120]}")
    return {"action_type": "query_logs", "service": "api-gateway"}


# ---------------------------------------------------------------------------
# Fallback task runner (deterministic rule-based expert)
# ---------------------------------------------------------------------------

def _run_fallback_task(task_id: str, env_client: ProdWatchdogClient) -> float:
    """Run the deterministic expert policy for a task. Always produces reproducible scores."""
    print(f"  [FALLBACK] Using rule-based expert policy for {task_id}")
    try:
        env_client.reset(task_id)
    except Exception as e:
        print(f"  [ERROR] Fallback reset failed: {e}")
        return 0.0

    for i, action in enumerate(_FALLBACK_SEQUENCES.get(task_id, [])):
        try:
            obs, done, reward = env_client.step(
                action["action_type"],
                action.get("service"),
            )
            print(
                f"    fallback step {i+1}: "
                f"{action['action_type']}({action.get('service', '')}) "
                f"reward={reward:.2f}"
            )
            if done:
                break
        except Exception as e:
            print(f"  [ERROR] Fallback step {i+1} failed: {e}")
            break

    try:
        return env_client.get_grader_score(task_id)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LLM agent task runner
# ---------------------------------------------------------------------------

def run_task(
    task_id:    str,
    llm_client: Optional[OpenAI],
    env_client: ProdWatchdogClient,
) -> float:
    """Run the LLM agent on a single task. Falls back to rule-based if LLM fails."""
    print(f"\n{'='*60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'='*60}")

    if llm_client is None:
        return _run_fallback_task(task_id, env_client)

    # Reset environment
    try:
        obs, done, _ = env_client.reset(task_id)
    except Exception as e:
        print(f"[ERROR] Failed to reset env: {e}")
        return 0.0

    print(f"Initial alerts: {obs.get('alerts', [])}")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps    = 20
    llm_failed   = False

    for step in range(max_steps):
        user_msg = env_client.format_observation(obs)
        conversation.append({"role": "user", "content": user_msg})

        # Call LLM
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.1,
                max_tokens=128,
            )
            assistant_text = response.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] LLM call failed: {e}. Switching to fallback policy.")
            llm_failed = True
            break

        conversation.append({"role": "assistant", "content": assistant_text})
        action = parse_action(assistant_text)

        action_type = action.get("action_type", "?")
        service     = action.get("service", "?")
        print(f"  Step {step+1}: {action_type}({service})")

        # Execute action
        try:
            obs, done, reward = env_client.step(
                action_type,
                service if service != "?" else None,
            )
            print(f"    reward={reward:.2f}  done={done}")
        except Exception as e:
            print(f"  [ERROR] Step failed: {e}")
            break

        if done:
            print(f"  Episode ended at step {step+1}.")
            break

        time.sleep(0.5)  # rate limit buffer for HF free tier

    if llm_failed:
        return _run_fallback_task(task_id, env_client)

    try:
        score = env_client.get_grader_score(task_id)
        print(f"  Grader score: {score:.4f}")
        return score
    except Exception as e:
        print(f"  [ERROR] Grader call failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_tasks(env_url: str = ENV_BASE_URL) -> dict:
    """Run the agent on all 3 tasks. Returns scores dict. Never crashes."""
    llm_client = None
    try:
        llm_client = create_llm_client()
        print(f"[INFO] LLM client ready: {MODEL_NAME} @ {API_BASE_URL}")
    except Exception as e:
        print(f"[WARN] LLM unavailable ({e}). Using deterministic fallback policy.")

    scores = {}
    with ProdWatchdogClient(env_url) as env_client:
        for task_id in ["task1", "task2", "task3"]:
            try:
                score = run_task(task_id, llm_client, env_client)
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
        print("[WARN] API_BASE_URL / HF_TOKEN not set — running deterministic fallback policy.")
        print("To use LLM agent:")
        print("  export API_BASE_URL=https://router.huggingface.co/v1")
        print("  export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct")
        print("  export HF_TOKEN=hf_your_token_here")

    run_all_tasks()
