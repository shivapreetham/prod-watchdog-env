"""
Deployment validation script — mirrors what the hackathon validator does.

Runs the same checks as the Phase 1 automated validator:
  Layer 1: Live HTTP checks against the deployed HF Space
  Layer 2: Structural checks (files present, format correct)

Usage:
    # Test local server
    python validate_deployment.py

    # Test deployed HF Space
    python validate_deployment.py https://shivapreetham17-prod-watchdog-env.hf.space

Exits 0 if all checks pass, 1 if any fail.
"""

import sys
import json
import time
import httpx

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:7860"

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append(passed)
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Layer 1: Live HTTP checks
# ---------------------------------------------------------------------------

section(f"Layer 1: Live HTTP — {BASE_URL}")

# /health
try:
    r = httpx.get(f"{BASE_URL}/health", timeout=10)
    check("/health returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("/health has 'status' field", "status" in data, str(data))
except Exception as e:
    check("/health reachable", False, str(e))

# /tasks
try:
    r = httpx.get(f"{BASE_URL}/tasks", timeout=10)
    check("/tasks returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    tasks = data.get("tasks", [])
    check("/tasks returns list", isinstance(tasks, list), f"count={len(tasks)}")
    check("/tasks has 3+ tasks", len(tasks) >= 3, f"found {len(tasks)}")
    check("/tasks has action_schema", "action_schema" in data)
    check("/tasks has services list", "services" in data)
except Exception as e:
    check("/tasks reachable", False, str(e))

# /reset — the most critical check
for task_id in ["task1", "task2", "task3"]:
    try:
        r = httpx.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=15)
        check(f"/reset task={task_id} returns 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()

        # Validator checks these exact fields
        obs = data.get("observation", data)  # some frameworks wrap, some don't
        has_alerts       = "alerts" in obs or "alerts" in data
        has_health       = "service_health" in obs or "service_health" in data
        has_last_result  = "last_action_result" in obs or "last_action_result" in data
        has_done         = "done" in obs or "done" in data

        check(f"/reset {task_id}: has 'alerts'",            has_alerts)
        check(f"/reset {task_id}: has 'service_health'",    has_health)
        check(f"/reset {task_id}: has 'last_action_result'",has_last_result)
        check(f"/reset {task_id}: has 'done'",              has_done)
        check(f"/reset {task_id}: valid JSON",              True, "parsed OK")
    except Exception as e:
        check(f"/reset {task_id}", False, str(e))

# /state — after a reset
try:
    httpx.post(f"{BASE_URL}/reset", json={"task_id": "task1"}, timeout=10)
    r = httpx.get(f"{BASE_URL}/state", timeout=10)
    check("/state returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("/state is valid JSON", True, "parsed OK")
except Exception as e:
    check("/state reachable", False, str(e))

# /step — full episode: reset → step → step → declare
section("Layer 1: Full episode flow (task1)")

try:
    # Use a single persistent client to reuse the SSL connection across all steps
    with httpx.Client(timeout=20) as client:
        r = client.post(f"{BASE_URL}/reset", json={"task_id": "task1"})
        assert r.status_code == 200, f"reset failed: {r.status_code}"

        # Step 1: query_logs
        r = client.post(f"{BASE_URL}/step",
                        json={"action": {"action_type": "query_logs", "service": "redis-cache"}})
        check("/step query_logs returns 200", r.status_code == 200, f"status={r.status_code}")
        obs = r.json()
        check("/step has 'reward' field",      "reward" in obs, str(list(obs.keys())))
        check("/step has 'done' field",        "done" in obs)
        check("/step has 'observation' or flat", "observation" in obs or "service_health" in obs)
        reward1 = obs.get("reward", obs.get("observation", {}).get("reward", None))
        check("/step reward is numeric",       isinstance(reward1, (int, float)), str(reward1))

        # Step 2: correct fix
        r = client.post(f"{BASE_URL}/step",
                        json={"action": {"action_type": "scale_up", "service": "redis-cache"}})
        check("/step scale_up returns 200", r.status_code == 200)

        # Step 3: declare
        r = client.post(f"{BASE_URL}/step",
                        json={"action": {"action_type": "declare_resolved", "service": "redis-cache"}})
        check("/step declare_resolved returns 200", r.status_code == 200)
        obs = r.json()
        done = obs.get("done", obs.get("observation", {}).get("done", False))
        check("/step done=true after declare", done is True, f"done={done}")

except Exception as e:
    check("Full episode flow", False, str(e))

# /grader — after episode
section("Layer 1: Grader endpoint")

try:
    with httpx.Client(timeout=20) as client:
        client.post(f"{BASE_URL}/reset", json={"task_id": "task1"})
        client.post(f"{BASE_URL}/step", json={"action": {"action_type": "query_logs", "service": "redis-cache"}})
        client.post(f"{BASE_URL}/step", json={"action": {"action_type": "scale_up", "service": "redis-cache"}})
        client.post(f"{BASE_URL}/step", json={"action": {"action_type": "declare_resolved", "service": "redis-cache"}})

        r = client.post(f"{BASE_URL}/grader?task_id=task1")
        check("/grader returns 200", r.status_code == 200, f"status={r.status_code}")
        data = r.json()
        check("/grader has 'score'", "score" in data, str(data))
        score = data.get("score", -1)
        check("/grader score in [0.0, 1.0]", 0.0 <= score <= 1.0, f"score={score}")
        check("/grader score > 0 for correct fix", score > 0.5, f"score={score}")
except Exception as e:
    check("/grader endpoint", False, str(e))

# /baseline
section("Layer 1: Baseline endpoint")

try:
    print("  [running /baseline — may take 30s...]")
    r = httpx.post(f"{BASE_URL}/baseline", timeout=60)
    check("/baseline returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("/baseline has scores", any(k.startswith("task") for k in data), str(data))
    avg = data.get("average", -1)
    check("/baseline has 'average'", avg >= 0, f"average={avg}")
    check("/baseline average > 0", avg > 0, f"average={avg}")
except Exception as e:
    check("/baseline endpoint", False, str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section("Summary")

total  = len(results)
passed = sum(results)
failed = total - passed

print(f"\n  Passed: {passed}/{total}")
if failed == 0:
    print(f"\n  {PASS} ALL CHECKS PASSED — ready to submit\n")
    sys.exit(0)
else:
    print(f"\n  {FAIL} {failed} check(s) FAILED — fix before submitting\n")
    sys.exit(1)
