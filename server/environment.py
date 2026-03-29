"""
ProdWatchdog Environment Implementation.

Simulates a production microservice incident response scenario.
The agent must diagnose cascading failures from noisy logs/metrics,
contain blast radius, and remediate the root cause.

Service Dependency Graph:
    api-gateway → auth-service
    api-gateway → order-service → payment-service → notification-service
                                → inventory-service
"""

import copy
from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ProdWatchdogAction, ProdWatchdogObservation, VALID_ACTION_TYPES
except ImportError:
    from models import ProdWatchdogAction, ProdWatchdogObservation, VALID_ACTION_TYPES


# ---------------------------------------------------------------------------
# Service graph
# ---------------------------------------------------------------------------

SERVICES = [
    "api-gateway",
    "auth-service",
    "order-service",
    "payment-service",
    "inventory-service",
    "notification-service",
]

# When service S heals, which services should also recover?
# (callers that were degraded due to S being down will heal when S heals)
HEAL_PROPAGATION = {
    "api-gateway": [],
    "auth-service": ["api-gateway"],
    "order-service": ["api-gateway"],
    "payment-service": ["order-service", "notification-service"],
    "inventory-service": ["order-service"],
    "notification-service": [],
}

# ---------------------------------------------------------------------------
# Fixed incident scenarios (deterministic per task_id)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "task1": {
        "description": "Single Service Outage — order-service has a bad deployment",
        "root_cause_service": "order-service",
        "root_cause_type": "bad_deploy",
        "fix_action": "rollback_deploy",
        "initial_health": {
            "api-gateway": "healthy",
            "auth-service": "healthy",
            "order-service": "down",
            "payment-service": "healthy",
            "inventory-service": "healthy",
            "notification-service": "healthy",
        },
        "initial_alerts": [
            "CRITICAL [order-service] Health check failed — 0/3 instances passing",
            "ERROR [order-service] 502 Bad Gateway rate: 87% over last 5 minutes",
        ],
        "logs": {
            "order-service": (
                "[ERROR] Application startup failed\n"
                "[ERROR] Config validation error: DATABASE_URL not found in env (version 2.1.3)\n"
                "[ERROR] Health check endpoint /health returning 500\n"
                "[INFO]  Deploy timestamp: 2026-03-29T14:22:00Z (version 2.1.3)\n"
                "[WARN]  Previous stable version: 2.1.2\n"
                "[ERROR] 502 responses spiking — downstream callers affected\n"
            ),
            "api-gateway": "[INFO] Upstream order-service returning errors. Retrying...\n",
            "auth-service": "[INFO] Normal traffic. 200 OK rate: 99.8%\n",
            "payment-service": "[INFO] Idle — no requests from order-service.\n",
            "inventory-service": "[INFO] Idle — no requests from order-service.\n",
            "notification-service": "[INFO] Queue empty. Standing by.\n",
        },
        "metrics": {
            "order-service": "cpu=2% | memory=18% | error_rate=87% | latency_p99=N/A (down)",
            "api-gateway": "cpu=12% | memory=45% | error_rate=8% | latency_p99=340ms",
            "auth-service": "cpu=14% | memory=40% | error_rate=0.1% | latency_p99=22ms",
            "payment-service": "cpu=1% | memory=10% | error_rate=0% | latency_p99=N/A",
            "inventory-service": "cpu=1% | memory=10% | error_rate=0% | latency_p99=N/A",
            "notification-service": "cpu=1% | memory=8% | error_rate=0% | latency_p99=N/A",
        },
        "max_steps": 10,
    },

    "task2": {
        "description": (
            "Cascading Failure — auth-service CPU spike degraded api-gateway. "
            "Red herring: notification-service shows elevated latency (unrelated)."
        ),
        "root_cause_service": "auth-service",
        "root_cause_type": "cpu_spike",
        "fix_action": "scale_up",
        "initial_health": {
            "api-gateway": "degraded",
            "auth-service": "down",
            "order-service": "healthy",
            "payment-service": "healthy",
            "inventory-service": "healthy",
            "notification-service": "healthy",
        },
        "initial_alerts": [
            "CRITICAL [auth-service] CPU at 99% — all 3 instances saturated",
            "WARNING  [api-gateway] Elevated 5xx rate: 23% — upstream auth unavailable",
            "WARNING  [notification-service] p99 latency 2.1s (SLA: 1.5s) — high send volume",
        ],
        "logs": {
            "auth-service": (
                "[ERROR] CPU throttling — request queue depth 4200 and growing\n"
                "[ERROR] JWT validation timeouts increasing: 3200ms avg (target <50ms)\n"
                "[WARN]  Token cache miss rate: 94% — cache overwhelmed\n"
                "[INFO]  Spike started 2026-03-29T13:55Z — coincides with promo campaign\n"
                "[ERROR] Instance auth-2 OOMKilled — JVM heap exhausted under CPU pressure\n"
            ),
            "api-gateway": (
                "[WARN] auth-service upstream returning 503 — circuit breaker threshold approaching\n"
                "[INFO] Retrying auth checks with 2s backoff\n"
            ),
            "notification-service": (
                "[INFO] Marketing campaign emails queued: 2.1M — processing normally\n"
                "[INFO] Send rate: 18k/min — expected high volume\n"
            ),
            "order-service": "[INFO] Normal operations. order volume nominal.\n",
            "payment-service": "[INFO] Processing normally.\n",
            "inventory-service": "[INFO] Stock sync running normally.\n",
        },
        "metrics": {
            "auth-service": "cpu=99% | memory=91% | error_rate=78% | latency_p99=3200ms",
            "api-gateway": "cpu=28% | memory=52% | error_rate=23% | latency_p99=890ms",
            "order-service": "cpu=18% | memory=44% | error_rate=0.2% | latency_p99=45ms",
            "payment-service": "cpu=12% | memory=38% | error_rate=0.1% | latency_p99=55ms",
            "inventory-service": "cpu=9% | memory=30% | error_rate=0% | latency_p99=30ms",
            "notification-service": "cpu=41% | memory=58% | error_rate=0.4% | latency_p99=2100ms",
        },
        "max_steps": 15,
    },

    "task3": {
        "description": (
            "Multi-Service Cascade with Noise — payment-service DB connection leak. "
            "Downstream notification-service is down. Noisy alerts with 2 red herrings: "
            "auth-service (unrelated error burst) and inventory-service (slow queries)."
        ),
        "root_cause_service": "payment-service",
        "root_cause_type": "db_leak",
        "fix_action": "restart_service",
        "requires_circuit_breaker": "notification-service",
        "initial_health": {
            "api-gateway": "healthy",
            "auth-service": "healthy",
            "order-service": "degraded",
            "payment-service": "degraded",
            "inventory-service": "healthy",
            "notification-service": "down",
        },
        "initial_alerts": [
            "CRITICAL [notification-service] Service unreachable — all health checks failing",
            "WARNING  [order-service] Checkout flow degraded — payment callbacks timing out",
            "WARNING  [auth-service] Elevated 401 error rate: 4.2% (baseline: 0.3%)",
            "WARNING  [inventory-service] Query latency elevated: p99=4.8s (baseline: 0.4s)",
            "INFO     [payment-service] DB connection pool at 97% utilization",
        ],
        "logs": {
            "notification-service": (
                "[ERROR] Cannot connect to payment-service webhook endpoint — connection refused\n"
                "[ERROR] Retry budget exhausted after 50 attempts\n"
                "[ERROR] Service entering crash loop\n"
            ),
            "order-service": (
                "[WARN] payment-service callback timeout after 30s — 34% of checkout requests affected\n"
                "[INFO] Falling back to async payment confirmation for affected orders\n"
            ),
            "payment-service": (
                "[WARN]  DB connection pool: 97/100 connections in use\n"
                "[ERROR] New DB connections being refused — pool exhausted\n"
                "[WARN]  Connection leak detected in txn_commit() — connections not released\n"
                "[INFO]  Uptime: 18d 4h — no restart since last deploy\n"
                "[ERROR] Outbound webhook delivery failing — worker threads blocked on DB\n"
            ),
            "auth-service": (
                "[WARN] Third-party OAuth provider intermittent 429s — rate limiting\n"
                "[INFO] Retry logic handling it — elevated 401s are provider-side\n"
            ),
            "inventory-service": (
                "[INFO] Analytics report job running — scheduled weekly heavy query\n"
                "[INFO] Query expected to finish in ~2h — normal slow period\n"
            ),
            "api-gateway": "[INFO] Routing normally. No upstream issues.\n",
        },
        "metrics": {
            "payment-service": "cpu=18% | memory=72% | error_rate=12% | db_connections=97/100",
            "notification-service": "cpu=0% | memory=0% | error_rate=100% | latency_p99=N/A (down)",
            "order-service": "cpu=22% | memory=48% | error_rate=3.4% | latency_p99=890ms",
            "auth-service": "cpu=16% | memory=42% | error_rate=4.2% | latency_p99=210ms",
            "inventory-service": "cpu=78% | memory=65% | error_rate=0% | latency_p99=4800ms",
            "api-gateway": "cpu=14% | memory=40% | error_rate=0.2% | latency_p99=120ms",
        },
        "max_steps": 20,
    },
}


# ---------------------------------------------------------------------------
# Module-level episode state (persists across stateless HTTP requests)
# ---------------------------------------------------------------------------

_EPISODE_STATE = {
    "task_id": "task1",
    "step_count": 0,
    "service_health": {},
    "episode_history": [],   # list of {"action": dict, "observation": dict}
    "circuit_breakers": [],
    "resolved": False,
    "resolution_claim": None,
    "episode_id": None,
}


def get_episode_state() -> dict:
    return _EPISODE_STATE


# ---------------------------------------------------------------------------
# Helper: compute active alerts based on current service health
# ---------------------------------------------------------------------------

def _compute_alerts(scenario: dict, health: dict) -> list:
    alerts = []
    for service, status in health.items():
        if status == "down":
            alerts.append(f"CRITICAL [{service}] Service is DOWN")
        elif status == "degraded":
            alerts.append(f"WARNING  [{service}] Service is DEGRADED")
    # Add scenario-specific contextual alerts
    for alert in scenario["initial_alerts"]:
        if alert not in alerts:
            alerts.append(alert)
    return list(dict.fromkeys(alerts))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class ProdWatchdogEnvironment(Environment):
    """
    Production Incident Response Environment.

    Models a real on-call SRE scenario with cascading microservice failures.
    The agent must diagnose root cause and remediate using available tools.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        task_id: Optional[str] = "task1",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ProdWatchdogObservation:
        """Reset the environment for a given task."""
        task_id = task_id or "task1"
        if task_id not in SCENARIOS:
            task_id = "task1"

        scenario = SCENARIOS[task_id]
        ep_id = episode_id or str(uuid4())

        # Reset global episode state
        _EPISODE_STATE["task_id"] = task_id
        _EPISODE_STATE["step_count"] = 0
        _EPISODE_STATE["service_health"] = copy.deepcopy(scenario["initial_health"])
        _EPISODE_STATE["episode_history"] = []
        _EPISODE_STATE["circuit_breakers"] = []
        _EPISODE_STATE["resolved"] = False
        _EPISODE_STATE["resolution_claim"] = None
        _EPISODE_STATE["episode_id"] = ep_id

        self._state = State(episode_id=ep_id, step_count=0)

        alerts = _compute_alerts(scenario, _EPISODE_STATE["service_health"])

        return ProdWatchdogObservation(
            alerts=alerts,
            service_health=copy.deepcopy(_EPISODE_STATE["service_health"]),
            last_action_result=(
                f"[INCIDENT STARTED] On-call paged. Scenario: {scenario['description']}. "
                f"Investigate and resolve."
            ),
            step_count=0,
            available_actions=VALID_ACTION_TYPES,
            done=False,
            reward=0.0,
        )

    def step(  # type: ignore[override]
        self,
        action: ProdWatchdogAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ProdWatchdogObservation:
        """Execute one action and return the resulting observation."""
        task_id = _EPISODE_STATE["task_id"]
        scenario = SCENARIOS[task_id]

        _EPISODE_STATE["step_count"] += 1
        step_count = _EPISODE_STATE["step_count"]
        max_steps = scenario["max_steps"]

        health = _EPISODE_STATE["service_health"]
        circuit_breakers = _EPISODE_STATE["circuit_breakers"]

        action_type = (action.action_type or "").strip().lower()
        service = (action.service or "").strip().lower()

        # --- Process action ---
        result, reward_val, done = _process_action(
            action_type, service, scenario, health, circuit_breakers, step_count
        )

        # Check if max steps reached
        if step_count >= max_steps and not done:
            done = True
            result += f"\n[TIMEOUT] Maximum steps ({max_steps}) reached. Episode ending."

        # Record history
        _EPISODE_STATE["episode_history"].append({
            "action": {
                "action_type": action_type,
                "service": service,
                "parameters": action.parameters,
            },
            "observation": {
                "step_count": step_count,
                "service_health": copy.deepcopy(health),
                "last_action_result": result,
            },
            "reward": reward_val,
        })

        if done:
            _EPISODE_STATE["resolved"] = True

        self._state = State(
            episode_id=_EPISODE_STATE["episode_id"],
            step_count=step_count,
        )

        alerts = _compute_alerts(scenario, health) if not done else []

        return ProdWatchdogObservation(
            alerts=alerts,
            service_health=copy.deepcopy(health),
            last_action_result=result,
            step_count=step_count,
            available_actions=VALID_ACTION_TYPES if not done else [],
            done=done,
            reward=reward_val,
        )

    @property
    def state(self) -> State:
        return State(
            episode_id=_EPISODE_STATE.get("episode_id"),
            step_count=_EPISODE_STATE.get("step_count", 0),
        )


# ---------------------------------------------------------------------------
# Action processor
# ---------------------------------------------------------------------------

def _process_action(
    action_type: str,
    service: str,
    scenario: dict,
    health: dict,
    circuit_breakers: list,
    step_count: int,
) -> tuple:
    """Returns (result_text, reward_value, done)."""

    root_service = scenario["root_cause_service"]
    fix_action = scenario["fix_action"]

    if action_type == "query_logs":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for query_logs.", -0.05, False
        logs = scenario["logs"].get(service, "[INFO] No unusual log entries.\n")
        reward = 0.2 if service == root_service else -0.05
        return f"[LOGS {service}]\n{logs}", reward, False

    elif action_type == "check_metrics":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for check_metrics.", -0.05, False
        metrics = scenario["metrics"].get(service, "cpu=5% | memory=20% | error_rate=0%")
        reward = 0.15 if service == root_service else -0.05
        return f"[METRICS {service}] {metrics}", reward, False

    elif action_type == "restart_service":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for restart_service.", -0.05, False
        if service == root_service and fix_action == "restart_service":
            # Correct fix!
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            reward = 0.5
            return (
                f"[OK] {service} restarted successfully. Health checks passing. "
                f"All dependent services recovering.",
                reward, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is already healthy. Restarting unnecessarily. Monitoring...", -0.1, False
        else:
            return f"[OK] {service} restarted. Monitoring for recovery...", 0.0, False

    elif action_type == "rollback_deploy":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for rollback_deploy.", -0.05, False
        if service == root_service and fix_action == "rollback_deploy":
            # Correct fix!
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            reward = 0.5
            return (
                f"[OK] Rolled back {service} to version 2.1.2. "
                f"Health checks passing. Service restored.",
                reward, False,
            )
        else:
            return f"[INFO] No recent problematic deployment found on {service}. Rollback skipped.", -0.05, False

    elif action_type == "enable_circuit_breaker":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for enable_circuit_breaker.", -0.05, False
        if service not in circuit_breakers:
            circuit_breakers.append(service)
        req_cb = scenario.get("requires_circuit_breaker")
        reward = 0.3 if req_cb and service == req_cb else 0.05
        return (
            f"[OK] Circuit breaker enabled on {service}. "
            f"Traffic rerouted. Downstream impact contained.",
            reward, False,
        )

    elif action_type == "scale_up":
        if not service or service not in health:
            return "[ERROR] Must specify a valid service for scale_up.", -0.05, False
        if service == root_service and fix_action == "scale_up":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            reward = 0.5
            return (
                f"[OK] {service} scaled to 6 instances. CPU normalizing. "
                f"All health checks passing.",
                reward, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is healthy — scaling up unnecessarily.", -0.1, False
        else:
            return f"[OK] Scaled up {service}. Monitoring...", 0.0, False

    elif action_type == "declare_resolved":
        _EPISODE_STATE["resolution_claim"] = service or "unknown"
        # Determine if correct
        all_healthy = all(h == "healthy" for h in health.values())
        root_fixed = health.get(root_service) == "healthy"
        if root_fixed and all_healthy:
            reward = 0.5
            result = "[RESOLVED] Incident resolved. All services healthy. Good work."
        elif root_fixed:
            reward = 0.3
            result = "[PARTIAL] Root service fixed but some dependents still degraded."
        else:
            reward = -0.1
            result = "[INCORRECT] Root cause not yet fixed. Incident persists."
        return result, reward, True  # always ends episode

    else:
        return (
            f"[ERROR] Unknown action_type: '{action_type}'. "
            f"Valid: {VALID_ACTION_TYPES}",
            -0.1, False,
        )


def _heal_downstream(service: str, health: dict, circuit_breakers: list):
    """When a service heals, propagate recovery to services that depended on it."""
    for dependent in HEAL_PROPAGATION.get(service, []):
        if dependent not in circuit_breakers and health.get(dependent) in ("degraded", "down"):
            health[dependent] = "healthy"
            _heal_downstream(dependent, health, circuit_breakers)


# ---------------------------------------------------------------------------
# Task graders (deterministic, 0.0 – 1.0)
# ---------------------------------------------------------------------------

def grader_task1(episode_history: list) -> float:
    """
    Easy: Single service outage (order-service, bad deploy).
    Score = investigated_correct_service(0.3) + took_correct_action(0.3) + declared_resolved(0.4)
    """
    if not episode_history:
        return 0.0

    investigated = any(
        h["action"]["action_type"] in ("query_logs", "check_metrics")
        and h["action"]["service"] == "order-service"
        for h in episode_history
    )

    correct_action = any(
        h["action"]["action_type"] == "rollback_deploy"
        and h["action"]["service"] == "order-service"
        for h in episode_history
    )

    declared = any(
        h["action"]["action_type"] == "declare_resolved"
        for h in episode_history
    )
    # Full credit for declared only if service was actually fixed
    final_health = episode_history[-1]["observation"]["service_health"]
    service_fixed = final_health.get("order-service") == "healthy"

    score = (
        0.3 * float(investigated)
        + 0.3 * float(correct_action)
        + 0.4 * float(declared and service_fixed)
    )
    return round(min(score, 1.0), 4)


def grader_task2(episode_history: list) -> float:
    """
    Medium: Cascading failure (auth-service CPU spike).
    Score = root_cause_investigated(0.4) + correct_fix(0.3) + declared_resolved(0.3)
    """
    if not episode_history:
        return 0.0

    investigated_root = any(
        h["action"]["action_type"] in ("query_logs", "check_metrics")
        and h["action"]["service"] == "auth-service"
        for h in episode_history
    )

    correct_fix = any(
        h["action"]["action_type"] == "scale_up"
        and h["action"]["service"] == "auth-service"
        for h in episode_history
    )

    declared = any(
        h["action"]["action_type"] == "declare_resolved"
        for h in episode_history
    )
    final_health = episode_history[-1]["observation"]["service_health"]
    root_fixed = final_health.get("auth-service") == "healthy"
    cascade_stopped = final_health.get("api-gateway") == "healthy"

    resolved_score = 0.3 * float(declared and root_fixed and cascade_stopped)

    score = (
        0.4 * float(investigated_root)
        + 0.3 * float(correct_fix)
        + resolved_score
    )
    return round(min(score, 1.0), 4)


def grader_task3(episode_history: list) -> float:
    """
    Hard: Multi-service cascade with noise (payment-service DB leak).
    Score = root_investigated(0.35) + circuit_breaker_used(0.25) + correct_fix(0.25) + efficiency(0.15)
    """
    if not episode_history:
        return 0.0

    investigated_root = any(
        h["action"]["action_type"] in ("query_logs", "check_metrics")
        and h["action"]["service"] == "payment-service"
        for h in episode_history
    )

    circuit_breaker_used = any(
        h["action"]["action_type"] == "enable_circuit_breaker"
        and h["action"]["service"] == "notification-service"
        for h in episode_history
    )

    correct_fix = any(
        h["action"]["action_type"] == "restart_service"
        and h["action"]["service"] == "payment-service"
        for h in episode_history
    )

    # Efficiency: fewer steps = better
    steps_taken = len(episode_history)
    if steps_taken <= 8:
        efficiency = 1.0
    elif steps_taken >= 20:
        efficiency = 0.0
    else:
        efficiency = 1.0 - (steps_taken - 8) / 12.0

    score = (
        0.35 * float(investigated_root)
        + 0.25 * float(circuit_breaker_used)
        + 0.25 * float(correct_fix)
        + 0.15 * efficiency
    )
    return round(min(score, 1.0), 4)


TASK_GRADERS = {
    "task1": grader_task1,
    "task2": grader_task2,
    "task3": grader_task3,
}
