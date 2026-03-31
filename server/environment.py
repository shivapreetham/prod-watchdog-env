"""
ProdWatchdog Environment — Improved v2.

Changes from v1:
  - Dynamic cascade spreading: failures worsen if root cause not fixed in time
  - No duplicate investigation rewards: re-querying same service gives no reward
  - Per-step urgency penalty: small negative reward per step to encourage efficiency
  - Better alert system: alerts appear/disappear based on actual service health
  - Harder scenarios: more convincing red herrings, noisier logs
  - Improved graders: efficiency components on task2, harder threshold on task3

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

# When service S heals, which services may also recover?
HEAL_PROPAGATION = {
    "api-gateway": [],
    "auth-service": ["api-gateway"],
    "order-service": ["api-gateway"],
    "payment-service": ["order-service", "notification-service"],
    "inventory-service": ["order-service"],
    "notification-service": [],
}

# Small per-step urgency penalty — encourages efficient investigation
STEP_PENALTY = -0.02


# ---------------------------------------------------------------------------
# Incident scenarios (deterministic per task_id)
# ---------------------------------------------------------------------------

SCENARIOS = {

    # ------------------------------------------------------------------
    # TASK 1 — Easy: Single service bad deployment
    # Root: order-service (rollback_deploy)
    # Noise: api-gateway showing errors — downstream effect, not cause
    # ------------------------------------------------------------------
    "task1": {
        "description": (
            "Single Service Outage — order-service has a bad deployment (v2.1.3). "
            "api-gateway is reporting upstream errors as a downstream effect."
        ),
        "root_cause_service": "order-service",
        "root_cause_type":    "bad_deploy",
        "fix_action":         "rollback_deploy",
        "requires_circuit_breaker": None,
        "max_steps": 10,

        "initial_health": {
            "api-gateway":        "degraded",
            "auth-service":       "healthy",
            "order-service":      "down",
            "payment-service":    "healthy",
            "inventory-service":  "healthy",
            "notification-service": "healthy",
        },

        # cascade_events: fires if root not fixed by threshold step
        "cascade_events": [],   # Task 1 is easy — no cascade spreading

        # Per-service, per-health-state alerts (dynamic: disappear when service heals)
        "service_alerts": {
            "order-service": {
                "down": [
                    "CRITICAL [order-service] Health check failed — 0/3 instances passing",
                    "ERROR    [order-service] 502 Bad Gateway rate: 91% over last 5 min",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] Upstream dependency failure — elevated 5xx rate",
                    "WARNING  [api-gateway] order-service upstream returning HTTP 502",
                ],
            },
        },

        # Static alerts — always visible, unrelated to health state (red herrings / context)
        "static_alerts": [],

        "logs": {
            "order-service": (
                "[ERROR] Application startup failed — environment configuration invalid\n"
                "[ERROR] Config validation: startup probe failed on /health (HTTP 500)\n"
                "[ERROR] Missing required runtime config — deployment v2.1.3 incomplete\n"
                "[INFO]  Deploy event: 2026-03-29T14:22:00Z version=v2.1.3 by CI pipeline\n"
                "[WARN]  Last stable version: v2.1.2 — rollback candidate available\n"
                "[ERROR] All 3 instances failing health checks — traffic routing stopped\n"
                "[ERROR] 502 responses to callers spiking — api-gateway affected\n"
            ),
            "api-gateway": (
                "[WARN] order-service upstream: HTTP 502 on 91% of /orders/* requests\n"
                "[INFO] auth-service upstream: OK (200 rate: 99.9%)\n"
                "[INFO] No configuration changes on api-gateway in last 72h\n"
                "[INFO] api-gateway itself is healthy — this is an upstream dependency issue\n"
                "[WARN] Retrying order-service requests with exponential backoff\n"
            ),
            "auth-service": "[INFO] All systems nominal. 200 OK rate: 99.8%. No anomalies.\n",
            "payment-service": "[INFO] Idle — no incoming requests from order-service.\n",
            "inventory-service": "[INFO] Idle — no stock reservation requests.\n",
            "notification-service": "[INFO] Queue empty. Awaiting order events.\n",
        },

        "metrics": {
            "order-service":        "cpu=2% | memory=18% | error_rate=91% | latency_p99=N/A (down)",
            "api-gateway":          "cpu=15% | memory=46% | error_rate=11% | latency_p99=420ms",
            "auth-service":         "cpu=13% | memory=40% | error_rate=0.1% | latency_p99=21ms",
            "payment-service":      "cpu=1%  | memory=10% | error_rate=0%   | latency_p99=N/A",
            "inventory-service":    "cpu=1%  | memory=10% | error_rate=0%   | latency_p99=N/A",
            "notification-service": "cpu=1%  | memory=8%  | error_rate=0%   | latency_p99=N/A",
        },
    },

    # ------------------------------------------------------------------
    # TASK 2 — Medium: CPU spike cascade + convincing red herring
    # Root: auth-service (scale_up)
    # Cascade: at step 8, order-service degrades if auth still down
    # Red herring: notification-service looks overloaded but is fine
    # ------------------------------------------------------------------
    "task2": {
        "description": (
            "Cascading Failure — auth-service CPU spike saturated all instances, "
            "degrading api-gateway. notification-service shows high resource usage "
            "from a concurrent marketing campaign — this is a red herring."
        ),
        "root_cause_service": "auth-service",
        "root_cause_type":    "cpu_spike",
        "fix_action":         "scale_up",
        "requires_circuit_breaker": None,
        "max_steps": 15,

        "initial_health": {
            "api-gateway":          "degraded",
            "auth-service":         "down",
            "order-service":        "healthy",
            "payment-service":      "healthy",
            "inventory-service":    "healthy",
            "notification-service": "healthy",
        },

        # If auth-service still down at step 8 → order-service degrades too
        "cascade_events": [
            {
                "at_step":    8,
                "target":     "order-service",
                "new_status": "degraded",
                "message": (
                    "[CASCADE] auth-service overload now blocking order authentication — "
                    "order-service checkout flow degrading"
                ),
            },
            {
                "at_step":    12,
                "target":     "order-service",
                "new_status": "down",
                "message": (
                    "[CASCADE] order-service auth retry budget exhausted — "
                    "service entering crash loop"
                ),
            },
        ],

        "service_alerts": {
            "auth-service": {
                "down": [
                    "CRITICAL [auth-service] CPU at 99% — all 3 instances saturated",
                    "CRITICAL [auth-service] JWT validation failing — 78% error rate",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] Elevated 5xx rate: 23% — auth-service upstream failing",
                    "WARNING  [api-gateway] Increased p99 latency: 890ms (SLA: 300ms)",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] Auth token validation failures — checkout degraded",
                ],
                "down": [
                    "CRITICAL [order-service] Auth retry budget exhausted — service down",
                ],
            },
        },

        # notification-service alert: always visible, this is the red herring
        "static_alerts": [
            "WARNING  [notification-service] Thread pool at 82% — high campaign email volume",
            "WARNING  [notification-service] p99 latency 2.3s (SLA: 1.5s) — send queue backing up",
        ],

        "logs": {
            "auth-service": (
                "[ERROR] CPU throttling active — request queue depth 4,200 and growing\n"
                "[ERROR] JWT validation timeouts: avg 3,200ms (target <50ms)\n"
                "[ERROR] Token cache hit rate: 6% — cache overwhelmed under load\n"
                "[ERROR] Instance auth-2 OOMKilled — JVM heap exhausted under CPU pressure\n"
                "[WARN]  Promo campaign started 2026-03-29T13:55Z — coincides with spike onset\n"
                "[ERROR] Instance auth-1 and auth-3 throttled at kernel level\n"
                "[INFO]  Current instance count: 3 — scaling limit not yet reached\n"
            ),
            "api-gateway": (
                "[WARN] auth-service: upstream 503 rate 78% — circuit breaker at 70% threshold\n"
                "[INFO] Retrying auth checks with 2s exponential backoff\n"
                "[INFO] No recent configuration changes — issue is upstream\n"
            ),
            "notification-service": (
                "[INFO]  Marketing campaign batch: 2.3M emails queued at 13:50Z\n"
                "[INFO]  Current send rate: 22k/min — within rated capacity (25k/min max)\n"
                "[WARN]  Thread pool utilization: 82% — high but not at limit\n"
                "[WARN]  External SMTP relay latency elevated: avg 1.8s (normal: 0.4s)\n"
                "[INFO]  All emails delivering successfully — zero bounce rate\n"
                "[INFO]  This service has no upstream dependency on auth-service\n"
            ),
            "order-service":    "[INFO] Normal operations. Auth tokens validating. Volume nominal.\n",
            "payment-service":  "[INFO] Processing normally. No upstream issues.\n",
            "inventory-service": "[INFO] Stock sync running. No anomalies.\n",
        },

        "metrics": {
            "auth-service":         "cpu=99% | memory=91% | error_rate=78% | latency_p99=3200ms",
            "api-gateway":          "cpu=28% | memory=52% | error_rate=23% | latency_p99=890ms",
            "order-service":        "cpu=18% | memory=44% | error_rate=0.2% | latency_p99=45ms",
            "payment-service":      "cpu=12% | memory=38% | error_rate=0.1% | latency_p99=55ms",
            "inventory-service":    "cpu=9%  | memory=30% | error_rate=0%   | latency_p99=30ms",
            "notification-service": "cpu=68% | memory=73% | error_rate=0.2% | latency_p99=2300ms",
        },
    },

    # ------------------------------------------------------------------
    # TASK 3 — Hard: DB connection leak + cascade + 2 deceptive red herrings
    # Root: payment-service (restart_service) + circuit breaker on notification-service
    # Cascade: step 7 → order-service DOWN, step 12 → api-gateway DEGRADED
    # Red herring 1: auth-service — DB errors but it's a different DB, different host
    # Red herring 2: inventory-service — slow queries but zero error rate
    # ------------------------------------------------------------------
    "task3": {
        "description": (
            "Multi-Service Cascade with Deceptive Noise — payment-service has a DB connection "
            "leak (pool exhausted), causing notification-service to crash-loop and "
            "order-service to degrade. Two red herrings: auth-service shows DB timeout errors "
            "(but different database host) and inventory-service shows high query latency "
            "(scheduled maintenance window). Act fast — blast radius grows each minute."
        ),
        "root_cause_service": "payment-service",
        "root_cause_type":    "db_leak",
        "fix_action":         "restart_service",
        "requires_circuit_breaker": "notification-service",
        "max_steps": 20,

        "initial_health": {
            "api-gateway":          "healthy",
            "auth-service":         "healthy",
            "order-service":        "degraded",
            "payment-service":      "degraded",
            "inventory-service":    "healthy",
            "notification-service": "down",
        },

        # Cascade spreads if payment-service not fixed
        "cascade_events": [
            {
                "at_step":    7,
                "target":     "order-service",
                "new_status": "down",
                "message": (
                    "[CASCADE] payment-service connection exhaustion now causing order-service "
                    "to crash — checkout completely unavailable"
                ),
            },
            {
                "at_step":    12,
                "target":     "api-gateway",
                "new_status": "degraded",
                "message": (
                    "[CASCADE] Upstream failures from order-service and payment-service now "
                    "overwhelming api-gateway retry buffers — gateway degraded"
                ),
            },
        ],

        "service_alerts": {
            "notification-service": {
                "down": [
                    "CRITICAL [notification-service] Service unreachable — crash loop detected",
                    "CRITICAL [notification-service] All 4 health checks failing for 8 minutes",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] Checkout flow degraded — payment callbacks timing out",
                    "WARNING  [order-service] 34% of orders stuck in pending state",
                ],
                "down": [
                    "CRITICAL [order-service] Checkout service down — payment dependency failure",
                ],
            },
            "payment-service": {
                "degraded": [
                    "WARNING  [payment-service] DB connection pool critical — 97% utilization",
                    "WARNING  [payment-service] Outbound webhook delivery failing",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] Upstream saturation — retry buffer at 90% capacity",
                ],
            },
        },

        # Static red-herring alerts always visible regardless of health state
        "static_alerts": [
            "WARNING  [auth-service] DB query timeouts: 4.2% of auth requests failing",
            "WARNING  [inventory-service] Slow query alert: p99 latency 4.8s (baseline: 0.4s)",
        ],

        "logs": {
            "notification-service": (
                "[ERROR] Cannot connect to payment-service webhook endpoint — ECONNREFUSED\n"
                "[ERROR] Webhook delivery retry budget exhausted (50/50 attempts)\n"
                "[ERROR] Worker thread pool blocked — all threads waiting on payment callbacks\n"
                "[ERROR] Service entering crash loop — restarting every 30s\n"
                "[INFO]  notification-service has no independent DB — fully dependent on payment webhooks\n"
            ),
            "order-service": (
                "[WARN] payment-service callback timeout after 30s — 34% of checkouts affected\n"
                "[INFO] Async payment fallback active for affected orders\n"
                "[WARN] Connection wait queue growing — payment dependency saturated\n"
                "[INFO] No code changes to order-service in last 7 days\n"
            ),
            "payment-service": (
                "[ERROR] DB connection pool: 97/100 connections active — new connections refused\n"
                "[ERROR] Connection leak in txn_commit() rollback path — connections not released on exception\n"
                "[ERROR] Webhook delivery threads blocked indefinitely waiting for DB connections\n"
                "[WARN]  Service uptime: 18d 6h — no restart since initial deployment\n"
                "[WARN]  DB pool leak rate: ~3 connections/min — pool fully exhausted in last 2h\n"
                "[INFO]  Leak introduced by unhandled exception path in v2.4.1 (deployed 18d ago)\n"
            ),
            "auth-service": (
                # Red herring: looks like DB issue, but different database, different host
                "[WARN] DB query timeout: user-auth-db query avg 480ms (normal: 12ms)\n"
                "[WARN] Third-party OAuth provider (oauth.external.com) returning 429 — rate limited\n"
                "[INFO] auth-db host: auth-postgres-1.internal — separate from payment-db\n"
                "[INFO] auth-db connection pool: 8/100 — fully healthy\n"
                "[INFO] Elevated auth failures are OAuth provider-side, not local DB issue\n"
                "[WARN] Retry logic active — elevated 401 error rate is transient\n"
            ),
            "inventory-service": (
                # Red herring: slow queries, but it's scheduled maintenance
                "[INFO] Weekly analytics report job started 2026-03-29T12:00Z\n"
                "[INFO] Heavy aggregation query running — expected duration ~3h\n"
                "[WARN] p99 query latency elevated: 4.8s (normal: 0.4s) — analytics job contention\n"
                "[INFO] Zero errors — all queries completing successfully, just slow\n"
                "[INFO] Maintenance window acknowledged by on-call team\n"
            ),
            "api-gateway": "[INFO] Routing normally. Upstream error rate: 0.2% (within SLA).\n",
        },

        "metrics": {
            "payment-service":      "cpu=19% | memory=74% | error_rate=12% | db_connections=97/100",
            "notification-service": "cpu=0%  | memory=0%  | error_rate=100% | latency_p99=N/A (down)",
            "order-service":        "cpu=24% | memory=50% | error_rate=3.4% | latency_p99=920ms",
            "auth-service":         "cpu=16% | memory=43% | error_rate=4.2% | latency_p99=480ms",
            "inventory-service":    "cpu=81% | memory=67% | error_rate=0%   | latency_p99=4800ms",
            "api-gateway":          "cpu=14% | memory=40% | error_rate=0.2% | latency_p99=120ms",
        },
    },
}


# ---------------------------------------------------------------------------
# Module-level episode state (persists across stateless HTTP requests)
# ---------------------------------------------------------------------------

_EPISODE_STATE: dict = {
    "task_id":          "task1",
    "step_count":       0,
    "service_health":   {},
    "episode_history":  [],
    "circuit_breakers": [],
    "services_queried": [],    # tracks which services have been investigated (logs or metrics)
    "resolved":         False,
    "resolution_claim": None,
    "episode_id":       None,
}


def get_episode_state() -> dict:
    return _EPISODE_STATE


# ---------------------------------------------------------------------------
# Alert computation — fully dynamic based on service health state
# ---------------------------------------------------------------------------

def _compute_alerts(scenario: dict, health: dict) -> list:
    """
    Build the active alert list from current service health.
    Alerts appear and disappear as service health changes.
    Static red-herring alerts always remain visible.
    """
    alerts = []
    service_alerts = scenario.get("service_alerts", {})

    for service, status in health.items():
        if status in ("down", "degraded"):
            svc_alerts = service_alerts.get(service, {}).get(status, [])
            if svc_alerts:
                alerts.extend(svc_alerts)
            else:
                # Generic fallback alert
                if status == "down":
                    alerts.append(f"CRITICAL [{service}] Service DOWN — all health checks failing")
                else:
                    alerts.append(f"WARNING  [{service}] Service DEGRADED — elevated error rate")

    # Static alerts (red herrings, context info — always visible)
    for alert in scenario.get("static_alerts", []):
        if alert not in alerts:
            alerts.append(alert)

    return list(dict.fromkeys(alerts))  # deduplicate, preserve order


# ---------------------------------------------------------------------------
# Cascade spreading — fires when root cause isn't fixed in time
# ---------------------------------------------------------------------------

def _apply_cascade_events(
    scenario: dict,
    health: dict,
    circuit_breakers: list,
    step_count: int,
) -> list:
    """
    Fire cascade events that trigger at the current step if root isn't fixed.
    Returns list of cascade message strings (empty if nothing happened).
    """
    root = scenario["root_cause_service"]
    if health.get(root) == "healthy":
        return []  # Root fixed — no more cascade spreading

    messages = []
    severity = {"healthy": 0, "degraded": 1, "down": 2}

    for event in scenario.get("cascade_events", []):
        target     = event["target"]
        threshold  = event["at_step"]
        new_status = event["new_status"]
        current    = health.get(target, "healthy")

        # Fire only at exactly the threshold step, and only if it worsens the state
        if (
            step_count == threshold
            and severity.get(current, 0) < severity.get(new_status, 1)
            and target not in circuit_breakers
        ):
            health[target] = new_status
            messages.append(event["message"])

    return messages


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class ProdWatchdogEnvironment(Environment):
    """
    Production Incident Response Environment.

    Simulates real on-call SRE work: cascading microservice failures,
    noisy alerts, red-herring logs. The agent must diagnose root cause,
    contain blast radius, and remediate before the cascade spreads.
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

        _EPISODE_STATE["task_id"]          = task_id
        _EPISODE_STATE["step_count"]       = 0
        _EPISODE_STATE["service_health"]   = copy.deepcopy(scenario["initial_health"])
        _EPISODE_STATE["episode_history"]  = []
        _EPISODE_STATE["circuit_breakers"] = []
        _EPISODE_STATE["services_queried"] = []
        _EPISODE_STATE["resolved"]         = False
        _EPISODE_STATE["resolution_claim"] = None
        _EPISODE_STATE["episode_id"]       = ep_id

        self._state = State(episode_id=ep_id, step_count=0)

        alerts = _compute_alerts(scenario, _EPISODE_STATE["service_health"])

        return ProdWatchdogObservation(
            alerts=alerts,
            service_health=copy.deepcopy(_EPISODE_STATE["service_health"]),
            last_action_result=(
                f"[INCIDENT STARTED] On-call paged. {scenario['description']} "
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
        task_id  = _EPISODE_STATE["task_id"]
        scenario = SCENARIOS[task_id]

        _EPISODE_STATE["step_count"] += 1
        step_count = _EPISODE_STATE["step_count"]
        max_steps  = scenario["max_steps"]

        health          = _EPISODE_STATE["service_health"]
        circuit_breakers = _EPISODE_STATE["circuit_breakers"]

        action_type = (action.action_type or "").strip().lower()
        service     = (action.service or "").strip().lower()

        # --- Process action ---
        result, reward_val, done = _process_action(
            action_type, service, scenario, health, circuit_breakers, step_count
        )

        # --- Apply urgency step penalty (discourages wasted steps) ---
        if not done:
            reward_val += STEP_PENALTY

        # --- Check for cascade spreading ---
        if not done:
            cascade_msgs = _apply_cascade_events(scenario, health, circuit_breakers, step_count)
            if cascade_msgs:
                result += "\n" + "\n".join(cascade_msgs)

        # --- Check max steps ---
        if step_count >= max_steps and not done:
            done = True
            result += f"\n[TIMEOUT] Maximum steps ({max_steps}) reached. Episode ending."

        # --- Record history ---
        _EPISODE_STATE["episode_history"].append({
            "action": {
                "action_type": action_type,
                "service":     service,
                "parameters":  action.parameters,
            },
            "observation": {
                "step_count":        step_count,
                "service_health":    copy.deepcopy(health),
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
            reward=round(reward_val, 4),
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
    action_type:     str,
    service:         str,
    scenario:        dict,
    health:          dict,
    circuit_breakers: list,
    step_count:      int,
) -> tuple:
    """Returns (result_text, reward_value, done). Step penalty applied separately."""

    root_service = scenario["root_cause_service"]
    fix_action   = scenario["fix_action"]

    # ---- QUERY LOGS ----
    if action_type == "query_logs":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for query_logs.", -0.1, False

        logs = scenario["logs"].get(service, "[INFO] No unusual log entries.\n")

        # No duplicate investigation reward
        if service in _EPISODE_STATE["services_queried"]:
            reward = 0.0  # already investigated, just data
        elif service == root_service:
            reward = 0.2
            _EPISODE_STATE["services_queried"].append(service)
        else:
            reward = -0.05
            _EPISODE_STATE["services_queried"].append(service)

        return f"[LOGS {service}]\n{logs}", reward, False

    # ---- CHECK METRICS ----
    elif action_type == "check_metrics":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for check_metrics.", -0.1, False

        metrics = scenario["metrics"].get(service, "cpu=5% | memory=20% | error_rate=0%")
        key = f"metrics:{service}"

        if key in _EPISODE_STATE["services_queried"]:
            reward = 0.0
        elif service == root_service:
            reward = 0.15
            _EPISODE_STATE["services_queried"].append(key)
        else:
            reward = -0.05
            _EPISODE_STATE["services_queried"].append(key)

        return f"[METRICS {service}] {metrics}", reward, False

    # ---- RESTART SERVICE ----
    elif action_type == "restart_service":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for restart_service.", -0.1, False

        if service == root_service and fix_action == "restart_service":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            return (
                f"[OK] {service} restarted. DB connections cleared. "
                f"Health checks passing. Dependent services recovering.",
                0.5, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is already healthy — unnecessary restart.", -0.1, False
        else:
            return f"[OK] {service} restarted. Monitoring for recovery...", 0.0, False

    # ---- ROLLBACK DEPLOY ----
    elif action_type == "rollback_deploy":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for rollback_deploy.", -0.1, False

        if service == root_service and fix_action == "rollback_deploy":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            return (
                f"[OK] Rolled back {service} to last stable version. "
                f"Health checks passing. Service restored.",
                0.5, False,
            )
        else:
            return (
                f"[INFO] No problematic recent deployment found on {service}. "
                f"Rollback not applicable.",
                -0.05, False,
            )

    # ---- ENABLE CIRCUIT BREAKER ----
    elif action_type == "enable_circuit_breaker":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for enable_circuit_breaker.", -0.1, False

        if service not in circuit_breakers:
            circuit_breakers.append(service)

        req_cb = scenario.get("requires_circuit_breaker")
        if req_cb and service == req_cb:
            reward = 0.3
            msg = (
                f"[OK] Circuit breaker enabled on {service}. "
                f"Downstream cascade contained. Blast radius limited."
            )
        elif health.get(service) == "healthy":
            reward = -0.1
            msg = f"[WARN] {service} is healthy — circuit breaker not needed."
        else:
            reward = 0.05
            msg = (
                f"[OK] Circuit breaker enabled on {service}. "
                f"Traffic rerouted. Impact partially contained."
            )
        return msg, reward, False

    # ---- SCALE UP ----
    elif action_type == "scale_up":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for scale_up.", -0.1, False

        if service == root_service and fix_action == "scale_up":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            return (
                f"[OK] {service} scaled to 6 instances. CPU normalizing. "
                f"All health checks passing.",
                0.5, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is healthy — scaling up unnecessarily.", -0.1, False
        else:
            return f"[OK] Scaled up {service}. Monitoring for recovery...", 0.0, False

    # ---- DECLARE RESOLVED ----
    elif action_type == "declare_resolved":
        _EPISODE_STATE["resolution_claim"] = service or "unknown"
        root_fixed = health.get(root_service) == "healthy"

        # Services that are down ONLY because of an intentional circuit breaker
        # do not count as unresolved — the circuit breaker IS the correct containment.
        unhealthy_non_cb = [
            svc for svc, h in health.items()
            if h != "healthy" and svc not in circuit_breakers
        ]
        all_resolved = root_fixed and len(unhealthy_non_cb) == 0

        if all_resolved:
            return "[RESOLVED] Incident resolved. Root fixed. Cascade contained. Good work.", 0.5, True
        elif root_fixed:
            return (
                "[PARTIAL] Root service fixed but some dependents still degraded. "
                "Consider checking for remaining cascade effects.",
                0.3, True,
            )
        else:
            return (
                "[INCORRECT] Root cause not yet resolved. Incident still active. "
                "Continue investigation.",
                -0.1, True,
            )

    # ---- UNKNOWN ACTION ----
    else:
        return (
            f"[ERROR] Unknown action_type: '{action_type}'. "
            f"Valid actions: {VALID_ACTION_TYPES}",
            -0.1, False,
        )


def _heal_downstream(service: str, health: dict, circuit_breakers: list):
    """Propagate recovery from a healed service to its dependents."""
    for dependent in HEAL_PROPAGATION.get(service, []):
        if dependent not in circuit_breakers and health.get(dependent) in ("degraded", "down"):
            health[dependent] = "healthy"
            _heal_downstream(dependent, health, circuit_breakers)


# ---------------------------------------------------------------------------
# Task graders (deterministic, 0.0 – 1.0)
# ---------------------------------------------------------------------------

def grader_task1(episode_history: list) -> float:
    """
    Easy: Single service outage (order-service bad deploy).
    The agent should not be confused by the downstream api-gateway alert.

    Score:
      0.30 × investigated_order_service (query_logs or check_metrics)
      0.30 × took_correct_action (rollback_deploy on order-service)
      0.40 × declared_resolved with service actually fixed
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
    declared = any(h["action"]["action_type"] == "declare_resolved" for h in episode_history)
    final_health = episode_history[-1]["observation"]["service_health"]
    service_fixed = final_health.get("order-service") == "healthy"

    score = (
        0.30 * float(investigated)
        + 0.30 * float(correct_action)
        + 0.40 * float(declared and service_fixed)
    )
    return round(min(score, 1.0), 4)


def grader_task2(episode_history: list) -> float:
    """
    Medium: Cascading failure (auth-service CPU spike → api-gateway degraded).
    Agent must find root cause before cascade spreads to order-service.

    Score:
      0.35 × investigated_root_cause (auth-service)
      0.25 × correct_fix (scale_up on auth-service)
      0.25 × declared resolved with root + cascade fixed
      0.15 × efficiency (steps ≤ 8 = 1.0, ≥ 15 = 0.0, linear)
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
    declared     = any(h["action"]["action_type"] == "declare_resolved" for h in episode_history)
    final_health = episode_history[-1]["observation"]["service_health"]
    root_fixed   = final_health.get("auth-service") == "healthy"
    cascade_ok   = final_health.get("api-gateway") == "healthy"

    steps_taken = len(episode_history)
    if steps_taken <= 6:
        efficiency = 1.0
    elif steps_taken >= 15:
        efficiency = 0.0
    else:
        efficiency = 1.0 - (steps_taken - 6) / 9.0

    score = (
        0.35 * float(investigated_root)
        + 0.25 * float(correct_fix)
        + 0.25 * float(declared and root_fixed and cascade_ok)
        + 0.15 * efficiency
    )
    return round(min(score, 1.0), 4)


def grader_task3(episode_history: list) -> float:
    """
    Hard: Multi-service cascade with deceptive red herrings.
    Agent must identify payment-service as root, use circuit breaker on
    notification-service, and act fast before cascade reaches api-gateway.

    Score:
      0.30 × investigated_root (payment-service)
      0.25 × circuit_breaker on notification-service
      0.25 × correct_fix (restart_service on payment-service)
      0.20 × efficiency (steps ≤ 6 = 1.0, ≥ 20 = 0.0, linear)
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

    steps_taken = len(episode_history)
    if steps_taken <= 6:
        efficiency = 1.0
    elif steps_taken >= 20:
        efficiency = 0.0
    else:
        efficiency = 1.0 - (steps_taken - 6) / 14.0

    score = (
        0.30 * float(investigated_root)
        + 0.25 * float(circuit_breaker_used)
        + 0.25 * float(correct_fix)
        + 0.20 * efficiency
    )
    return round(min(score, 1.0), 4)


TASK_GRADERS = {
    "task1": grader_task1,
    "task2": grader_task2,
    "task3": grader_task3,
}
