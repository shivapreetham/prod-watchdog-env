"""
ProdWatchdog Environment v3 — 6-Task SRE Incident Response.

Expanded service graph (11 services) with realistic infra layers:
  nginx-lb → api-gateway ← redis-cache
  api-gateway → auth-service → postgres-primary
  api-gateway → order-service → payment-service → postgres-primary
                             → inventory-service → postgres-replica
                             → kafka-broker → notification-service

Tasks (difficulty progression):
  task1 (easy)        — Redis Cache OOM → Thundering Herd
  task2 (easy)        — nginx-lb Bad Config Deploy
  task3 (medium)      — Kafka Broker Partition Leader Failure
  task4 (medium)      — Postgres Replica WAL Corruption
  task5 (medium-hard) — Auth Service JWT Memory Leak + Cascade
  task6 (hard)        — Postgres Primary Disk Full + Split-Brain (3-step fix)

Reward shaping: potential-based (Ng 1999) — dense signal every step.
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
    "nginx-lb",
    "api-gateway",
    "redis-cache",
    "auth-service",
    "order-service",
    "payment-service",
    "inventory-service",
    "notification-service",
    "kafka-broker",
    "postgres-primary",
    "postgres-replica",
]

# When service X heals, which services may auto-recover?
HEAL_PROPAGATION = {
    "nginx-lb":             ["api-gateway"],
    "api-gateway":          [],
    "redis-cache":          ["api-gateway"],
    "auth-service":         ["api-gateway", "order-service", "payment-service"],
    "order-service":        ["api-gateway"],
    "payment-service":      ["order-service", "notification-service"],
    "inventory-service":    ["order-service"],
    "notification-service": [],
    "kafka-broker":         ["order-service", "notification-service"],
    "postgres-primary":     ["auth-service", "payment-service", "order-service", "postgres-replica"],
    "postgres-replica":     ["inventory-service"],
}

# Per-step penalty to encourage efficiency
STEP_PENALTY = -0.02
GAMMA = 0.95

# Criticality weights for potential-based reward shaping
SERVICE_WEIGHTS = {
    "postgres-primary":     1.5,
    "kafka-broker":         1.4,
    "redis-cache":          1.3,
    "payment-service":      1.2,
    "order-service":        1.2,
    "api-gateway":          1.1,
    "auth-service":         1.0,
    "nginx-lb":             1.0,
    "notification-service": 0.9,
    "inventory-service":    0.8,
    "postgres-replica":     0.8,
}
HEALTH_VALUE = {"healthy": 1.0, "degraded": 0.4, "down": 0.0}
_TOTAL_WEIGHT = sum(SERVICE_WEIGHTS.values())


def _compute_potential(health: dict) -> float:
    """Weighted-average service health in [0, 1]. Used for shaping rewards."""
    return sum(
        SERVICE_WEIGHTS.get(svc, 1.0) * HEALTH_VALUE.get(status, 0.0)
        for svc, status in health.items()
    ) / _TOTAL_WEIGHT


# ---------------------------------------------------------------------------
# Incident scenarios
# ---------------------------------------------------------------------------

_H = "healthy"
_D = "degraded"
_X = "down"

SCENARIOS = {

    # ------------------------------------------------------------------
    # TASK 1 — Easy (12 steps)
    # Root: redis-cache (OOM → 100% cache miss → thundering herd on DB)
    # Fix: scale_up(redis-cache)
    # Tempting wrong: scale_up(api-gateway) or flush_cache(redis-cache)
    # Red herring: api-gateway 18% errors + 4200ms latency looks primary
    # Cascade step 6: nginx-lb degrades under connection overload
    # ------------------------------------------------------------------
    "task1": {
        "description": (
            "Cache Saturation — redis-cache has hit memory OOM (100% utilization). "
            "Cache hit rate is 0%: every request is a thundering herd hitting the DB directly. "
            "api-gateway p99 latency is 4200ms (SLA: 300ms) and error rate 18%. "
            "Act fast — nginx-lb will start dropping connections if redis-cache is not fixed."
        ),
        "root_cause_service":    "redis-cache",
        "root_cause_type":       "memory_oom",
        "fix_action":            "scale_up",
        "requires_circuit_breaker": None,
        "max_steps": 12,

        "initial_health": {
            "nginx-lb":             _H, "api-gateway":      _D,
            "redis-cache":          _X, "auth-service":     _H,
            "order-service":        _H, "payment-service":  _H,
            "inventory-service":    _H, "notification-service": _H,
            "kafka-broker":         _H, "postgres-primary": _H,
            "postgres-replica":     _H,
        },

        "cascade_events": [
            {
                "at_step": 6, "target": "nginx-lb", "new_status": "degraded",
                "message": (
                    "[CASCADE] redis-cache still OOM — thundering herd overwhelming "
                    "nginx-lb connection table (94% capacity). nginx-lb dropping new connections."
                ),
            },
        ],

        "service_alerts": {
            "redis-cache": {
                "down": [
                    "CRITICAL [redis-cache] OOM: memory 100% — eviction storm active",
                    "CRITICAL [redis-cache] Cache hit rate: 0% — all requests bypassing cache",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] p99 latency 4200ms (SLA: 300ms) — thundering herd",
                    "WARNING  [api-gateway] 5xx error rate: 18% — DB overloaded by direct requests",
                ],
            },
            "nginx-lb": {
                "degraded": [
                    "CRITICAL [nginx-lb] Connection table 94% capacity — dropping new connections",
                    "WARNING  [nginx-lb] Active connections: 47,200 / 50,000",
                ],
            },
        },

        "static_alerts": [
            "INFO     [order-service] DB query latency elevated: 380ms (normal 45ms) — cache miss load",
            "INFO     [auth-service] Token validation cache miss rate elevated — DB fallback active",
        ],

        "logs": {
            "redis-cache": (
                "[ERROR] OOM: maxmemory reached — 8192MB / 8192MB used\n"
                "[ERROR] All write ops rejected: ERR OOM command not allowed\n"
                "[ERROR] Key eviction rate: 142,000 keys/sec — LRU eviction storm\n"
                "[WARN]  Memory growth: +2.1GB over last 4h — session key TTL leak\n"
                "[ERROR] Cache hit rate dropped from 94% → 0% over 15 minutes\n"
                "[INFO]  Connected clients: 847 — all experiencing OOM errors\n"
                "[INFO]  Fix: scale_up redis-cache to increase memory allocation\n"
            ),
            "api-gateway": (
                "[WARN] redis-cache: all reads returning ERR OOM — falling back to DB\n"
                "[WARN] Backend DB query rate: 94,000 req/min (normal with cache: 8,200 req/min)\n"
                "[WARN] p99 latency 4200ms — all requests hitting DB directly\n"
                "[INFO] api-gateway config unchanged — upstream cache dependency issue\n"
                "[INFO] auth-service: responding normally (has local token cache)\n"
            ),
            "nginx-lb": (
                "[INFO] Connection routing nominal. No config changes in 30 days.\n"
                "[INFO] Active connections: 31,200 — within limits.\n"
                "[INFO] Upstream api-gateway responding. No upstream errors detected.\n"
            ),
            "auth-service": (
                "[INFO] JWT validation using local in-memory cache — not affected by redis OOM.\n"
                "[INFO] Token cache hit rate: 87% (local). All auth succeeding.\n"
            ),
            "order-service": (
                "[WARN] Session lookup latency: avg 380ms (normal: 12ms) — redis miss\n"
                "[INFO] Falling back to postgres for session data — no errors, just slow\n"
            ),
            "payment-service":      "[INFO] Processing normally. No upstream dependencies on redis-cache.\n",
            "inventory-service":    "[INFO] Stock sync nominal. postgres-replica healthy.\n",
            "notification-service": "[INFO] Kafka consumer running. No backlog.\n",
            "kafka-broker":         "[INFO] All partition leaders healthy. Consumer lag: 0ms.\n",
            "postgres-primary":     "[WARN] Connection pool: 141/200 elevated — cache miss fallback traffic.\n",
            "postgres-replica":     "[INFO] Replication lag: 0ms. Read queries serving normally.\n",
        },

        "metrics": {
            "redis-cache":          "cpu=98% | memory=100% | hit_rate=0% | eviction_rate=142k/sec",
            "api-gateway":          "cpu=71% | memory=68% | error_rate=18% | latency_p99=4200ms",
            "nginx-lb":             "cpu=22% | memory=35% | connections=31200 | error_rate=0.1%",
            "auth-service":         "cpu=14% | memory=41% | error_rate=0.2% | latency_p99=28ms",
            "order-service":        "cpu=38% | memory=54% | error_rate=1.1% | latency_p99=380ms",
            "payment-service":      "cpu=12% | memory=38% | error_rate=0.1% | latency_p99=55ms",
            "inventory-service":    "cpu=9%  | memory=29% | error_rate=0%   | latency_p99=30ms",
            "notification-service": "cpu=8%  | memory=22% | error_rate=0%   | latency_p99=N/A",
            "kafka-broker":         "cpu=11% | memory=44% | partitions_with_leader=8/8 | consumer_lag=0",
            "postgres-primary":     "cpu=89% | memory=78% | connections=141/200 | latency_p99=420ms",
            "postgres-replica":     "cpu=12% | memory=45% | connections=8/100  | latency_p99=38ms",
        },
    },

    # ------------------------------------------------------------------
    # TASK 2 — Easy (10 steps)
    # Root: nginx-lb (bad Nginx config deploy — worker connections misconfigured)
    # Fix: rollback_deploy(nginx-lb)
    # Tempting wrong: restart_service(api-gateway) or scale_up(api-gateway)
    # Red herring: api-gateway shows errors → looks like gateway config issue
    # Key insight: when EVERYTHING appears degraded, check the entry point first
    # ------------------------------------------------------------------
    "task2": {
        "description": (
            "Load Balancer Misconfiguration — nginx-lb received a bad config deploy. "
            "worker_connections was set to 4 (was 4096), causing nginx-lb to drop 99% of traffic. "
            "api-gateway sees constant upstream connection resets. "
            "All services appear degraded to external monitors but backends are actually healthy."
        ),
        "root_cause_service":    "nginx-lb",
        "root_cause_type":       "bad_config_deploy",
        "fix_action":            "rollback_deploy",
        "requires_circuit_breaker": None,
        "max_steps": 10,

        "initial_health": {
            "nginx-lb":             _X, "api-gateway":      _D,
            "redis-cache":          _H, "auth-service":     _H,
            "order-service":        _H, "payment-service":  _H,
            "inventory-service":    _H, "notification-service": _H,
            "kafka-broker":         _H, "postgres-primary": _H,
            "postgres-replica":     _H,
        },

        "cascade_events": [],  # nginx-lb down is already the critical failure — no further cascades

        "service_alerts": {
            "nginx-lb": {
                "down": [
                    "CRITICAL [nginx-lb] Worker process crashing — config validation failed at start",
                    "CRITICAL [nginx-lb] 99% of incoming connections rejected — worker_connections=4",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] Upstream connection resets: 99% of requests failing",
                    "WARNING  [api-gateway] nginx-lb keepalive connections exhausted — ECONNRESET flood",
                ],
            },
        },

        "static_alerts": [
            "WARNING  [api-gateway] 5xx error rate: 94% — appears to be gateway failure (it is NOT)",
            "INFO     [nginx-lb] Config deploy triggered at 2026-04-07T08:14Z by CI pipeline (commit a3f91b2)",
        ],

        "logs": {
            "nginx-lb": (
                "[ERROR] nginx: [emerg] 1024 worker_connections are not enough\n"
                "[ERROR] Worker process crashed: configuration validation failed\n"
                "[ERROR] Directive 'worker_connections' set to 4 (was 4096 in previous config)\n"
                "[ERROR] nginx: [warn] worker_rlimit_nofile is unset — using kernel default\n"
                "[INFO]  Config deploy at 2026-04-07T08:14Z — commit a3f91b2 'update nginx worker config'\n"
                "[INFO]  Previous stable config: worker_connections 4096 (commit 7de22a1)\n"
                "[INFO]  Fix: rollback_deploy nginx-lb to revert to commit 7de22a1\n"
            ),
            "api-gateway": (
                "[ERROR] ECONNRESET: nginx-lb upstream connection reset on 99% of requests\n"
                "[ERROR] Upstream keepalive pool exhausted — all slots occupied by reset connections\n"
                "[INFO]  api-gateway process is healthy — issue is the upstream nginx-lb\n"
                "[INFO]  Internal health check (bypass nginx): api-gateway returns 200 OK\n"
                "[INFO]  All backend services (auth, order, payment) responding normally\n"
            ),
            "auth-service":     "[INFO] All systems nominal. Internal health OK. 200 rate: 99.9%.\n",
            "order-service":    "[INFO] Processing normally. No errors. DB connections healthy.\n",
            "payment-service":  "[INFO] Transaction processing normal. No upstream errors.\n",
            "redis-cache":      "[INFO] Memory: 42%. Hit rate: 94%. All operations normal.\n",
            "inventory-service":    "[INFO] Stock sync normal. postgres-replica healthy.\n",
            "notification-service": "[INFO] Kafka consumer running. No backlog. All healthy.\n",
            "kafka-broker":         "[INFO] All partitions healthy. Consumer lag: 0ms.\n",
            "postgres-primary":     "[INFO] Connection pool: 22/200. Write throughput normal.\n",
            "postgres-replica":     "[INFO] Replication lag: 0ms. Read workload normal.\n",
        },

        "metrics": {
            "nginx-lb":             "cpu=2%  | memory=8%  | connections=4/4 | error_rate=99%",
            "api-gateway":          "cpu=44% | memory=61% | error_rate=94% | latency_p99=N/A (resets)",
            "redis-cache":          "cpu=14% | memory=42% | hit_rate=94%   | eviction_rate=0",
            "auth-service":         "cpu=13% | memory=40% | error_rate=0.1% | latency_p99=22ms",
            "order-service":        "cpu=18% | memory=44% | error_rate=0.2% | latency_p99=45ms",
            "payment-service":      "cpu=12% | memory=38% | error_rate=0.1% | latency_p99=55ms",
            "inventory-service":    "cpu=8%  | memory=28% | error_rate=0%   | latency_p99=30ms",
            "notification-service": "cpu=8%  | memory=22% | error_rate=0%   | latency_p99=N/A",
            "kafka-broker":         "cpu=11% | memory=44% | partitions_with_leader=8/8 | consumer_lag=0",
            "postgres-primary":     "cpu=22% | memory=55% | connections=22/200 | latency_p99=38ms",
            "postgres-replica":     "cpu=9%  | memory=40% | connections=4/100  | latency_p99=30ms",
        },
    },

    # ------------------------------------------------------------------
    # TASK 3 — Medium (15 steps)
    # Root: kafka-broker (JVM GC pause → broker fenced → partition leader failure)
    # Fix: restart_service(kafka-broker)
    # Tempting wrong: restart_service(notification-service) or rebalance_partitions(kafka-broker)
    # Red herring 1: notification-service consumer lag 847k messages (symptom)
    # Red herring 2: order-service memory 78% (write backpressure, not leak)
    # Cascade step 8: order-service=down; step 12: notification-service=down
    # ------------------------------------------------------------------
    "task3": {
        "description": (
            "Kafka Broker Failure — kafka-broker partition leader election failed after a 47s JVM GC pause. "
            "order-service producer is timing out silently (orders queued locally, memory growing). "
            "notification-service consumer group stuck with 847k undelivered messages. "
            "Red herrings: notification consumer lag and order-service memory spike."
        ),
        "root_cause_service":    "kafka-broker",
        "root_cause_type":       "partition_leader_failure",
        "fix_action":            "restart_service",
        "requires_circuit_breaker": None,
        "max_steps": 15,

        "initial_health": {
            "nginx-lb":             _H, "api-gateway":      _H,
            "redis-cache":          _H, "auth-service":     _H,
            "order-service":        _D, "payment-service":  _H,
            "inventory-service":    _H, "notification-service": _D,
            "kafka-broker":         _X, "postgres-primary": _H,
            "postgres-replica":     _H,
        },

        "cascade_events": [
            {
                "at_step": 8, "target": "order-service", "new_status": "down",
                "message": (
                    "[CASCADE] kafka-broker still down — order-service local producer buffer full (2GB). "
                    "New orders now being rejected. order-service entering error state."
                ),
            },
            {
                "at_step": 12, "target": "notification-service", "new_status": "down",
                "message": (
                    "[CASCADE] notification-service consumer group session timeout after 12 empty polls. "
                    "Consumer group rebalancing loop — service down."
                ),
            },
        ],

        "service_alerts": {
            "kafka-broker": {
                "down": [
                    "CRITICAL [kafka-broker] Partition leader election failed — 3/8 partitions leaderless",
                    "CRITICAL [kafka-broker] JVM GC pause 47s — broker fenced by cluster controller",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] Kafka producer timeout: 92% async writes queued locally",
                    "WARNING  [order-service] Local producer buffer: 1.4GB / 2GB — backpressure",
                ],
                "down": [
                    "CRITICAL [order-service] Kafka producer buffer full — new orders rejected",
                    "CRITICAL [order-service] ProducerFencedException: buffer overflow",
                ],
            },
            "notification-service": {
                "degraded": [
                    "WARNING  [notification-service] Consumer group lag: 847,293 messages unprocessed",
                    "WARNING  [notification-service] poll() returning 0 records — broker unreachable",
                ],
                "down": [
                    "CRITICAL [notification-service] Consumer group session timeout — rebalancing loop",
                    "CRITICAL [notification-service] All consumer threads blocked on poll()",
                ],
            },
        },

        "static_alerts": [
            "WARNING  [order-service] Memory: 78% (1.4GB producer buffer) — high but not OOM",
            "INFO     [postgres-primary] Connection pool 38/200 — normal load",
        ],

        "logs": {
            "kafka-broker": (
                "[ERROR] BrokerFencedException: this broker fenced by cluster controller\n"
                "[ERROR] GC pause duration: 47,312ms — exceeded heartbeat.interval.ms (3000ms)\n"
                "[ERROR] Partition leader election failed: partitions [orders-0, orders-1, orders-2] no leader\n"
                "[ERROR] Controller removed broker from ISR — epoch mismatch after GC\n"
                "[WARN]  JVM heap: 28GB / 30GB — GC pressure from large in-memory message index\n"
                "[WARN]  Broker uptime: 42 days — JVM heap fragmented over long run\n"
                "[INFO]  Fix: restart kafka-broker to rejoin cluster and trigger new leader election\n"
                "[INFO]  kafka-broker-2 and kafka-broker-3 healthy but cannot elect leaders without broker-1\n"
            ),
            "order-service": (
                "[WARN] KafkaProducer: send() timeout after 30000ms — broker not responding\n"
                "[WARN] Producer buffer occupancy: 1,412MB / 2,048MB — backpressure active\n"
                "[INFO] Orders being queued locally — no data loss yet, buffer filling\n"
                "[INFO] Postgres writes (inventory, payment) are succeeding — only Kafka affected\n"
                "[WARN] Memory usage elevated due to local Kafka buffer: 78%\n"
                "[INFO] No code changes to order-service in last 14 days\n"
            ),
            "notification-service": (
                "[WARN] KafkaConsumer: poll() returning 0 records for 847 consecutive calls\n"
                "[WARN] Consumer group 'notification-consumers' lag: 847,293 messages pending\n"
                "[WARN] Broker connection lost: kafka-broker-1:9092 — UNREACHABLE\n"
                "[INFO] notification-service itself is healthy — waiting for broker to recover\n"
                "[INFO] No independent DB or external dependencies — purely Kafka-driven\n"
                "[INFO] Once kafka-broker restarts, consumer will auto-resume from last committed offset\n"
            ),
            "api-gateway":      "[INFO] Routing normally. All upstreams responding. No anomalies.\n",
            "nginx-lb":         "[INFO] Connections nominal. No upstream errors.\n",
            "auth-service":     "[INFO] JWT validation normal. No DB issues. Latency nominal.\n",
            "payment-service":  "[INFO] Processing normally. DB connections healthy.\n",
            "inventory-service":"[INFO] Stock queries from postgres-replica nominal.\n",
            "redis-cache":      "[INFO] Memory: 42%. Hit rate: 94%. All healthy.\n",
            "postgres-primary": "[INFO] Connection pool 38/200. Write throughput normal.\n",
            "postgres-replica": "[INFO] Replication lag: 0ms. Read workload normal.\n",
        },

        "metrics": {
            "kafka-broker":         "cpu=2%  | memory=93% | partitions_with_leader=5/8 | gc_pause=47312ms",
            "order-service":        "cpu=31% | memory=78% | error_rate=8.2% | producer_buffer=1412MB/2048MB",
            "notification-service": "cpu=4%  | memory=28% | consumer_lag=847293 | error_rate=0%",
            "api-gateway":          "cpu=18% | memory=46% | error_rate=0.2% | latency_p99=95ms",
            "nginx-lb":             "cpu=9%  | memory=28% | connections=12100 | error_rate=0%",
            "auth-service":         "cpu=13% | memory=40% | error_rate=0.1% | latency_p99=22ms",
            "payment-service":      "cpu=12% | memory=38% | error_rate=0.1% | latency_p99=55ms",
            "inventory-service":    "cpu=8%  | memory=28% | error_rate=0%   | latency_p99=30ms",
            "redis-cache":          "cpu=15% | memory=42% | hit_rate=94%    | eviction_rate=0",
            "postgres-primary":     "cpu=34% | memory=58% | connections=38/200 | latency_p99=42ms",
            "postgres-replica":     "cpu=11% | memory=42% | connections=6/100  | latency_p99=35ms",
        },
    },

    # ------------------------------------------------------------------
    # TASK 4 — Medium (15 steps)
    # Root: postgres-replica (WAL segment corruption after OS-level force stop)
    # Fix: restart_service(postgres-replica) — clears bad WAL, re-syncs from primary
    # Tempting wrong: restart_service(postgres-primary) or restart_service(inventory-service)
    # Red herring: postgres-primary shows high CPU (fallback read traffic, not the cause)
    # Red herring: order-service shows DB errors (downstream of inventory failure)
    # Cascade step 9: order-service=down
    # ------------------------------------------------------------------
    "task4": {
        "description": (
            "Postgres Replica Corruption — postgres-replica has a corrupted WAL segment "
            "after an unclean OS-level shutdown 2h ago. inventory-service reads are all failing. "
            "order-service cannot verify stock availability. "
            "Red herring: postgres-primary shows high CPU (it's absorbing fallback read traffic, not the root)."
        ),
        "root_cause_service":    "postgres-replica",
        "root_cause_type":       "wal_corruption",
        "fix_action":            "restart_service",
        "requires_circuit_breaker": None,
        "max_steps": 15,

        "initial_health": {
            "nginx-lb":             _H, "api-gateway":      _H,
            "redis-cache":          _H, "auth-service":     _H,
            "order-service":        _D, "payment-service":  _H,
            "inventory-service":    _D, "notification-service": _H,
            "kafka-broker":         _H, "postgres-primary": _H,
            "postgres-replica":     _X,
        },

        "cascade_events": [
            {
                "at_step": 9, "target": "order-service", "new_status": "down",
                "message": (
                    "[CASCADE] postgres-replica still down — inventory-service stock checks fully failing. "
                    "order-service cannot place any orders — service down."
                ),
            },
        ],

        "service_alerts": {
            "postgres-replica": {
                "down": [
                    "CRITICAL [postgres-replica] WAL replay failed — CRC mismatch in pg_wal segment",
                    "CRITICAL [postgres-replica] Crash recovery failed — all connections rejected",
                ],
            },
            "inventory-service": {
                "degraded": [
                    "WARNING  [inventory-service] DB reads failing: postgres-replica connection refused",
                    "WARNING  [inventory-service] Stock level checks returning errors — 100% failure",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] 34% of orders failing — inventory check unavailable",
                    "WARNING  [order-service] stock_check() throwing DB connection error",
                ],
                "down": [
                    "CRITICAL [order-service] All order placements failing — inventory dependency down",
                ],
            },
        },

        "static_alerts": [
            "WARNING  [postgres-primary] CPU 87% — elevated read fallback traffic from inventory-service",
            "WARNING  [postgres-primary] Connections: 94/200 — higher than normal (normal: 12 write-only)",
        ],

        "logs": {
            "postgres-replica": (
                "[ERROR] WAL segment corruption: pg_wal/0000000200000003000000A4 CRC mismatch\n"
                "[ERROR] WAL replay failed at LSN 3/A4001234 — cannot continue replication stream\n"
                "[ERROR] Crash recovery failed — corrupt WAL segment cannot be replayed\n"
                "[ERROR] All connections rejected: FATAL database system is in crash recovery mode\n"
                "[WARN]  Force-stopped during OS kernel upgrade at 2026-04-07T06:30Z\n"
                "[INFO]  postgres-primary is healthy — re-sync estimated 45 seconds after restart\n"
                "[INFO]  Fix: restart postgres-replica to clear bad WAL buffer and re-sync from primary\n"
            ),
            "postgres-primary": (
                "[WARN]  Unusual read traffic: 89 connections (normal write workload: 12)\n"
                "[WARN]  Connection pool 94% utilized — inventory-service read fallback traffic\n"
                "[INFO]  Write performance: nominal. No errors on write path.\n"
                "[INFO]  postgres-primary is healthy — elevated load is from inventory read fallback\n"
                "[INFO]  Root issue is postgres-replica WAL corruption — fix the replica\n"
            ),
            "inventory-service": (
                "[ERROR] DB connection failed: postgres-replica:5432 — Connection refused\n"
                "[ERROR] All read queries failing: FATAL database system is in crash recovery mode\n"
                "[WARN]  Fallback to postgres-primary attempted but adding to primary load\n"
                "[INFO]  inventory-service itself is healthy — this is a DB infrastructure issue\n"
                "[INFO]  No code changes to inventory-service in last 30 days\n"
            ),
            "order-service": (
                "[WARN] inventory-service stock_check() failing: postgres-replica connection refused\n"
                "[WARN] 34% of order placements failing — cannot verify stock availability\n"
                "[INFO] Payment processing still working — postgres-primary write path is healthy\n"
                "[INFO] No code changes to order-service\n"
                "[WARN] This is an inventory dependency issue, not an order-service bug\n"
            ),
            "api-gateway":      "[INFO] Routing normally. auth and payment upstreams OK.\n",
            "nginx-lb":         "[INFO] Connections nominal. No upstream errors.\n",
            "auth-service":     "[INFO] JWT validation normal. No DB issues.\n",
            "payment-service":  "[INFO] Processing normally. postgres-primary write path healthy.\n",
            "redis-cache":      "[INFO] Memory 44%. Hit rate 93%. All operations normal.\n",
            "kafka-broker":     "[INFO] All partitions healthy. Consumer lag: 0ms.\n",
            "notification-service": "[INFO] Kafka consumer running. No backlog.\n",
        },

        "metrics": {
            "postgres-replica":     "cpu=0%  | memory=0%  | status=DOWN (crash recovery) | connections=0",
            "postgres-primary":     "cpu=87% | memory=78% | connections=94/200 | latency_p99=280ms",
            "inventory-service":    "cpu=18% | memory=41% | error_rate=100% | db_connections=0",
            "order-service":        "cpu=24% | memory=50% | error_rate=34%  | latency_p99=920ms",
            "api-gateway":          "cpu=19% | memory=44% | error_rate=1.2% | latency_p99=145ms",
            "nginx-lb":             "cpu=8%  | memory=26% | connections=14200 | error_rate=0%",
            "auth-service":         "cpu=13% | memory=40% | error_rate=0.1% | latency_p99=22ms",
            "payment-service":      "cpu=14% | memory=39% | error_rate=0.1% | latency_p99=55ms",
            "notification-service": "cpu=9%  | memory=24% | error_rate=0%   | latency_p99=N/A",
            "redis-cache":          "cpu=14% | memory=44% | hit_rate=93%    | eviction_rate=0",
            "kafka-broker":         "cpu=12% | memory=46% | partitions_with_leader=8/8 | consumer_lag=0",
        },
    },

    # ------------------------------------------------------------------
    # TASK 5 — Medium-Hard (20 steps)
    # Root: auth-service (JWT validation cache memory leak → GC pauses → latency spike)
    # Fix: restart_service(auth-service) — clears heap, resets JWT cache
    # Requires CB: enable_circuit_breaker(api-gateway) — contain retry storm
    # Tempting wrong: scale_up(api-gateway) (api-gateway looks overloaded but isn't root)
    # Red herring: api-gateway 34% error rate + high CPU looks like gateway config issue
    # Cascade step 8: order-service=degraded; step 14: auth-service=down, payment-service=degraded
    # ------------------------------------------------------------------
    "task5": {
        "description": (
            "Auth Service Memory Leak — auth-service JWT validation cache has a 7-day memory leak. "
            "JVM heap is at 96%: GC pauses causing 4s latency spikes → api-gateway timeouts. "
            "order-service checkout auth failures rising. "
            "Enable circuit breaker on api-gateway to stop retry storm, then restart auth-service. "
            "Red herring: api-gateway 34% error rate looks like gateway misconfiguration."
        ),
        "root_cause_service":    "auth-service",
        "root_cause_type":       "jwt_cache_memory_leak",
        "fix_action":            "restart_service",
        "requires_circuit_breaker": "api-gateway",
        "max_steps": 20,

        "initial_health": {
            "nginx-lb":             _H, "api-gateway":      _D,
            "redis-cache":          _H, "auth-service":     _D,
            "order-service":        _H, "payment-service":  _H,
            "inventory-service":    _H, "notification-service": _H,
            "kafka-broker":         _H, "postgres-primary": _H,
            "postgres-replica":     _H,
        },

        "cascade_events": [
            {
                "at_step": 8, "target": "order-service", "new_status": "degraded",
                "message": (
                    "[CASCADE] auth-service GC pauses causing JWT validation timeouts. "
                    "order-service checkout auth failures now exceeding threshold."
                ),
            },
            {
                "at_step": 14, "target": "auth-service", "new_status": "down",
                "message": (
                    "[CASCADE] auth-service JVM heap exhausted — OOMKilled. Service down."
                ),
            },
            {
                "at_step": 14, "target": "payment-service", "new_status": "degraded",
                "message": (
                    "[CASCADE] payment-service payment confirmation auth checks now failing — degraded."
                ),
            },
        ],

        "service_alerts": {
            "auth-service": {
                "degraded": [
                    "WARNING  [auth-service] JVM heap: 96% — GC pauses causing 4s latency spikes",
                    "WARNING  [auth-service] JWT cache size: 18GB / 19GB heap — cache not evicting",
                ],
                "down": [
                    "CRITICAL [auth-service] OOMKilled — JVM heap exhausted",
                    "CRITICAL [auth-service] All JWT validation failing — services unauthenticated",
                ],
            },
            "api-gateway": {
                "degraded": [
                    "WARNING  [api-gateway] auth-service upstream 4s timeouts — 34% of requests failing",
                    "WARNING  [api-gateway] Retry storm: 4,200 in-flight retries — CPU 78%",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] Auth token validation timing out — checkout degraded",
                    "WARNING  [order-service] 28% of checkout attempts failing JWT validation",
                ],
            },
            "payment-service": {
                "degraded": [
                    "WARNING  [payment-service] Payment auth confirmation failing — JWT timeouts",
                ],
            },
        },

        "static_alerts": [
            "WARNING  [api-gateway] CPU 78% — retry storm from auth timeouts (not gateway bug)",
            "INFO     [auth-service] Last restart: 7 days ago — JWT cache memory has been growing since",
        ],

        "logs": {
            "auth-service": (
                "[ERROR] GC pause: 4,182ms — Stop-the-World GC triggered by heap pressure\n"
                "[ERROR] JVM heap: 18,432MB / 19,200MB (96%) — JWT validation cache not evicting\n"
                "[ERROR] JWT cache entries: 142M objects — TTL eviction thread appears deadlocked\n"
                "[WARN]  JWT validation avg latency: 4,100ms (SLA: 50ms) — all GC-paused\n"
                "[WARN]  auth-service uptime: 7d 3h — heap leak accumulating since last restart\n"
                "[INFO]  Fix: restart auth-service to clear JVM heap and reset JWT cache\n"
                "[INFO]  After restart, expected: heap <2GB, latency <50ms, GC pauses <10ms\n"
            ),
            "api-gateway": (
                "[ERROR] auth-service upstream: 34% requests timing out after 4s\n"
                "[ERROR] Retry storm: 4,200 in-flight retry attempts — CPU elevated\n"
                "[INFO]  api-gateway config unchanged — issue is auth-service upstream\n"
                "[INFO]  scale_up api-gateway will NOT help — bottleneck is auth latency, not api capacity\n"
                "[WARN]  enable_circuit_breaker on api-gateway will stop retry storm while auth restarts\n"
            ),
            "order-service": (
                "[WARN] JWT token validation timing out: avg 4.1s on 28% of checkout requests\n"
                "[INFO] No code changes to order-service in last 14 days\n"
                "[INFO] This is an upstream auth-service dependency issue\n"
            ),
            "payment-service":  "[INFO] Processing normally. No upstream auth issues yet.\n",
            "nginx-lb":         "[INFO] Connections nominal. api-gateway upstream errors only.\n",
            "redis-cache":      "[INFO] Memory 42%. Hit rate 93%. All normal.\n",
            "inventory-service":"[INFO] Stock sync normal. All healthy.\n",
            "notification-service": "[INFO] Kafka consumer running. No backlog.\n",
            "kafka-broker":     "[INFO] All partitions healthy. Consumer lag: 0ms.\n",
            "postgres-primary": "[INFO] Connection pool 28/200. Write throughput normal.\n",
            "postgres-replica": "[INFO] Replication lag: 0ms. Read workload normal.\n",
        },

        "metrics": {
            "auth-service":         "cpu=94% | memory=96% | error_rate=34% | latency_p99=4100ms | gc_pause=4182ms",
            "api-gateway":          "cpu=78% | memory=62% | error_rate=34% | latency_p99=4800ms",
            "nginx-lb":             "cpu=18% | memory=32% | connections=22100 | error_rate=0.1%",
            "redis-cache":          "cpu=14% | memory=42% | hit_rate=93%    | eviction_rate=0",
            "order-service":        "cpu=28% | memory=48% | error_rate=5.8% | latency_p99=4200ms",
            "payment-service":      "cpu=14% | memory=39% | error_rate=0.2% | latency_p99=58ms",
            "inventory-service":    "cpu=8%  | memory=28% | error_rate=0%   | latency_p99=30ms",
            "notification-service": "cpu=9%  | memory=22% | error_rate=0%   | latency_p99=N/A",
            "kafka-broker":         "cpu=12% | memory=46% | partitions_with_leader=8/8 | consumer_lag=0",
            "postgres-primary":     "cpu=24% | memory=58% | connections=28/200 | latency_p99=40ms",
            "postgres-replica":     "cpu=10% | memory=40% | connections=4/100  | latency_p99=32ms",
        },
    },

    # ------------------------------------------------------------------
    # TASK 6 — Hard (25 steps)
    # Root: postgres-primary (disk full → auto-failover ran but replica still read-only)
    # 3-step fix in order:
    #   1. enable_circuit_breaker(payment-service) — stop write storm
    #   2. rollback_deploy(postgres-primary)       — restore from snapshot
    #   3. restart_service(payment-service)         — reconnect to restored primary
    # Tempting wrong: promote_replica(postgres-replica) — already promoted, still read-only
    # Red herring 1: auth-service 23% errors look like JWT key rotation issue
    # Red herring 2: order-service memory growth looks like memory leak
    # Cascade step 7: inventory-service=degraded; step 12: order-service=down; step 16: auth-service=down
    # ------------------------------------------------------------------
    "task6": {
        "description": (
            "Postgres Split-Brain — postgres-primary failed (disk 100% full), auto-failover ran. "
            "BUT postgres-replica is still read-only (postgresql.conf not updated). "
            "payment-service write transactions failing (ReadOnlyDatabaseException). "
            "order-service write retries accumulating memory. auth-service sessions can't persist. "
            "Fix order: (1) circuit_breaker(payment-service), (2) rollback_deploy(postgres-primary), "
            "(3) restart_service(payment-service) after primary is restored."
        ),
        "root_cause_service":    "postgres-primary",
        "root_cause_type":       "disk_full_split_brain",
        "fix_action":            "rollback_deploy",
        "requires_circuit_breaker": "payment-service",
        "secondary_fix_service": "payment-service",
        "secondary_fix_action":  "restart_service",
        "max_steps": 25,

        "initial_health": {
            "nginx-lb":             _H, "api-gateway":      _H,
            "redis-cache":          _H, "auth-service":     _D,
            "order-service":        _D, "payment-service":  _D,
            "inventory-service":    _H, "notification-service": _H,
            "kafka-broker":         _H, "postgres-primary": _X,
            "postgres-replica":     _D,
        },

        "cascade_events": [
            {
                "at_step": 7, "target": "inventory-service", "new_status": "degraded",
                "message": (
                    "[CASCADE] postgres-replica overloaded — now handling both reads AND writes "
                    "(split-brain write traffic routed here). inventory-service read queries degraded."
                ),
            },
            {
                "at_step": 12, "target": "order-service", "new_status": "down",
                "message": (
                    "[CASCADE] order-service write retry accumulation: memory 94%. "
                    "OOM imminent. order-service entering crash loop."
                ),
            },
            {
                "at_step": 16, "target": "auth-service", "new_status": "down",
                "message": (
                    "[CASCADE] auth-service session store fully exhausted — cannot persist any new tokens. "
                    "All login attempts failing. auth-service down."
                ),
            },
        ],

        "service_alerts": {
            "postgres-primary": {
                "down": [
                    "CRITICAL [postgres-primary] DOWN — disk 100% full, PostgreSQL OOM-killed",
                    "CRITICAL [postgres-primary] Auto-failover triggered to postgres-replica (READ-ONLY)",
                ],
            },
            "postgres-replica": {
                "degraded": [
                    "WARNING  [postgres-replica] Rejecting writes — read-only mode active (split-brain)",
                    "WARNING  [postgres-replica] ERROR: cannot execute INSERT in a read-only transaction",
                ],
            },
            "payment-service": {
                "degraded": [
                    "WARNING  [payment-service] Write transactions failing: ReadOnlyDatabaseException",
                    "WARNING  [payment-service] Transaction rollback rate: 89%",
                ],
            },
            "order-service": {
                "degraded": [
                    "WARNING  [order-service] Order commits failing — postgres write rejected",
                    "WARNING  [order-service] Memory: 71% — write retry accumulation",
                ],
                "down": [
                    "CRITICAL [order-service] OOM: write retry buffer exhausted — service down",
                ],
            },
            "auth-service": {
                "degraded": [
                    "WARNING  [auth-service] Session persistence failing — DB write rejected",
                    "WARNING  [auth-service] New logins failing: cannot write session token",
                ],
                "down": [
                    "CRITICAL [auth-service] Session store exhausted — all logins failing",
                ],
            },
            "inventory-service": {
                "degraded": [
                    "WARNING  [inventory-service] postgres-replica query latency: 4.8s (replica overloaded)",
                ],
            },
        },

        "static_alerts": [
            "WARNING  [auth-service] JWT errors 23% — LOOKS like key rotation issue (it is NOT)",
            "WARNING  [order-service] Memory +180MB/hr — LOOKS like memory leak (it is write-retry accumulation)",
            "INFO     [postgres-primary] STONITH fencing at 2026-04-07T03:14Z — disk was 100%",
        ],

        "logs": {
            "postgres-primary": (
                "[ERROR] PostgreSQL OOM-killed — disk at 100% (500GB / 500GB)\n"
                "[ERROR] Failed to write WAL segment — disk full, aborting all transactions\n"
                "[INFO]  Auto-failover triggered at 2026-04-07T03:14:22Z by Patroni HA manager\n"
                "[INFO]  postgres-replica promoted by Patroni\n"
                "[WARN]  Promotion incomplete — postgresql.conf still has 'default_transaction_read_only=on'\n"
                "[INFO]  Fix: rollback_deploy postgres-primary to restore from snapshot (2h ago, 450GB)\n"
            ),
            "postgres-replica": (
                "[ERROR] Rejecting write: ERROR: cannot execute INSERT in a read-only transaction\n"
                "[ERROR] Rejecting write: ERROR: cannot execute UPDATE in a read-only transaction\n"
                "[WARN]  Promoted as primary but read-only config was NOT cleared\n"
                "[WARN]  Split-brain: applications routing writes here AND to dead primary\n"
                "[INFO]  Read queries: serving normally. 6 connections active.\n"
                "[INFO]  promote_replica will NOT help — this node is already promoted but still read-only\n"
            ),
            "payment-service": (
                "[ERROR] ReadOnlyDatabaseException: cannot execute transaction — database is read-only\n"
                "[ERROR] Transaction rollback: payment_id=7f3a2d at postgres-replica:5432\n"
                "[WARN]  DB connection pool pointing to postgres-replica (failover auto-redirect)\n"
                "[WARN]  All write operations failing since 03:14Z — 89% transaction rollback rate\n"
                "[INFO]  Read operations (balance checks) succeeding — DB responds to reads\n"
                "[INFO]  No code changes to payment-service in 30 days\n"
            ),
            "order-service": (
                "[WARN] Order commit failed: ReadOnlyDatabaseException — retrying (attempt 14/30)\n"
                "[WARN] Write retry accumulation: 2,847 pending order commits in local retry queue\n"
                "[WARN] Memory usage: 71% — growing at +180MB/hr from retry buffer\n"
                "[INFO] Read operations (order history) still working\n"
                "[INFO] No deployment changes — DB infrastructure issue\n"
            ),
            "auth-service": (
                "[WARN] Session write failed: ReadOnlyDatabaseException at auth-db → postgres-replica\n"
                "[WARN] 23% of login attempts failing — cannot persist session token\n"
                "[INFO] JWT validation for existing tokens still working (reads only)\n"
                "[WARN] LOOKS like JWT key issue but is actually DB write rejection — check postgres\n"
            ),
            "inventory-service": (
                "[INFO] Reading stock levels from postgres-replica — queries succeeding but slow\n"
                "[INFO] No write ops by inventory-service — reads only\n"
                "[INFO] p99 query latency: 380ms (normal: 35ms) — replica under write-redirect pressure\n"
            ),
            "api-gateway":      "[INFO] Routing normally. No upstream errors at gateway level.\n",
            "nginx-lb":         "[INFO] All connections routing normally.\n",
            "redis-cache":      "[INFO] Memory 44%. Hit rate 93%. All operations normal.\n",
            "kafka-broker":     "[INFO] All partitions healthy. Consumer lag: 0ms.\n",
            "notification-service": "[INFO] Kafka consumer running. Message delivery nominal.\n",
        },

        "metrics": {
            "postgres-primary":     "cpu=0%  | memory=0%  | disk=100% | status=DOWN (OOM-killed)",
            "postgres-replica":     "cpu=78% | memory=71% | connections=89/100 | write_rejections=2847/min",
            "payment-service":      "cpu=22% | memory=48% | error_rate=89% | transaction_rollbacks=89%",
            "order-service":        "cpu=28% | memory=71% | error_rate=12% | retry_queue=2847",
            "auth-service":         "cpu=18% | memory=45% | error_rate=23% | session_write_fail=23%",
            "inventory-service":    "cpu=14% | memory=38% | error_rate=0%  | latency_p99=380ms",
            "api-gateway":          "cpu=19% | memory=44% | error_rate=0.4% | latency_p99=145ms",
            "nginx-lb":             "cpu=8%  | memory=26% | connections=14200 | error_rate=0%",
            "redis-cache":          "cpu=14% | memory=44% | hit_rate=93%   | eviction_rate=0",
            "kafka-broker":         "cpu=12% | memory=46% | partitions_with_leader=8/8 | consumer_lag=0",
            "notification-service": "cpu=9%  | memory=24% | error_rate=0%  | latency_p99=N/A",
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
    "services_queried": [],
    "resolved":         False,
    "resolution_claim": None,
    "episode_id":       None,
    "primary_fixed":    False,   # task6 state machine: postgres-primary restored?
}


def get_episode_state() -> dict:
    return _EPISODE_STATE


# ---------------------------------------------------------------------------
# Alert computation
# ---------------------------------------------------------------------------

def _compute_alerts(scenario: dict, health: dict) -> list:
    alerts = []
    service_alerts = scenario.get("service_alerts", {})

    for service, status in health.items():
        if status in ("down", "degraded"):
            svc_alerts = service_alerts.get(service, {}).get(status, [])
            if svc_alerts:
                alerts.extend(svc_alerts)
            else:
                if status == "down":
                    alerts.append(f"CRITICAL [{service}] Service DOWN — all health checks failing")
                else:
                    alerts.append(f"WARNING  [{service}] Service DEGRADED — elevated error rate")

    for alert in scenario.get("static_alerts", []):
        if alert not in alerts:
            alerts.append(alert)

    return list(dict.fromkeys(alerts))


# ---------------------------------------------------------------------------
# Cascade spreading
# ---------------------------------------------------------------------------

def _apply_cascade_events(
    scenario: dict,
    health: dict,
    circuit_breakers: list,
    step_count: int,
) -> list:
    root = scenario["root_cause_service"]
    if health.get(root) == "healthy":
        return []

    messages = []
    severity = {"healthy": 0, "degraded": 1, "down": 2}

    for event in scenario.get("cascade_events", []):
        target     = event["target"]
        threshold  = event["at_step"]
        new_status = event["new_status"]
        current    = health.get(target, "healthy")

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

    6 tasks covering real SRE failure modes: cache OOM, LB config rollback,
    message broker failure, DB replica corruption, auth memory leak,
    and database split-brain requiring multi-step remediation.
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
        _EPISODE_STATE["primary_fixed"]    = False

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

    def step(
        self,
        action: ProdWatchdogAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> ProdWatchdogObservation:
        task_id  = _EPISODE_STATE["task_id"]
        scenario = SCENARIOS[task_id]

        _EPISODE_STATE["step_count"] += 1
        step_count = _EPISODE_STATE["step_count"]
        max_steps  = scenario["max_steps"]

        health           = _EPISODE_STATE["service_health"]
        circuit_breakers = _EPISODE_STATE["circuit_breakers"]

        action_type = (action.action_type or "").strip().lower()
        service     = (action.service or "").strip().lower()

        # Snapshot health before action for potential-based shaping
        health_before = _compute_potential(health)

        result, reward_val, done = _process_action(
            action_type, service, scenario, health, circuit_breakers, step_count
        )

        # Potential-based shaping: dense reward signal every step
        health_after = _compute_potential(health)
        shaping = GAMMA * health_after - health_before
        reward_val += shaping

        # Per-step urgency penalty
        if not done:
            reward_val += STEP_PENALTY

        # Cascade spreading
        if not done:
            cascade_msgs = _apply_cascade_events(scenario, health, circuit_breakers, step_count)
            if cascade_msgs:
                result += "\n" + "\n".join(cascade_msgs)

        # Max steps
        if step_count >= max_steps and not done:
            done = True
            result += f"\n[TIMEOUT] Maximum steps ({max_steps}) reached. Episode ending."

        _EPISODE_STATE["episode_history"].append({
            "action": {
                "action_type": action_type,
                "service":     service,
                "parameters":  action.parameters,
            },
            "observation": {
                "step_count":         step_count,
                "service_health":     copy.deepcopy(health),
                "last_action_result": result,
            },
            "reward": round(reward_val, 4),
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
    action_type:      str,
    service:          str,
    scenario:         dict,
    health:           dict,
    circuit_breakers: list,
    step_count:       int,
) -> tuple:
    """Returns (result_text, reward_value, done). Shaping and step penalty applied in step()."""

    root_service = scenario["root_cause_service"]
    fix_action   = scenario["fix_action"]
    task_id      = _EPISODE_STATE["task_id"]

    max_steps = scenario["max_steps"]

    # ---- QUERY LOGS ----
    if action_type == "query_logs":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for query_logs.", -0.1, False

        base_log = scenario["logs"].get(service, "[INFO] No unusual log entries.\n")
        logs = _enrich_log(base_log, service, step_count, max_steps, health)

        if service in _EPISODE_STATE["services_queried"]:
            reward = 0.0
        elif service == root_service:
            reward = 0.2
            _EPISODE_STATE["services_queried"].append(service)
        else:
            reward = -0.05
            _EPISODE_STATE["services_queried"].append(service)

        return logs, reward, False

    # ---- CHECK METRICS ----
    elif action_type == "check_metrics":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for check_metrics.", -0.1, False

        base_metrics = scenario["metrics"].get(service, "cpu=5% | memory=20% | error_rate=0%")
        metrics = _enrich_metrics(base_metrics, service, step_count, max_steps, health)
        key = f"metrics:{service}"

        if key in _EPISODE_STATE["services_queried"]:
            reward = 0.0
        elif service == root_service:
            reward = 0.15
            _EPISODE_STATE["services_queried"].append(key)
        else:
            reward = -0.05
            _EPISODE_STATE["services_queried"].append(key)

        return metrics, reward, False

    # ---- RESTART SERVICE ----
    elif action_type == "restart_service":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for restart_service.", -0.1, False

        # Task6 special: restart payment-service only effective after postgres-primary restored
        if task_id == "task6" and service == "payment-service":
            if _EPISODE_STATE.get("primary_fixed"):
                if service in circuit_breakers:
                    circuit_breakers.remove(service)
                health[service] = "healthy"
                _heal_downstream(service, health, circuit_breakers)
                return (
                    "[OK] payment-service reconnected to restored postgres-primary. "
                    "DB writes succeeding. Dependents recovering.",
                    0.5, False,
                )
            else:
                return (
                    "[WARN] payment-service restarted but postgres-primary is still down "
                    "(replica still read-only). payment-service will immediately fail writes again. "
                    "Restore postgres-primary first via rollback_deploy.",
                    -0.05, False,
                )

        if service == root_service and fix_action == "restart_service":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            return (
                f"[OK] {service} restarted successfully. "
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
            # Task6: set flag so restart_service(payment-service) can complete the fix
            if task_id == "task6" and service == "postgres-primary":
                _EPISODE_STATE["primary_fixed"] = True
            return (
                f"[OK] {service} rolled back to last stable snapshot. "
                f"Health checks passing. Service restored.",
                0.5, False,
            )
        else:
            return (
                f"[INFO] No problematic recent deployment found on {service}. "
                f"Rollback not applicable to current incident.",
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
            return (
                f"[OK] Circuit breaker enabled on {service}. "
                f"Cascade contained. Blast radius limited.",
                0.3, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is healthy — circuit breaker not needed.", -0.1, False
        else:
            return (
                f"[OK] Circuit breaker enabled on {service}. "
                f"Traffic rerouted. Impact partially contained.",
                0.05, False,
            )

    # ---- SCALE UP ----
    elif action_type == "scale_up":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for scale_up.", -0.1, False

        if service == root_service and fix_action == "scale_up":
            health[service] = "healthy"
            _heal_downstream(service, health, circuit_breakers)
            return (
                f"[OK] {service} scaled to additional instances. "
                f"Resource pressure relieved. Health checks passing.",
                0.5, False,
            )
        elif health.get(service) == "healthy":
            return f"[WARN] {service} is healthy — scaling up unnecessarily.", -0.1, False
        else:
            return f"[OK] Scaled up {service}. Monitoring for recovery...", 0.0, False

    # ---- FLUSH CACHE ----
    elif action_type == "flush_cache":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for flush_cache.", -0.1, False
        if service != "redis-cache":
            return (
                f"[WARN] flush_cache is only applicable to redis-cache. "
                f"'{service}' has no cache to flush.",
                -0.05, False,
            )
        if task_id == "task1":
            return (
                "[PARTIAL] Cache flushed — but redis-cache OOM persists. Memory limit is still 100%. "
                "Cache will refill immediately from DB requests. "
                "Memory allocation must be increased: use scale_up(redis-cache).",
                0.05, False,
            )
        return "[OK] Cache flushed. No active cache-related incident.", 0.0, False

    # ---- PROMOTE REPLICA ----
    elif action_type == "promote_replica":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for promote_replica.", -0.1, False
        if service != "postgres-replica":
            return (
                f"[WARN] promote_replica only applies to postgres-replica. "
                f"'{service}' is not a database replica.",
                -0.05, False,
            )
        if task_id == "task6":
            return (
                "[WARN] postgres-replica is already promoted — Patroni failover ran automatically. "
                "Problem is that postgresql.conf still has 'default_transaction_read_only=on'. "
                "Promotion is not the fix. Restore postgres-primary via rollback_deploy instead.",
                0.0, False,
            )
        if task_id == "task4":
            return (
                "[WARN] postgres-replica has WAL corruption — promoting a corrupted replica will "
                "not help. Restart postgres-replica to clear bad WAL and re-sync from primary.",
                0.0, False,
            )
        return "[INFO] postgres-replica is healthy. No promotion needed.", 0.0, False

    # ---- REBALANCE PARTITIONS ----
    elif action_type == "rebalance_partitions":
        if not service or service not in health:
            return "[ERROR] Specify a valid service name for rebalance_partitions.", -0.1, False
        if service != "kafka-broker":
            return (
                f"[WARN] rebalance_partitions only applies to kafka-broker. "
                f"'{service}' has no Kafka partitions.",
                -0.05, False,
            )
        if task_id == "task3":
            return (
                "[WARN] Cannot rebalance — kafka-broker is fenced and not accepting connections. "
                "Partition rebalancing requires a running broker. "
                "Restart kafka-broker first to rejoin the cluster.",
                0.0, False,
            )
        return "[INFO] Partition rebalance triggered. No active Kafka incident.", 0.0, False

    # ---- DECLARE RESOLVED ----
    elif action_type == "declare_resolved":
        _EPISODE_STATE["resolution_claim"] = service or "unknown"
        root_fixed = health.get(root_service) == "healthy"

        unhealthy_non_cb = [
            svc for svc, h in health.items()
            if h != "healthy" and svc not in circuit_breakers
        ]
        all_resolved = root_fixed and len(unhealthy_non_cb) == 0

        if all_resolved:
            return "[RESOLVED] Incident fully resolved. Root fixed. Cascade contained.", 0.5, True
        elif root_fixed:
            return (
                "[PARTIAL] Root service fixed but some dependents still degraded. "
                "Consider checking remaining cascade effects.",
                0.3, True,
            )
        else:
            return (
                "[INCORRECT] Root cause not yet resolved. Incident still active.",
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
    """Propagate recovery from a healed service to dependents."""
    for dependent in HEAL_PROPAGATION.get(service, []):
        if dependent not in circuit_breakers and health.get(dependent) in ("degraded", "down"):
            health[dependent] = "healthy"
            _heal_downstream(dependent, health, circuit_breakers)


# ---------------------------------------------------------------------------
# Dynamic log/metrics enrichment — observations change with incident state
# ---------------------------------------------------------------------------

_URGENCY_PREFIXES = [
    "",                         # steps 1-3: no urgency header
    "[ESCALATING] ",            # steps 4-6
    "[HIGH URGENCY] ",          # steps 7-10
    "[CRITICAL — CASCADE ACTIVE] ",  # steps 11+
]

def _urgency_prefix(step: int, max_steps: int) -> str:
    frac = step / max(max_steps, 1)
    if frac < 0.25:
        return _URGENCY_PREFIXES[0]
    elif frac < 0.5:
        return _URGENCY_PREFIXES[1]
    elif frac < 0.75:
        return _URGENCY_PREFIXES[2]
    return _URGENCY_PREFIXES[3]


def _enrich_log(base_log: str, service: str, step: int, max_steps: int, health: dict) -> str:
    """
    Wrap a static log string with dynamic context:
    - Incident step counter (time pressure)
    - Cascade damage summary (which services have fallen since incident start)
    - Urgency level based on step fraction
    """
    prefix = _urgency_prefix(step, max_steps)
    header = f"=== {prefix}LOGS: {service} | incident step {step}/{max_steps} ===\n"

    # Show services that have cascaded down/degraded (snapshot of collateral damage)
    cascade_now = [
        svc for svc, st in health.items()
        if st in ("down", "degraded") and svc != service
    ]
    cascade_line = ""
    if cascade_now:
        cascade_line = f"[INCIDENT STATUS] Affected services: {', '.join(cascade_now)}\n"

    return header + cascade_line + base_log


def _enrich_metrics(base_metrics: str, service: str, step: int, max_steps: int, health: dict) -> str:
    """
    Wrap static metric string with dynamic context matching incident progression.
    Numeric values for cascaded services show degradation (not fake precision —
    just step-count offsets on top of base to show trend).
    """
    prefix = _urgency_prefix(step, max_steps)
    header = f"=== {prefix}METRICS: {service} | step {step}/{max_steps} ===\n"

    svc_status = health.get(service, "healthy")
    status_line = f"[SERVICE STATUS] {service}: {svc_status.upper()}\n"

    return header + status_line + base_metrics


# ---------------------------------------------------------------------------
# Task graders — deterministic, scores vary 0.10–1.00 based on agent quality
# ---------------------------------------------------------------------------

def grader_task1(episode_history: list) -> float:
    """
    Redis Cache OOM — Thundering Herd.
    A(0.25): diagnosed redis-cache
    B(0.15): first investigation was redis-cache
    C(0.25): correct fix scale_up(redis-cache) [0.10 partial for flush_cache]
    D(0.20): final state: redis-cache healthy + nginx-lb not down
    E(0.15): efficiency ≤4=1.0, ≥12=0.0
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    diagnosed_redis = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "redis-cache"
        for h in episode_history
    )

    first_inv = next((h for h in episode_history if h["action"]["action_type"] in investigate), None)
    correct_first = first_inv is not None and first_inv["action"]["service"] == "redis-cache"

    correct_fix = any(
        h["action"]["action_type"] == "scale_up" and h["action"]["service"] == "redis-cache"
        for h in episode_history
    )
    partial_fix = any(
        h["action"]["action_type"] == "flush_cache" and h["action"]["service"] == "redis-cache"
        for h in episode_history
    )
    fix_score = 0.25 if correct_fix else (0.10 if partial_fix else 0.0)

    final_health = episode_history[-1]["observation"]["service_health"]
    redis_fixed   = float(final_health.get("redis-cache") == "healthy")
    nginx_not_down = float(final_health.get("nginx-lb") != "down")
    state_score = 0.15 * redis_fixed + 0.05 * nginx_not_down

    steps = len(episode_history)
    efficiency = 1.0 if steps <= 4 else (0.0 if steps >= 12 else 1.0 - (steps - 4) / 8.0)

    score = (
        0.25 * float(diagnosed_redis)
        + 0.15 * float(correct_first)
        + fix_score
        + state_score
        + 0.15 * efficiency
    )
    return round(min(score, 1.0), 4)


def grader_task2(episode_history: list) -> float:
    """
    nginx-lb Bad Config Deploy.
    A(0.30): investigated nginx-lb
    B(0.20): first investigation was nginx-lb (1.0) or api-gateway (0.5) or other (0.0)
    C(0.30): rollback_deploy(nginx-lb)
    D(0.20): final state: nginx-lb + api-gateway both healthy
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    diagnosed_lb = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "nginx-lb"
        for h in episode_history
    )

    first_inv = next((h for h in episode_history if h["action"]["action_type"] in investigate), None)
    if first_inv is None:
        first_quality = 0.0
    elif first_inv["action"]["service"] == "nginx-lb":
        first_quality = 1.0
    elif first_inv["action"]["service"] == "api-gateway":
        first_quality = 0.5
    else:
        first_quality = 0.0

    correct_fix = any(
        h["action"]["action_type"] == "rollback_deploy" and h["action"]["service"] == "nginx-lb"
        for h in episode_history
    )

    final_health = episode_history[-1]["observation"]["service_health"]
    lb_ok  = float(final_health.get("nginx-lb") == "healthy")
    gw_ok  = float(final_health.get("api-gateway") == "healthy")
    state_score = 0.12 * lb_ok + 0.08 * gw_ok

    score = (
        0.30 * float(diagnosed_lb)
        + 0.20 * first_quality
        + 0.30 * float(correct_fix)
        + state_score
    )
    return round(min(score, 1.0), 4)


def grader_task3(episode_history: list) -> float:
    """
    Kafka Broker Partition Failure.
    A(0.20): diagnosed kafka-broker
    B(0.15): kafka found in first 2 investigations (1.0), 3rd-4th (0.5), later (0.2), never (0.0)
    C(0.30): restart_service(kafka-broker)
    D(0.20): final state: kafka healthy + order not down + notif not down
    E(0.15): efficiency ≤5=1.0, ≥15=0.0
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    diagnosed_kafka = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "kafka-broker"
        for h in episode_history
    )

    inv_steps = [h for h in episode_history if h["action"]["action_type"] in investigate]
    kafka_pos = next(
        (i for i, h in enumerate(inv_steps) if h["action"]["service"] == "kafka-broker"), None
    )
    if kafka_pos is None:
        inv_quality = 0.0
    elif kafka_pos <= 1:
        inv_quality = 1.0
    elif kafka_pos <= 3:
        inv_quality = 0.5
    else:
        inv_quality = 0.2

    correct_fix = any(
        h["action"]["action_type"] == "restart_service" and h["action"]["service"] == "kafka-broker"
        for h in episode_history
    )

    final_health = episode_history[-1]["observation"]["service_health"]
    kafka_ok = float(final_health.get("kafka-broker") == "healthy")
    order_ok = float(final_health.get("order-service") != "down")
    notif_ok = float(final_health.get("notification-service") != "down")
    state_score = 0.10 * kafka_ok + 0.06 * order_ok + 0.04 * notif_ok

    steps = len(episode_history)
    efficiency = 1.0 if steps <= 5 else (0.0 if steps >= 15 else 1.0 - (steps - 5) / 10.0)

    score = (
        0.20 * float(diagnosed_kafka)
        + 0.15 * inv_quality
        + 0.30 * float(correct_fix)
        + state_score
        + 0.15 * efficiency
    )
    return round(min(score, 1.0), 4)


def grader_task4(episode_history: list) -> float:
    """
    Postgres Replica WAL Corruption.
    A(0.25): diagnosed postgres-replica
    B(0.15): first investigation was postgres-replica or inventory-service (on right track)
    C(0.30): restart_service(postgres-replica)
    D(0.20): final state: postgres-replica + inventory-service both healthy
    E(0.10): efficiency ≤5=1.0, ≥15=0.0
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    diagnosed_replica = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "postgres-replica"
        for h in episode_history
    )

    first_inv = next((h for h in episode_history if h["action"]["action_type"] in investigate), None)
    right_track = (
        first_inv is not None
        and first_inv["action"]["service"] in ("postgres-replica", "inventory-service")
    )

    correct_fix = any(
        h["action"]["action_type"] == "restart_service" and h["action"]["service"] == "postgres-replica"
        for h in episode_history
    )

    final_health = episode_history[-1]["observation"]["service_health"]
    replica_ok = float(final_health.get("postgres-replica") == "healthy")
    inv_ok     = float(final_health.get("inventory-service") == "healthy")
    state_score = 0.12 * replica_ok + 0.08 * inv_ok

    steps = len(episode_history)
    efficiency = 1.0 if steps <= 5 else (0.0 if steps >= 15 else 1.0 - (steps - 5) / 10.0)

    score = (
        0.25 * float(diagnosed_replica)
        + 0.15 * float(right_track)
        + 0.30 * float(correct_fix)
        + state_score
        + 0.10 * efficiency
    )
    return round(min(score, 1.0), 4)


def grader_task5(episode_history: list) -> float:
    """
    Auth Service JWT Memory Leak + Cascade.
    A(0.25): diagnosed auth-service
    B(0.20): enable_circuit_breaker(api-gateway) applied
    C(0.30): restart_service(auth-service)
    D(0.15): final state: auth healthy + order not down + payment not down
    E(0.10): efficiency ≤6=1.0, ≥20=0.0
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    diagnosed_auth = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "auth-service"
        for h in episode_history
    )

    cb_gw = any(
        h["action"]["action_type"] == "enable_circuit_breaker" and h["action"]["service"] == "api-gateway"
        for h in episode_history
    )

    correct_fix = any(
        h["action"]["action_type"] == "restart_service" and h["action"]["service"] == "auth-service"
        for h in episode_history
    )

    final_health = episode_history[-1]["observation"]["service_health"]
    auth_ok    = float(final_health.get("auth-service") == "healthy")
    order_ok   = float(final_health.get("order-service") != "down")
    payment_ok = float(final_health.get("payment-service") != "down")
    state_score = 0.08 * auth_ok + 0.04 * order_ok + 0.03 * payment_ok

    steps = len(episode_history)
    efficiency = 1.0 if steps <= 6 else (0.0 if steps >= 20 else 1.0 - (steps - 6) / 14.0)

    score = (
        0.25 * float(diagnosed_auth)
        + 0.20 * float(cb_gw)
        + 0.30 * float(correct_fix)
        + state_score
        + 0.10 * efficiency
    )
    return round(min(score, 1.0), 4)


def grader_task6(episode_history: list) -> float:
    """
    Postgres Primary Disk Full + Split-Brain (3-step fix).
    A(0.20): investigated postgres-primary (1.0) or postgres-replica (0.5)
    B(0.20): enable_circuit_breaker(payment-service) applied
    C(0.25): rollback_deploy(postgres-primary) [partial: restart=0.10, promote_replica=0.05]
    D(0.15): restart_service(payment-service) AFTER postgres-primary was fixed
    E(0.10): final state: postgres-primary + payment-service healthy
    F(0.10): efficiency ≤7=1.0, ≥25=0.0
    """
    if not episode_history:
        return 0.0

    investigate = {"query_logs", "check_metrics"}

    inv_primary = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "postgres-primary"
        for h in episode_history
    )
    inv_replica = any(
        h["action"]["action_type"] in investigate and h["action"]["service"] == "postgres-replica"
        for h in episode_history
    )
    postgres_inv_score = 1.0 if inv_primary else (0.5 if inv_replica else 0.0)

    cb_payment = any(
        h["action"]["action_type"] == "enable_circuit_breaker" and h["action"]["service"] == "payment-service"
        for h in episode_history
    )
    cb_other = any(
        h["action"]["action_type"] == "enable_circuit_breaker" and h["action"]["service"] != "payment-service"
        for h in episode_history
    )
    cb_score = 1.0 if cb_payment else (0.05 if cb_other else 0.0)

    correct_primary = any(
        h["action"]["action_type"] == "rollback_deploy" and h["action"]["service"] == "postgres-primary"
        for h in episode_history
    )
    restart_primary = any(
        h["action"]["action_type"] == "restart_service" and h["action"]["service"] == "postgres-primary"
        for h in episode_history
    )
    promote_wrong = any(
        h["action"]["action_type"] == "promote_replica" and h["action"]["service"] == "postgres-replica"
        for h in episode_history
    )
    primary_fix_score = (
        1.0 if correct_primary
        else (0.10 if restart_primary else (0.05 if promote_wrong else 0.0))
    )

    # Secondary fix: restart_service(payment-service) must come AFTER rollback_deploy(postgres-primary)
    postgres_fix_step = next(
        (i for i, h in enumerate(episode_history)
         if h["action"]["action_type"] == "rollback_deploy"
         and h["action"]["service"] == "postgres-primary"),
        None,
    )
    secondary_fix = False
    if postgres_fix_step is not None:
        secondary_fix = any(
            h["action"]["action_type"] == "restart_service"
            and h["action"]["service"] == "payment-service"
            for h in episode_history[postgres_fix_step + 1:]
        )

    final_health = episode_history[-1]["observation"]["service_health"]
    pg_ok  = float(final_health.get("postgres-primary") == "healthy")
    pay_ok = float(final_health.get("payment-service") == "healthy")
    state_score = 0.06 * pg_ok + 0.04 * pay_ok

    steps = len(episode_history)
    efficiency = 1.0 if steps <= 7 else (0.0 if steps >= 25 else 1.0 - (steps - 7) / 18.0)

    score = (
        0.20 * postgres_inv_score
        + 0.20 * cb_score
        + 0.25 * primary_fix_score
        + 0.15 * float(secondary_fix)
        + state_score
        + 0.10 * efficiency
    )
    return round(min(score, 1.0), 4)


TASK_GRADERS = {
    "task1": grader_task1,
    "task2": grader_task2,
    "task3": grader_task3,
    "task4": grader_task4,
    "task5": grader_task5,
    "task6": grader_task6,
}
