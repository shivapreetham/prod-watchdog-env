"""
Data models for the ProdWatchdog Environment.

Typed Action and Observation classes following the OpenEnv spec.
"""

from typing import Dict, List, Optional, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


VALID_ACTION_TYPES = [
    "query_logs",
    "check_metrics",
    "restart_service",
    "rollback_deploy",
    "enable_circuit_breaker",
    "scale_up",
    "declare_resolved",
]

VALID_SERVICES = [
    "api-gateway",
    "auth-service",
    "order-service",
    "payment-service",
    "inventory-service",
    "notification-service",
]


class ProdWatchdogAction(Action):
    """
    Action for the ProdWatchdog environment.

    The agent selects an action type and optionally a target service.

    action_type options:
      - query_logs: Read logs for a service to find error patterns
      - check_metrics: Check CPU/memory/latency metrics for a service
      - restart_service: Restart a service (fixes DB leaks, memory issues)
      - rollback_deploy: Rollback the latest deployment (fixes bad deploys)
      - enable_circuit_breaker: Isolate a service to stop cascade propagation
      - scale_up: Add more instances (fixes CPU spikes, high load)
      - declare_resolved: End the episode and declare root cause fixed
    """

    action_type: str = Field(
        ...,
        description=f"Type of action to take. One of: {VALID_ACTION_TYPES}",
    )
    service: Optional[str] = Field(
        default=None,
        description=f"Target service. One of: {VALID_SERVICES}",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional extra parameters (e.g. metric name for check_metrics)",
    )


class ProdWatchdogObservation(Observation):
    """
    Observation from the ProdWatchdog environment.

    What the on-call engineer (agent) sees after each action.
    """

    alerts: List[str] = Field(
        default_factory=list,
        description="Active PagerDuty-style alerts firing right now",
    )
    service_health: Dict[str, str] = Field(
        default_factory=dict,
        description="Current health status of each service: healthy | degraded | down",
    )
    last_action_result: str = Field(
        default="",
        description="Output/result of the last action taken (logs, metrics, command output)",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken in this episode",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="List of valid action_type values",
    )
