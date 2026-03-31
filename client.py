"""
ProdWatchdog HTTP Client.

Clean Python interface for interacting with the ProdWatchdog environment server.
Used by inference.py and usable directly for external agent integration.

Usage:
    from client import ProdWatchdogClient

    with ProdWatchdogClient("http://localhost:7860") as env:
        obs, done, reward = env.reset("task1")
        while not done:
            obs, done, reward = env.step("query_logs", "order-service")
        score = env.get_grader_score("task1")
        print(f"Score: {score}")
"""

import httpx
from typing import Optional, Tuple


class ProdWatchdogClient:
    """
    HTTP client wrapper for the ProdWatchdog environment server.

    Handles all HTTP communication, response parsing, and error handling.
    Returns clean Python dicts/tuples — no raw httpx responses exposed.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Core OpenEnv endpoints
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Tuple[dict, bool, float]:
        """
        Reset the environment for a given task.

        Returns:
            (observation, done, reward) tuple.
            observation: dict with alerts, service_health, last_action_result, step_count
        """
        r = self._http.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        r.raise_for_status()
        raw = r.json()
        obs    = raw.get("observation", raw)
        done   = raw.get("done", obs.get("done", False))
        reward = raw.get("reward", obs.get("reward", 0.0))
        return obs, done, reward

    def step(
        self,
        action_type: str,
        service: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> Tuple[dict, bool, float]:
        """
        Execute one action in the environment.

        Args:
            action_type: One of query_logs, check_metrics, restart_service,
                         rollback_deploy, enable_circuit_breaker, scale_up, declare_resolved
            service:     Target service name (required for most actions)
            parameters:  Optional extra parameters dict

        Returns:
            (observation, done, reward) tuple.
        """
        action: dict = {"action_type": action_type}
        if service:
            action["service"] = service
        if parameters:
            action["parameters"] = parameters

        r = self._http.post(
            f"{self.base_url}/step",
            json={"action": action},
        )
        r.raise_for_status()
        raw    = r.json()
        obs    = raw.get("observation", raw)
        done   = raw.get("done", obs.get("done", False))
        reward = raw.get("reward", obs.get("reward", 0.0))
        return obs, done, reward

    def get_state(self) -> dict:
        """Get current environment state (episode_id, step_count)."""
        r = self._http.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Hackathon-required custom endpoints
    # ------------------------------------------------------------------

    def get_tasks(self) -> list:
        """
        Get list of available tasks with descriptions and action schema.

        Returns:
            List of task dicts: {id, name, difficulty, description, max_steps}
        """
        r = self._http.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json().get("tasks", [])

    def get_grader_score(self, task_id: str) -> float:
        """
        Get the grader score for a completed episode.

        Must be called after reset + step sequence completes.

        Returns:
            float score between 0.0 and 1.0
        """
        r = self._http.post(
            f"{self.base_url}/grader",
            params={"task_id": task_id},
        )
        r.raise_for_status()
        return float(r.json().get("score", 0.0))

    def run_baseline(self, timeout: float = 60.0) -> dict:
        """
        Trigger the rule-based baseline agent on all 3 tasks.

        Returns:
            dict with task1, task2, task3, average scores
        """
        r = self._http.post(f"{self.base_url}/baseline", timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check if the server is healthy."""
        r = self._http.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def format_observation(self, obs: dict) -> str:
        """
        Format an observation dict into a human-readable string for LLM prompts.

        Args:
            obs: Observation dict from reset() or step()

        Returns:
            Formatted string ready to use as LLM user message
        """
        health      = obs.get("service_health", {})
        alerts      = obs.get("alerts", [])
        last_result = obs.get("last_action_result", "")
        step        = obs.get("step_count", 0)

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

    def close(self):
        """Close the underlying HTTP connection."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
