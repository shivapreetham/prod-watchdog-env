"""
FastAPI application for the ProdWatchdog Environment.

Exposes the ProdWatchdog incident response environment over HTTP, WebSocket,
and MCP — compatible with the OpenEnv spec.

Standard endpoints (via openenv create_app):
    POST /reset       Reset environment (pass task_id in body)
    POST /step        Execute an action
    GET  /state       Get current environment state
    GET  /health      Health check
    GET  /schema      Action/observation/state schemas
    POST /mcp         MCP JSON-RPC endpoint
    WS   /ws          WebSocket for stateful sessions

Additional hackathon endpoints:
    GET  /tasks       List tasks with descriptions and action schema
    POST /grader      Run grader for a completed episode
    POST /baseline    Run all 3 tasks with a rule-based baseline agent

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. Install with: pip install openenv-core") from e

try:
    from ..models import ProdWatchdogAction, ProdWatchdogObservation, VALID_SERVICES
    from .environment import (
        ProdWatchdogEnvironment,
        TASK_GRADERS,
        SCENARIOS,
        get_episode_state,
        _EPISODE_STATE,
    )
except ImportError:
    from models import ProdWatchdogAction, ProdWatchdogObservation, VALID_SERVICES
    from server.environment import (
        ProdWatchdogEnvironment,
        TASK_GRADERS,
        SCENARIOS,
        get_episode_state,
        _EPISODE_STATE,
    )


# ---------------------------------------------------------------------------
# Core app (provides /reset /step /state /health /schema /mcp /ws)
# ---------------------------------------------------------------------------

app = create_app(
    ProdWatchdogEnvironment,
    ProdWatchdogAction,
    ProdWatchdogObservation,
    env_name="prod-watchdog",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Custom endpoints router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/tasks")
def list_tasks():
    """
    Returns the list of available tasks with their descriptions,
    difficulty levels, and the action schema (fields required for a step).
    """
    tasks = []
    for task_id, scenario in SCENARIOS.items():
        tasks.append({
            "id": task_id,
            "name": {
                "task1": "Single Service Outage",
                "task2": "Cascading Failure Diagnosis",
                "task3": "Multi-Service Cascade with Noise",
            }.get(task_id, task_id),
            "difficulty": {
                "task1": "easy",
                "task2": "medium",
                "task3": "hard",
            }.get(task_id, "unknown"),
            "description": scenario["description"],
            "max_steps": scenario["max_steps"],
            "grader": f"grader_{task_id}",
        })

    action_schema = {
        "action_type": {
            "type": "string",
            "description": "Action to take",
            "enum": [
                "query_logs",
                "check_metrics",
                "restart_service",
                "rollback_deploy",
                "enable_circuit_breaker",
                "scale_up",
                "declare_resolved",
            ],
        },
        "service": {
            "type": "string",
            "description": "Target service (required for most actions)",
            "enum": VALID_SERVICES,
        },
        "parameters": {
            "type": "object",
            "description": "Optional extra parameters",
        },
    }

    return JSONResponse({
        "tasks": tasks,
        "action_schema": action_schema,
        "services": VALID_SERVICES,
    })


@router.post("/grader")
def run_grader(task_id: str = None):
    """
    Run the grader for the current (or specified) task's completed episode.
    Returns a score between 0.0 and 1.0.

    Must be called after an episode completes (after reset + step sequence).
    """
    state = get_episode_state()
    tid = task_id or state.get("task_id", "task1")

    if tid not in TASK_GRADERS:
        return JSONResponse({"error": f"Unknown task_id: {tid}"}, status_code=400)

    history = state.get("episode_history", [])
    grader_fn = TASK_GRADERS[tid]
    score = grader_fn(history)

    return JSONResponse({
        "task_id": tid,
        "score": score,
        "steps_taken": len(history),
        "episode_id": state.get("episode_id"),
    })


@router.post("/baseline")
def run_baseline():
    """
    Run a rule-based baseline agent against all 3 tasks.
    Returns deterministic baseline scores for automated evaluation.

    This uses a hardcoded expert policy that knows the correct actions,
    establishing the upper-bound baseline for each task.
    """
    from server.environment import (
        ProdWatchdogEnvironment,
        _EPISODE_STATE,
        SCENARIOS,
    )

    results = {}

    for task_id in ["task1", "task2", "task3"]:
        env = ProdWatchdogEnvironment()
        env.reset(task_id=task_id)

        scenario = SCENARIOS[task_id]
        root_service = scenario["root_cause_service"]
        fix_action = scenario["fix_action"]
        cb_service = scenario.get("requires_circuit_breaker")

        # Expert baseline policy: investigate, fix, declare
        expert_actions = [
            ProdWatchdogAction(action_type="query_logs", service=root_service),
            ProdWatchdogAction(action_type="check_metrics", service=root_service),
        ]
        if cb_service:
            expert_actions.append(
                ProdWatchdogAction(action_type="enable_circuit_breaker", service=cb_service)
            )
        expert_actions.append(
            ProdWatchdogAction(action_type=fix_action, service=root_service)
        )
        expert_actions.append(
            ProdWatchdogAction(action_type="declare_resolved", service=root_service)
        )

        for action in expert_actions:
            obs = env.step(action)
            if obs.done:
                break

        grader_fn = TASK_GRADERS[task_id]
        score = grader_fn(_EPISODE_STATE["episode_history"])
        results[task_id] = score

    average = round(sum(results.values()) / len(results), 4)
    results["average"] = average

    return JSONResponse(results)


# Register custom router
app.include_router(router)


# ---------------------------------------------------------------------------
# Entry point for uv run / python -m
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for local execution and HF Spaces deployment."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
