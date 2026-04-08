import gradio as gr
import json
import time

from inference import (
    run_task_stream,
    create_llm_client,
    create_groq_client,
    MODEL_NAME,
    GROQ_MODEL,
)
from client import ProdWatchdogClient

BASE_URL = "http://localhost:7860"


# -----------------------------
# RUN AGENT (STREAM)
# -----------------------------
def run_agent(task_id):
    logs_history = ""
    actions_history = ""

    # create LLM clients
    llm_client = create_llm_client()
    groq_client = create_groq_client()

    primary = llm_client if llm_client else groq_client
    model = MODEL_NAME if llm_client else GROQ_MODEL

    with ProdWatchdogClient(BASE_URL) as env_client:

        for data in run_task_stream(
            task_id,
            env_client,
            primary,
            model,
            None,
            None,
        ):

            step = data.get("step", 0)
            action = data.get("action", {})
            logs = data.get("logs", "")
            health = data.get("health") or {}
            alerts = data.get("alerts", [])
            reward = data.get("reward", 0.0)

            action_type = action.get("action_type", "unknown")
            service = action.get("service", "unknown")

            # -----------------------------
            # BUILD UI OUTPUT
            # -----------------------------

            # Logs
            logs_history += f"\n\n[Step {step}]\n{logs}"

            # Actions
            actions_history += (
                f"\nStep {step} → {action_type}({service}) | Reward: {reward:+.4f}"
            )

            # Health rendering
            if health:
                health_text = "\n".join(
                    f"{'🟢' if v=='healthy' else '🟡' if v=='degraded' else '🔴'} {k}: {v}"
                    for k, v in health.items()
                )
            else:
                health_text = "No health data available"

            # LLM Input (agent view)
            llm_input = f"""
ALERTS:
{json.dumps(alerts, indent=2)}

SERVICE HEALTH:
{json.dumps(health, indent=2)}

LATEST LOGS:
{logs}
"""

            # -----------------------------
            # STREAM UPDATE
            # -----------------------------
            yield (
                logs_history.strip(),
                actions_history.strip(),
                llm_input.strip(),
                f"Step: {step}",
                f"Step Reward: {reward:+.4f}",
                health_text,
                f"Running {task_id}... Step {step}",
            )

            # -----------------------------
            # FINAL STEP HANDLING 
            # -----------------------------
            if data.get("done"):

                #  small delay to ensure backend state sync
                time.sleep(0.5)

                #  get real score from backend
                try:
                    score = env_client.get_grader_score(task_id)
                except Exception as e:
                    print("Grader error:", e)
                    score = 0.0

                #  get actual backend step count
                try:
                    state = env_client.get_state()
                    steps = state.get("step_count", step)
                except:
                    steps = step

                final_status = f" Finished {task_id} | Score: {score:.3f} | Steps: {steps}"

                yield (
                    logs_history.strip(),
                    actions_history.strip(),
                    llm_input.strip(),
                    f"Final Step: {steps}",
                    f"Final Reward: {reward:+.4f} | Score: {score:.3f}",
                    health_text,
                    final_status,
                )

                break

            time.sleep(0.5)


# -----------------------------
# RESET UI
# -----------------------------
def reset_ui():
    return "", "", "", "Step: 0", "Reward: 0", "", "Idle"


# -----------------------------
# UI LAYOUT
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 🤖 ProdWatchdog Live Incident Dashboard")

    # Task selector
    task_selector = gr.Dropdown(
        ["task1", "task2", "task3", "task4", "task5", "task6"],
        value="task1",
        label="Select Task",
    )

    # Row 1 → Logs + Health
    with gr.Row():
        logs_box = gr.Textbox(label="📜 Logs (Streaming)", lines=20)
        health_box = gr.Textbox(label="🏥 Service Health", lines=20)

    # Row 2 → Actions + LLM input
    with gr.Row():
        actions_box = gr.Textbox(label="🤖 Agent Actions", lines=10)
        llm_box = gr.Textbox(label="🧠 LLM Input (Agent View)", lines=10)

    # Row 3 → Step + Reward + Status
    with gr.Row():
        step_box = gr.Textbox(label="⏱ Step")
        reward_box = gr.Textbox(label="💰 Reward")
        status_box = gr.Textbox(label="Status")

    # Buttons
    with gr.Row():
        start_btn = gr.Button("▶ Start Agent")
        reset_btn = gr.Button("🔁 Reset")

    # Start action
    start_btn.click(
        run_agent,
        inputs=[task_selector],
        outputs=[
            logs_box,
            actions_box,
            llm_box,
            step_box,
            reward_box,
            health_box,
            status_box,
        ],
    )

    # Reset action
    reset_btn.click(
        reset_ui,
        outputs=[
            logs_box,
            actions_box,
            llm_box,
            step_box,
            reward_box,
            health_box,
            status_box,
        ],
    )


# -----------------------------
# LAUNCH
# -----------------------------
demo.launch()