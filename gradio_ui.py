"""
Gradio UI for ProdWatchdog Environment.

Provides an interactive web interface for the ProdWatchdog incident response environment.
Users can select tasks, view alerts, take actions, and see real-time feedback.
"""

import gradio as gr
import json
from typing import Dict, Any, List, Tuple
import time

from client import ProdWatchdogClient
from models import VALID_SERVICES, VALID_ACTION_TYPES


class ProdWatchdogUI:
    """Gradio-based UI for the ProdWatchdog environment."""

    def __init__(self, server_url: str = "http://localhost:7860"):
        self.client = ProdWatchdogClient(server_url)
        self.current_task = None
        self.episode_history = []
        self.step_count = 0

    def get_tasks(self) -> List[str]:
        """Get available tasks from the server."""
        try:
            response = self.client._http.get(f"{self.client.base_url}/tasks")
            response.raise_for_status()
            tasks_data = response.json()
            return [task["id"] for task in tasks_data.get("tasks", [])]
        except Exception as e:
            print(f"Error fetching tasks: {e}")
            return ["task1", "task2", "task3"]  # fallback

    def reset_environment(self, task_id: str) -> Tuple[str, str, str, str]:
        """Reset the environment for a new task."""
        try:
            obs, done, reward = self.client.reset(task_id)
            self.current_task = task_id
            self.episode_history = []
            self.step_count = 0

            alerts_text = "\n".join(obs.get("alerts", []))
            health_text = "\n".join([
                f"{service}: {status}"
                for service, status in obs.get("service_health", {}).items()
            ])
            last_action = obs.get("last_action_result", "Environment reset")
            step_info = f"Step: {self.step_count} | Task: {task_id}"

            return alerts_text, health_text, last_action, step_info

        except Exception as e:
            error_msg = f"Failed to reset environment: {str(e)}"
            return error_msg, "", "", "Error"

    def take_action(self, action_type: str, service: str) -> Tuple[str, str, str, str]:
        """Execute an action and update the UI."""
        if not self.current_task:
            return "Please select a task first", "", "", ""

        try:
            # Prepare action
            action = {"action_type": action_type}
            if service and service != "none":
                action["service"] = service

            # Execute action
            obs, done, reward = self.client.step(action["action_type"], action.get("service"))

            # Update history
            self.step_count += 1
            action_str = f"{action_type}({service})" if service and service != "none" else action_type
            self.episode_history.append(f"Step {self.step_count}: {action_str} → Reward: {reward:.2f}")

            # Format output
            alerts_text = "\n".join(obs.get("alerts", []))
            health_text = "\n".join([
                f"{service}: {status}"
                for service, status in obs.get("service_health", {}).items()
            ])
            last_action = obs.get("last_action_result", f"Action executed: {action_str}")
            step_info = f"Step: {self.step_count} | Task: {self.current_task} | Reward: {reward:.2f}"

            if done:
                step_info += " | EPISODE COMPLETE"
                # Try to get score
                try:
                    score = self.client.get_grader_score(self.current_task)
                    step_info += f" | Score: {score:.3f}"
                except:
                    pass

            return alerts_text, health_text, last_action, step_info

        except Exception as e:
            error_msg = f"Action failed: {str(e)}"
            return error_msg, "", "", f"Step: {self.step_count} | ERROR"

    def get_history(self) -> str:
        """Get the episode history."""
        if not self.episode_history:
            return "No actions taken yet"
        return "\n".join(self.episode_history)

    def get_available_actions(self) -> List[str]:
        """Get list of available action types."""
        return VALID_ACTION_TYPES

    def get_available_services(self) -> List[str]:
        """Get list of available services."""
        return ["none"] + VALID_SERVICES


def create_gradio_ui():
    """Create and launch the Gradio UI."""
    ui = ProdWatchdogUI()

    # Get initial data
    tasks = ui.get_tasks()
    actions = ui.get_available_actions()
    services = ui.get_available_services()

    with gr.Blocks(title="ProdWatchdog - SRE Incident Response") as demo:
        gr.Markdown("# 🚨 ProdWatchdog - SRE Incident Response Environment")
        gr.Markdown("Simulate real on-call SRE work by diagnosing and fixing microservice failures.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Task Selection")
                task_dropdown = gr.Dropdown(
                    choices=tasks,
                    label="Select Incident Task",
                    value=tasks[0] if tasks else "task1",
                    info="Choose which production incident to respond to"
                )
                reset_btn = gr.Button("🔄 Start New Incident", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Current Status")
                step_info = gr.Textbox(
                    label="Episode Status",
                    value="Ready to start",
                    interactive=False
                )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Active Alerts")
                alerts_display = gr.Textbox(
                    label="Production Alerts",
                    lines=5,
                    interactive=False,
                    placeholder="No active alerts"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Service Health")
                health_display = gr.Textbox(
                    label="Service Status",
                    lines=6,
                    interactive=False,
                    placeholder="No services monitored"
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Take Action")
                with gr.Row():
                    action_dropdown = gr.Dropdown(
                        choices=actions,
                        label="Action Type",
                        value="query_logs",
                        info="What action do you want to take?"
                    )
                    service_dropdown = gr.Dropdown(
                        choices=services,
                        label="Target Service",
                        value="api-gateway",
                        info="Which service to target? (not needed for declare_resolved)"
                    )
                action_btn = gr.Button("⚡ Execute Action", variant="secondary")

            with gr.Column(scale=1):
                gr.Markdown("### Action Result")
                action_result = gr.Textbox(
                    label="Last Action Output",
                    lines=4,
                    interactive=False,
                    placeholder="Action results will appear here"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Episode History")
                history_display = gr.Textbox(
                    label="Action History",
                    lines=8,
                    interactive=False,
                    placeholder="Your action history will appear here"
                )
                refresh_history_btn = gr.Button("🔄 Refresh History", size="sm")

        # Event handlers
        reset_btn.click(
            fn=ui.reset_environment,
            inputs=[task_dropdown],
            outputs=[alerts_display, health_display, action_result, step_info]
        )

        action_btn.click(
            fn=ui.take_action,
            inputs=[action_dropdown, service_dropdown],
            outputs=[alerts_display, health_display, action_result, step_info]
        )

        refresh_history_btn.click(
            fn=ui.get_history,
            inputs=[],
            outputs=[history_display]
        )

        # Auto-refresh history when actions are taken
        reset_btn.click(fn=ui.get_history, inputs=[], outputs=[history_display])
        action_btn.click(fn=ui.get_history, inputs=[], outputs=[history_display])

        gr.Markdown("""
        ### How to Use:
        1. **Select a Task**: Choose which production incident to respond to
        2. **Start Incident**: Click "Start New Incident" to begin
        3. **Investigate**: Use "query_logs" and "check_metrics" to gather information
        4. **Take Action**: Apply fixes like "rollback_deploy", "scale_up", or "enable_circuit_breaker"
        5. **Resolve**: When ready, use "declare_resolved" to end the incident

        ### Action Guide:
        - **query_logs**: Read recent logs for a service to find error patterns
        - **check_metrics**: Check CPU/memory/error metrics for a service
        - **restart_service**: Restart a service (fixes DB leaks, memory issues)
        - **rollback_deploy**: Rollback last deployment (fixes bad deploys)
        - **enable_circuit_breaker**: Isolate service to stop cascade propagation
        - **scale_up**: Add instances (fixes CPU spikes, high load)
        - **declare_resolved**: End episode when root cause is fixed
        """)

    return demo


def launch_gradio_ui():
    """Launch the Gradio UI."""
    demo = create_gradio_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        theme=gr.themes.Soft(),
        share=False,  # Set to True for public sharing
        show_error=True
    )


if __name__ == "__main__":
    launch_gradio_ui()