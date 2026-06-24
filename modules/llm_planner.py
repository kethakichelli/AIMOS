"""
AIMOS v2 — LLM Planner
Converts natural language goals into AIMOS action plans.
Uses Ollama (local LLM) — no API key, no internet needed.

Flow:
  User: "My Python script is slow"
  LLM  → ["boost_cpu_priority('python3')",
           "set_io_realtime('python3')",
           "prefetch_memory_pages()"]
  AIMOS enforcer → executes on real kernel
"""

import json
import os
import sys
import logging
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

logger = logging.getLogger(__name__)

# Available AIMOS actions the LLM can call
AVAILABLE_ACTIONS = {
    "boost_cpu_priority": {
        "description": "Boost CPU priority of a process (renice to -5)",
        "params": ["process_name_or_pid"],
        "example": "boost_cpu_priority('python3')"
    },
    "lower_cpu_priority": {
        "description": "Lower CPU priority of a process (renice to +10)",
        "params": ["process_name_or_pid"],
        "example": "lower_cpu_priority('chrome')"
    },
    "set_io_realtime": {
        "description": "Set I/O scheduling to realtime class for a process",
        "params": ["process_name_or_pid"],
        "example": "set_io_realtime('python3')"
    },
    "set_io_idle": {
        "description": "Set I/O scheduling to idle class (lowest priority)",
        "params": ["process_name_or_pid"],
        "example": "set_io_idle('backup_process')"
    },
    "set_cpu_governor": {
        "description": "Set CPU frequency governor",
        "params": ["governor: performance|powersave|ondemand|schedutil"],
        "example": "set_cpu_governor('performance')"
    },
    "kill_anomalous_processes": {
        "description": "Kill or throttle processes flagged as anomalous by the AI detector",
        "params": [],
        "example": "kill_anomalous_processes()"
    },
    "get_system_state": {
        "description": "Get current system metrics (CPU, memory, disk, top processes)",
        "params": [],
        "example": "get_system_state()"
    },
    "optimize_for_ml": {
        "description": "Full ML training optimization: boost CPU+IO, set performance governor",
        "params": [],
        "example": "optimize_for_ml()"
    },
    "optimize_for_battery": {
        "description": "Power saving mode: powersave governor, lower background processes",
        "params": [],
        "example": "optimize_for_battery()"
    },
    "optimize_for_web_server": {
        "description": "Web server mode: balanced CPU, realtime IO for server processes",
        "params": [],
        "example": "optimize_for_web_server()"
    },
    "show_anomalies": {
        "description": "Show currently detected anomalous processes",
        "params": [],
        "example": "show_anomalies()"
    },
    "show_deadlock_risk": {
        "description": "Show current deadlock risk score from the RF predictor",
        "params": [],
        "example": "show_deadlock_risk()"
    },
    "show_memory_stats": {
        "description": "Show page fault rates and LSTM memory prediction accuracy",
        "params": [],
        "example": "show_memory_stats()"
    },
    "show_disk_clusters": {
        "description": "Show K-Means disk access clusters and seek time savings",
        "params": [],
        "example": "show_disk_clusters()"
    },
    "set_cpu_limit": {
        "description": "Limit a process to a CPU percentage via cgroups",
        "params": ["process_name_or_pid", "percent: 1-100"],
        "example": "set_cpu_limit('stress-ng', 20)"
    },
    "free_memory": {
        "description": "Drop caches and free unused memory",
        "params": [],
        "example": "free_memory()"
    },
}

SYSTEM_PROMPT = """You are the AIMOS AI Operating System Brain.
AIMOS is an AI-driven OS management layer running on Linux.
You have access to the following OS control actions:

{actions}

When the user gives you a goal, respond ONLY with a JSON object in this exact format:
{{
  "interpretation": "one sentence explaining what you understood",
  "actions": ["action1('param')", "action2()", "action3('param')"],
  "explanation": "one sentence explaining what these actions will do",
  "estimated_effect": "one sentence on the expected outcome"
}}

Rules:
- Only use actions from the list above
- Use 1-5 actions per response
- Parameters must match the available params
- Never invent actions that are not in the list
- If the goal is unclear, use get_system_state() first
- If the goal involves safety (deleting files, killing important processes), include show_anomalies() first
- Respond ONLY with the JSON object, no other text
"""


class AIMOSLLMPlanner:

    def __init__(self, model="llama3.2"):
        self.model = model
        self.conversation_history = []
        self._check_ollama()

    def _check_ollama(self):
        try:
            import ollama
            models = ollama.list()
            available = [m.model for m in models.models]
            if not any(self.model in m for m in available):
                logger.warning(
                    f"Model {self.model} not found. "
                    f"Run: ollama pull {self.model}"
                )
            else:
                logger.info(f"LLM Planner ready — model: {self.model}")
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}")

    def _build_actions_description(self):
        lines = []
        for name, info in AVAILABLE_ACTIONS.items():
            lines.append(
                f"- {name}({', '.join(info['params'])}): "
                f"{info['description']}"
            )
        return "\n".join(lines)

    def plan(self, user_goal: str) -> dict:
        """
        Convert a natural language goal into an AIMOS action plan.
        Returns dict with interpretation, actions, explanation.
        """
        import ollama

        system = SYSTEM_PROMPT.format(
            actions=self._build_actions_description()
        )

        self.conversation_history.append({
            "role": "user",
            "content": user_goal
        })

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *self.conversation_history
                ]
            )

            content = response.message.content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            plan = json.loads(content)

            self.conversation_history.append({
                "role": "assistant",
                "content": content
            })

            return {
                "success": True,
                "goal": user_goal,
                "interpretation": plan.get("interpretation", ""),
                "actions": plan.get("actions", []),
                "explanation": plan.get("explanation", ""),
                "estimated_effect": plan.get("estimated_effect", "")
            }

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return {
                "success": False,
                "goal": user_goal,
                "error": "LLM response was not valid JSON",
                "raw": content if 'content' in dir() else ""
            }
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            return {
                "success": False,
                "goal": user_goal,
                "error": str(e)
            }

    def clear_history(self):
        self.conversation_history = []


if __name__ == "__main__":
    planner = AIMOSLLMPlanner()
    test_goals = [
        "My Python script is running too slow",
        "I want to save battery power",
        "Something suspicious is using too much CPU",
    ]
    for goal in test_goals:
        print(f"\nGoal: {goal}")
        print("-" * 40)
        result = planner.plan(goal)
        if result["success"]:
            print(f"Interpretation : {result['interpretation']}")
            print(f"Actions        : {result['actions']}")
            print(f"Explanation    : {result['explanation']}")
            print(f"Expected effect: {result['estimated_effect']}")
        else:
            print(f"Error: {result['error']}")
