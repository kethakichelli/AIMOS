"""
AIMOS v2 — Agentic OS Terminal
Natural language interface for AIMOS.
Type goals, get OS-level results.

Usage:
  python aos_terminal.py
"""

import os
import sys
import logging
sys.path.insert(0, os.path.expanduser("~/AIMOS"))

logging.basicConfig(level=logging.WARNING)

from modules.llm_planner   import AIMOSLLMPlanner
from modules.action_executor import AIMOSActionExecutor

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.table   import Table
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def print_banner():
    banner = """
  █████╗  ██████╗ ███████╗
 ██╔══██╗██╔═══██╗██╔════╝
 ███████║██║   ██║███████╗
 ██╔══██║██║   ██║╚════██║
 ██║  ██║╚██████╔╝███████║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
 Agentic Operating System v2
 Type a goal. Press Enter. AIMOS acts.
 Type 'help' for examples. Type 'exit' to quit.
    """
    print(banner)


def print_examples():
    examples = [
        "My Python script is running too slow",
        "Save battery power — I'm on the go",
        "Something is using too much CPU — investigate",
        "Optimise my system for ML training",
        "Show me the current system state",
        "What is the deadlock risk right now?",
        "Show memory prediction accuracy",
        "Set up for web server workload",
        "Check for anomalous processes",
        "Show disk access cluster stats",
    ]
    print("\nExample goals you can type:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i:2d}. {ex}")
    print()


def run():
    print_banner()

    print("Initialising LLM planner...")
    planner  = AIMOSLLMPlanner(model="llama3.2")
    executor = AIMOSActionExecutor()
    print("Ready. AIMOS AI brain is online.\n")

    while True:
        try:
            goal = input("AIMOS > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down AIMOS AOS. Goodbye.")
            break

        if not goal:
            continue

        if goal.lower() in ('exit', 'quit', 'q'):
            print("Shutting down AIMOS AOS. Goodbye.")
            break

        if goal.lower() == 'help':
            print_examples()
            continue

        if goal.lower() == 'clear':
            planner.clear_history()
            print("Conversation history cleared.")
            continue

        print(f"\nPlanning: {goal}")
        print("Thinking", end="", flush=True)

        plan = planner.plan(goal)

        print("\r" + " " * 30 + "\r", end="")

        if not plan["success"]:
            print(f"Planning failed: {plan.get('error','unknown error')}")
            continue

        # Show plan
        print(f"\n Interpretation : {plan['interpretation']}")
        print(f" Actions        : {plan['actions']}")
        print(f" Explanation    : {plan['explanation']}")
        print(f" Expected effect: {plan['estimated_effect']}")

        # Confirm before executing
        print()
        confirm = input("Execute? [Y/n] ").strip().lower()
        if confirm in ('n', 'no'):
            print("Cancelled.\n")
            continue

        # Execute
        print("\nExecuting...")
        results = executor.execute_plan(plan["actions"])

        print("\n Results:")
        for action, action_results in results.items():
            print(f"  [{action}]")
            for r in action_results:
                print(f"    → {r}")

        print(f"\n Done. {len(plan['actions'])} action(s) executed.\n")


if __name__ == "__main__":
    run()
