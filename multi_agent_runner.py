"""
multi_agent_runner.py
CLI runner for the multi-agent supply chain system.
Usage:
    python multi_agent_runner.py --task task_easy
    python multi_agent_runner.py --task task_medium --verbose
    python multi_agent_runner.py --task all
    python multi_agent_runner.py --task task_hard --max-steps 150 --url http://localhost:7860
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Runner")
def validate_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not key.startswith("sk-"):
        print("❌  OPENAI_API_KEY not set or invalid.")
        print("    Set it in .env file:  OPENAI_API_KEY=sk-...")
        sys.exit(1)
    return key
def run_task(
    task_id:  str,
    api_key:  str,
    base_url: str,
    max_steps: int,
    verbose:  bool,
    output_dir: Path,
) -> dict:
    """Run a full multi-agent episode for one task."""
    from agents import OrchestratorAgent
    print(f"\n{'='*65}")
    print(f"  🚢 Multi-Agent Supply Chain — {task_id.upper()}")
    print(f"{'='*65}\n")
    orchestrator = OrchestratorAgent(
        task_id=task_id,
        api_key=api_key,
        base_url=base_url,
        verbose=verbose,
    )
    start = time.perf_counter()
    result = orchestrator.run_episode(max_steps=max_steps)
    elapsed = time.perf_counter() - start
    # ── Print summary ──────────────────────────────────────────
    grade = result.get("final_grade", {})
    tokens = result.get("token_usage", {})
    print(f"\n{'='*65}")
    print(f"  📊 EPISODE SUMMARY — {task_id}")
    print(f"{'='*65}")
    print(f"  Score:          {grade.get('score', 0):.4f}  ({'✅ PASS' if grade.get('passed') else '❌ FAIL'})")
    print(f"  Steps taken:    {result.get('total_steps', 0)}")
    print(f"  On-time rate:   {result.get('on_time_rate', 0):.1%}")
    print(f"  Mean reward:    {result.get('mean_reward', 0):+.4f}")
    print(f"  Budget used:    {result.get('budget_utilization', 0):.1%}")
    print(f"  Premium rate:   {result.get('premium_route_rate', 0):.1%}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"\n  💰 Token Usage:")
    print(f"     Orchestrator:    {tokens.get('Orchestrator', {}).get('total', 0):,} tokens")
    print(f"     RoutingAgent:    {tokens.get('RoutingAgent', {}).get('total', 0):,} tokens")
    print(f"     DisruptionAgent: {tokens.get('DisruptionAgent', {}).get('total', 0):,} tokens")
    print(f"     BudgetGuardian:  {tokens.get('BudgetGuardian', {}).get('total', 0):,} tokens")
    print(f"     Total API cost:  ${tokens.get('total_cost_usd', 0):.4f}")
    if grade.get("subscores"):
        print(f"\n  📋 Subscores:")
        for k, v in grade["subscores"].items():
            print(f"     {k:<30} {v:.4f}")
    print(f"{'='*65}\n")
    # ── Save result ────────────────────────────────────────────
    out_file = output_dir / f"{task_id}_result.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))
    print(f"  📁 Full result saved → {out_file}")
    return result
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Supply Chain Runner — OpenAI powered"
    )
    parser.add_argument(
        "--task", "-t",
        default="task_easy",
        choices=["task_easy", "task_medium", "task_hard", "all"],
        help="Task to run (default: task_easy)",
    )
    parser.add_argument(
        "--max-steps", "-m",
        type=int,
        default=200,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--url", "-u",
        default="http://localhost:7860",
        help="Base URL of the supply chain API (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory for result JSON files (default: ./results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Show per-step decision logs (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-step logs",
    )
    args = parser.parse_args()
    api_key = validate_api_key()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = args.verbose and not args.quiet
    tasks = (
        ["task_easy", "task_medium", "task_hard"]
        if args.task == "all"
        else [args.task]
    )
    all_results = {}
    for task_id in tasks:
        result = run_task(
            task_id=task_id,
            api_key=api_key,
            base_url=args.url,
            max_steps=args.max_steps,
            verbose=verbose,
            output_dir=output_dir,
        )
        all_results[task_id] = {
            "score":    result.get("final_grade", {}).get("score", 0),
            "passed":   result.get("final_grade", {}).get("passed", False),
            "cost_usd": result.get("token_usage", {}).get("total_cost_usd", 0),
        }
    if len(tasks) > 1:
        print(f"\n{'='*65}")
        print("  🏆 MULTI-TASK SUMMARY")
        print(f"{'='*65}")
        for t, r in all_results.items():
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"  {t:<15} score={r['score']:.4f}  {status}  cost=${r['cost_usd']:.4f}")
        print(f"{'='*65}\n")
if __name__ == "__main__":
    main()