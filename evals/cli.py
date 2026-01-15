#!/usr/bin/env python3
"""
CLI for Java Test Agent Evaluation Framework.

Usage:
    python -m evals.cli run --dataset petclinic --level l3 --k 5
    python -m evals.cli run --case petclinic-vet-001 --k 5
    python -m evals.cli list --datasets
    python -m evals.cli list --runs
    python -m evals.cli report --run abc123
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .runner import EvalRunner
from .models.eval_case import EvalCase, EvalLevel
from .storage.result_store import ResultStore
from .storage.baseline_store import BaselineStore
from .reports.console_reporter import ConsoleReporter


def main():
    parser = argparse.ArgumentParser(
        description="Java Test Agent Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # === RUN command ===
    run_parser = subparsers.add_parser("run", help="Run evaluations")
    run_parser.add_argument(
        "--dataset", "-d",
        help="Dataset name (e.g., petclinic)",
    )
    run_parser.add_argument(
        "--case", "-c",
        help="Specific case ID to run",
    )
    run_parser.add_argument(
        "--level", "-l",
        choices=["l1", "l2", "l3"],
        help="Filter by eval level",
    )
    run_parser.add_argument(
        "--model", "-m",
        default="anthropic/claude-3.5-haiku",
        help="Model to use (default: claude-3.5-haiku)",
    )
    run_parser.add_argument(
        "--k", "-k",
        type=int,
        default=1,
        help="Number of samples for pass@k (default: 1)",
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Output file for results JSON",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    run_parser.add_argument(
        "--api-key",
        help="API key (or set OPENROUTER_API_KEY env var)",
    )
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: limit to 2 methods per class for quicker evals",
    )
    run_parser.add_argument(
        "--max-methods",
        type=int,
        default=0,
        help="Max methods per class (0=no limit, overrides --fast)",
    )
    run_parser.add_argument(
        "--generator", "-g",
        choices=["spec", "direct", "dsl"],
        default="spec",
        help="Generator type: 'spec' (JSON spec + template), 'direct' (code + self-verify), or 'dsl' (DSL + compiler)",
    )
    run_parser.add_argument(
        "--no-self-verify",
        action="store_true",
        help="For direct generator: disable self-verification loop (for comparison)",
    )

    # === LIST command ===
    list_parser = subparsers.add_parser("list", help="List resources")
    list_parser.add_argument(
        "--datasets", action="store_true",
        help="List available datasets",
    )
    list_parser.add_argument(
        "--runs", action="store_true",
        help="List past eval runs",
    )
    list_parser.add_argument(
        "--cases", metavar="DATASET",
        help="List cases in a dataset",
    )

    # === REPORT command ===
    report_parser = subparsers.add_parser("report", help="Show run report")
    report_parser.add_argument(
        "--run", "-r", required=True,
        help="Run ID to report on",
    )

    # === BASELINE command ===
    baseline_parser = subparsers.add_parser("baseline", help="Manage baselines")
    baseline_subparsers = baseline_parser.add_subparsers(dest="baseline_cmd")

    # baseline create
    baseline_create = baseline_subparsers.add_parser("create", help="Create baseline from run")
    baseline_create.add_argument(
        "--run", "-r", required=True,
        help="Run ID to use as baseline",
    )
    baseline_create.add_argument(
        "--name", "-n", required=True,
        help="Name for the baseline (e.g., v1.0)",
    )
    baseline_create.add_argument(
        "--dataset", "-d", required=True,
        help="Dataset name",
    )
    baseline_create.add_argument(
        "--notes",
        help="Optional notes about this baseline",
    )

    # baseline list
    baseline_list = baseline_subparsers.add_parser("list", help="List baselines")
    baseline_list.add_argument(
        "--dataset", "-d",
        help="Dataset name (optional, lists all if not specified)",
    )

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "baseline":
        return cmd_baseline(args)
    else:
        parser.print_help()
        return 1


def cmd_run(args):
    """Execute eval run."""
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: API key required. Set --api-key or OPENROUTER_API_KEY env var")
        return 1

    # Load dataset or single case
    cases = []
    if args.case:
        store = ResultStore()
        case = store.load_case(args.case)
        if not case:
            print(f"Error: Case {args.case} not found")
            return 1
        cases = [case]
    elif args.dataset:
        cases = load_dataset(args.dataset)
        if not cases:
            print(f"Error: Dataset {args.dataset} not found or empty")
            return 1
    else:
        print("Error: Specify --dataset or --case")
        return 1

    # Apply k_samples override
    if args.k > 1:
        for case in cases:
            case.k_samples = args.k

    # Apply level filter
    level_filter = None
    if args.level:
        level_map = {
            "l1": EvalLevel.L1_COMPONENT,
            "l2": EvalLevel.L2_PIPELINE,
            "l3": EvalLevel.L3_AGENT
        }
        level_filter = level_map[args.level]

    # Determine max_methods (--fast sets to 2, --max-methods overrides)
    max_methods = args.max_methods if args.max_methods > 0 else (2 if args.fast else 0)

    # Create runner
    runner = EvalRunner(
        model=args.model,
        api_key=api_key,
        verbose=args.verbose,
        max_methods=max_methods,
        generator_type=args.generator,
        self_verify=not args.no_self_verify,
    )

    # Run
    generator_info = f" (generator={args.generator}" + (", self_verify=False" if args.no_self_verify else "") + ")"
    print(f"\nRunning {len(cases)} eval case(s) with {args.model}{generator_info}")
    print(f"k={args.k} sample(s) per case\n")

    eval_run = runner.run_dataset(cases, level_filter=level_filter)

    # Print summary
    ConsoleReporter().report(eval_run)

    # Save results
    store = ResultStore()
    filepath = store.save_run(eval_run)
    print(f"Run saved: {filepath}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(eval_run.to_dict(), f, indent=2, default=str)
        print(f"Results written to {args.output}")

    # Return code based on pass rate
    return 0 if eval_run.summary and eval_run.summary.overall_pass_rate >= 0.5 else 1


def cmd_list(args):
    """List resources."""
    if args.datasets:
        datasets_dir = Path(__file__).parent / "datasets"
        print("\nAvailable datasets:")
        print("-" * 40)

        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    desc = manifest.get("description", "No description")
                    print(f"  {d.name}: {desc}")
            else:
                print(f"  {d.name}: (no manifest)")

        print()

    elif args.runs:
        store = ResultStore()
        runs = store.list_runs()

        print("\nPast eval runs:")
        print("-" * 60)

        if not runs:
            print("  No runs found")
        else:
            for run in runs[:20]:
                summary = run.get("summary", {})
                pass_rate = summary.get("overall_pass_rate", 0) * 100 if summary else 0
                print(f"  {run['run_id']}: {run['run_name']} ({run['model']}) - {pass_rate:.0f}% pass")

        print()

    elif args.cases:
        cases = load_dataset(args.cases)

        print(f"\nCases in {args.cases}:")
        print("-" * 60)

        if not cases:
            print("  No cases found")
        else:
            for case in cases:
                level = case.level.value.replace("_", " ").upper()
                print(f"  {case.id}: {case.name} [{level}]")

        print()

    return 0


def cmd_report(args):
    """Show report for a run."""
    store = ResultStore()
    run = store.load_run(args.run)

    if not run:
        print(f"Error: Run {args.run} not found")
        return 1

    ConsoleReporter().report(run)
    return 0


def cmd_baseline(args):
    """Manage baselines."""
    store = BaselineStore()

    if args.baseline_cmd == "create":
        success = store.create_baseline(
            dataset_name=args.dataset,
            run_id=args.run,
            baseline_name=args.name,
            notes=args.notes,
        )
        return 0 if success else 1

    elif args.baseline_cmd == "list":
        if args.dataset:
            baselines = store.list_baselines(args.dataset)
            print(f"\nBaselines for {args.dataset}:")
            print("-" * 60)
            if not baselines:
                print("  No baselines found")
            else:
                for b in baselines:
                    pass_rate = b.get("pass_rate", 0) * 100
                    print(f"  {b['name']}: {b.get('model', 'unknown')} - {pass_rate:.1f}% pass")
                    if b.get("notes"):
                        print(f"    Notes: {b['notes']}")
            print()
        else:
            all_baselines = store.list_all_baselines()
            print("\nAll baselines:")
            print("-" * 60)
            if not all_baselines:
                print("  No baselines found")
            else:
                for dataset, baselines in all_baselines.items():
                    print(f"\n  {dataset}:")
                    for b in baselines:
                        pass_rate = b.get("pass_rate", 0) * 100
                        print(f"    {b['name']}: {b.get('model', 'unknown')} - {pass_rate:.1f}% pass")
            print()
        return 0

    else:
        print("Error: Specify 'create' or 'list' subcommand")
        return 1


def load_dataset(dataset_name: str) -> list[EvalCase]:
    """Load all cases from a dataset directory."""
    datasets_dir = Path(__file__).parent / "datasets" / dataset_name

    if not datasets_dir.exists():
        return []

    manifest_path = datasets_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    cases = []

    # Load cases from referenced files
    for level, files in manifest.get("cases", {}).items():
        for filename in files:
            file_path = datasets_dir / filename
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    for case_data in data.get("cases", []):
                        cases.append(EvalCase.from_dict(case_data))

    return cases


if __name__ == "__main__":
    sys.exit(main())
