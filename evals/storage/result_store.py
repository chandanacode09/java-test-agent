"""
Result storage.

Persists evaluation runs to JSON files for later analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..models.eval_run import EvalRun
from ..models.eval_case import EvalCase


class ResultStore:
    """
    Stores and retrieves evaluation results.

    Uses JSON files in evals/results/ directory.
    """

    def __init__(self, results_dir: Path = None):
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: EvalRun) -> Path:
        """
        Save an eval run to disk.

        Returns:
            Path to the saved file
        """
        filename = f"{run.run_id}_{run.run_name}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)

        return filepath

    def load_run(self, run_id: str) -> Optional[EvalRun]:
        """
        Load an eval run by ID.

        Args:
            run_id: The run ID to load

        Returns:
            EvalRun if found, None otherwise
        """
        # Search for file matching run_id
        for filepath in self.results_dir.glob(f"{run_id}_*.json"):
            with open(filepath) as f:
                data = json.load(f)
                return EvalRun.from_dict(data)

        return None

    def list_runs(self, limit: int = 50) -> list[dict]:
        """
        List recent eval runs.

        Returns:
            List of run metadata dictionaries
        """
        runs = []

        for filepath in sorted(self.results_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    runs.append({
                        "run_id": data["run_id"],
                        "run_name": data["run_name"],
                        "model": data.get("model", "unknown"),
                        "completed_at": data.get("completed_at"),
                        "summary": data.get("summary"),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return runs

    def load_case(self, case_id: str) -> Optional[EvalCase]:
        """
        Load an eval case by ID from datasets.

        Args:
            case_id: The case ID to find

        Returns:
            EvalCase if found, None otherwise
        """
        datasets_dir = Path(__file__).parent.parent / "datasets"

        for dataset_dir in datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            for json_file in dataset_dir.glob("*.json"):
                if json_file.name == "manifest.json":
                    continue

                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        for case_data in data.get("cases", []):
                            if case_data.get("id") == case_id:
                                return EvalCase.from_dict(case_data)
                except (json.JSONDecodeError, KeyError):
                    continue

        return None
