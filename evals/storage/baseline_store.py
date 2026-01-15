"""
Baseline storage and management for evaluation framework.

Baselines are reference runs that can be compared against new runs.
They are stored in the dataset manifest files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .result_store import ResultStore


class BaselineStore:
    """Manages baselines for evaluation datasets."""

    def __init__(self):
        self.datasets_dir = Path(__file__).parent.parent / "datasets"
        self.result_store = ResultStore()

    def create_baseline(
        self,
        dataset_name: str,
        run_id: str,
        baseline_name: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Create a baseline from an existing run.

        Args:
            dataset_name: Name of the dataset (e.g., 'petclinic')
            run_id: ID of the run to use as baseline
            baseline_name: Name for the baseline (e.g., 'v1.0')
            notes: Optional notes about this baseline

        Returns:
            True if baseline was created successfully
        """
        # Load the run
        run = self.result_store.load_run(run_id)
        if not run:
            print(f"Error: Run {run_id} not found")
            return False

        # Load the manifest
        manifest_path = self.datasets_dir / dataset_name / "manifest.json"
        if not manifest_path.exists():
            print(f"Error: Dataset {dataset_name} not found")
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Create baseline entry
        baseline_entry = {
            "run_id": run_id,
            "model": run.model,
            "created_at": datetime.now().isoformat()[:10],
            "pass_rate": run.summary.overall_pass_rate if run.summary else 0.0,
            "total_cases": len(run.results),
            "passed_cases": run.summary.passed if run.summary else 0,
        }
        if notes:
            baseline_entry["notes"] = notes

        # Add to manifest
        if "baselines" not in manifest:
            manifest["baselines"] = {}

        manifest["baselines"][baseline_name] = baseline_entry

        # Save manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Created baseline '{baseline_name}' for dataset '{dataset_name}'")
        print(f"  Run ID: {run_id}")
        print(f"  Model: {run.model}")
        print(f"  Pass rate: {baseline_entry['pass_rate']*100:.1f}%")

        return True

    def list_baselines(self, dataset_name: str) -> list[dict]:
        """List all baselines for a dataset."""
        manifest_path = self.datasets_dir / dataset_name / "manifest.json"
        if not manifest_path.exists():
            return []

        with open(manifest_path) as f:
            manifest = json.load(f)

        baselines = manifest.get("baselines", {})
        return [
            {"name": name, **data}
            for name, data in baselines.items()
        ]

    def get_baseline(self, dataset_name: str, baseline_name: str) -> Optional[dict]:
        """Get a specific baseline."""
        baselines = self.list_baselines(dataset_name)
        for b in baselines:
            if b["name"] == baseline_name:
                return b
        return None

    def delete_baseline(self, dataset_name: str, baseline_name: str) -> bool:
        """Delete a baseline."""
        manifest_path = self.datasets_dir / dataset_name / "manifest.json"
        if not manifest_path.exists():
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        baselines = manifest.get("baselines", {})
        if baseline_name not in baselines:
            return False

        del baselines[baseline_name]
        manifest["baselines"] = baselines

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return True

    def list_all_baselines(self) -> dict[str, list[dict]]:
        """List baselines for all datasets."""
        result = {}
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                baselines = self.list_baselines(dataset_dir.name)
                if baselines:
                    result[dataset_dir.name] = baselines
        return result
