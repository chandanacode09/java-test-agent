"""
Evaluation run data models.

Aggregates multiple EvalResults into a complete evaluation run
with summary statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .eval_result import EvalResult, EvalStatus


@dataclass
class EvalRunSummary:
    """Summary statistics for an eval run."""
    total_cases: int
    passed: int
    failed: int
    errors: int
    timeouts: int
    skipped: int

    # Aggregate metrics
    overall_pass_rate: float
    avg_tests_generated: float
    avg_tests_passed: float
    avg_duration_sec: float
    avg_react_iterations: float

    # Breakdown by complexity
    by_complexity: dict[str, dict] = field(default_factory=dict)

    # Breakdown by level
    by_level: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "skipped": self.skipped,
            "overall_pass_rate": self.overall_pass_rate,
            "avg_tests_generated": self.avg_tests_generated,
            "avg_tests_passed": self.avg_tests_passed,
            "avg_duration_sec": self.avg_duration_sec,
            "avg_react_iterations": self.avg_react_iterations,
            "by_complexity": self.by_complexity,
            "by_level": self.by_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalRunSummary":
        return cls(
            total_cases=data["total_cases"],
            passed=data["passed"],
            failed=data["failed"],
            errors=data["errors"],
            timeouts=data["timeouts"],
            skipped=data["skipped"],
            overall_pass_rate=data["overall_pass_rate"],
            avg_tests_generated=data["avg_tests_generated"],
            avg_tests_passed=data["avg_tests_passed"],
            avg_duration_sec=data["avg_duration_sec"],
            avg_react_iterations=data["avg_react_iterations"],
            by_complexity=data.get("by_complexity", {}),
            by_level=data.get("by_level", {}),
        )


@dataclass
class EvalRun:
    """
    Complete evaluation run containing multiple EvalResults.

    Represents a single execution of the eval framework.
    """
    # Identity
    run_id: str
    run_name: str
    description: Optional[str] = None

    # Configuration
    dataset_name: str = ""
    level_filter: Optional[str] = None
    model: str = ""
    k_samples: int = 1

    # Results
    results: list[EvalResult] = field(default_factory=list)

    # Summary
    summary: Optional[EvalRunSummary] = None

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Comparison reference
    baseline_run_id: Optional[str] = None

    def compute_summary(self) -> EvalRunSummary:
        """Compute summary statistics from results."""
        passed = sum(1 for r in self.results if r.status == EvalStatus.PASS)
        failed = sum(1 for r in self.results if r.status == EvalStatus.FAIL)
        errors = sum(1 for r in self.results if r.status == EvalStatus.ERROR)
        timeouts = sum(1 for r in self.results if r.status == EvalStatus.TIMEOUT)
        skipped = sum(1 for r in self.results if r.status == EvalStatus.SKIPPED)

        total = len(self.results)

        # Compute averages from all samples
        all_samples = [s for r in self.results for s in r.sample_results]
        passing_samples = [s for s in all_samples if s.passed]

        avg_tests_generated = (
            sum(s.tests_generated for s in all_samples) / len(all_samples)
            if all_samples else 0.0
        )
        avg_tests_passed = (
            sum(s.tests_passed for s in all_samples) / len(all_samples)
            if all_samples else 0.0
        )
        avg_duration = (
            sum(s.duration_sec for s in all_samples) / len(all_samples)
            if all_samples else 0.0
        )
        avg_iterations = (
            sum(s.react_iterations for s in all_samples) / len(all_samples)
            if all_samples else 0.0
        )

        # Pass rate excludes skipped cases (they're not failures)
        testable_cases = total - skipped

        self.summary = EvalRunSummary(
            total_cases=total,
            passed=passed,
            failed=failed,
            errors=errors,
            timeouts=timeouts,
            skipped=skipped,
            overall_pass_rate=passed / testable_cases if testable_cases > 0 else 0.0,
            avg_tests_generated=avg_tests_generated,
            avg_tests_passed=avg_tests_passed,
            avg_duration_sec=avg_duration,
            avg_react_iterations=avg_iterations,
            by_complexity={},
            by_level={},
        )

        return self.summary

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "description": self.description,
            "dataset_name": self.dataset_name,
            "level_filter": self.level_filter,
            "model": self.model,
            "k_samples": self.k_samples,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary.to_dict() if self.summary else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "baseline_run_id": self.baseline_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalRun":
        return cls(
            run_id=data["run_id"],
            run_name=data["run_name"],
            description=data.get("description"),
            dataset_name=data.get("dataset_name", ""),
            level_filter=data.get("level_filter"),
            model=data.get("model", ""),
            k_samples=data.get("k_samples", 1),
            results=[EvalResult.from_dict(r) for r in data.get("results", [])],
            summary=EvalRunSummary.from_dict(data["summary"]) if data.get("summary") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            baseline_run_id=data.get("baseline_run_id"),
        )
