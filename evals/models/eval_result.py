"""
Evaluation result data models.

Defines the output structure for evaluation results including
individual samples, grader outputs, and pass@k metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import lgamma, exp
from typing import Any, Optional


class EvalStatus(str, Enum):
    """Evaluation outcome status."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"                  # Eval infrastructure error
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class GradeResult(str, Enum):
    """Individual grader result."""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    NOT_APPLICABLE = "n/a"


@dataclass
class GraderOutput:
    """Output from a single grader."""
    grader_name: str
    result: GradeResult
    score: float                     # 0.0 to 1.0
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "grader_name": self.grader_name,
            "result": self.result.value,
            "score": self.score,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraderOutput":
        return cls(
            grader_name=data["grader_name"],
            result=GradeResult(data["result"]),
            score=data["score"],
            message=data["message"],
            details=data.get("details", {}),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class SampleResult:
    """Result from a single sample run (for pass@k)."""
    sample_index: int
    status: EvalStatus
    grader_outputs: list[GraderOutput] = field(default_factory=list)

    # Test execution metrics
    tests_generated: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    compilation_errors: int = 0

    # ReAct loop metrics (L3 only)
    react_iterations: int = 0
    fixes_applied: int = 0

    # Timing
    duration_sec: float = 0.0

    # Artifacts
    generated_test_code: Optional[str] = None
    maven_output: Optional[str] = None
    error_log: Optional[str] = None

    @property
    def passed(self) -> bool:
        """Did this sample pass all graders?"""
        return self.status == EvalStatus.PASS

    def to_dict(self) -> dict:
        return {
            "sample_index": self.sample_index,
            "status": self.status.value,
            "grader_outputs": [g.to_dict() for g in self.grader_outputs],
            "tests_generated": self.tests_generated,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "compilation_errors": self.compilation_errors,
            "react_iterations": self.react_iterations,
            "fixes_applied": self.fixes_applied,
            "duration_sec": self.duration_sec,
            "generated_test_code": self.generated_test_code,
            "maven_output": self.maven_output,
            "error_log": self.error_log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SampleResult":
        return cls(
            sample_index=data["sample_index"],
            status=EvalStatus(data["status"]),
            grader_outputs=[GraderOutput.from_dict(g) for g in data.get("grader_outputs", [])],
            tests_generated=data.get("tests_generated", 0),
            tests_passed=data.get("tests_passed", 0),
            tests_failed=data.get("tests_failed", 0),
            compilation_errors=data.get("compilation_errors", 0),
            react_iterations=data.get("react_iterations", 0),
            fixes_applied=data.get("fixes_applied", 0),
            duration_sec=data.get("duration_sec", 0.0),
            generated_test_code=data.get("generated_test_code"),
            maven_output=data.get("maven_output"),
            error_log=data.get("error_log"),
        )


def _comb(n: int, k: int) -> float:
    """Compute n choose k using log-gamma to avoid overflow."""
    if k > n or k < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    return exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))


@dataclass
class PassAtKMetrics:
    """Pass@k metrics for a single eval case."""
    k: int                           # Number of samples
    n_pass: int                      # Number of passing samples
    pass_rate: float                 # n_pass / k

    # Unbiased pass@k estimators
    pass_at_1: float
    pass_at_5: Optional[float] = None
    pass_at_10: Optional[float] = None

    @classmethod
    def compute(cls, sample_results: list[SampleResult], k: int = None) -> "PassAtKMetrics":
        """
        Compute pass@k metrics from sample results.

        Uses the unbiased estimator from the Codex paper:
        pass@k = 1 - (n-c choose k) / (n choose k)
        where n = total samples, c = correct samples
        """
        if k is None:
            k = len(sample_results)

        n = len(sample_results)
        c = sum(1 for s in sample_results if s.passed)
        pass_rate = c / k if k > 0 else 0.0

        def pass_at_k_estimate(n: int, c: int, k: int) -> float:
            if n - c < k:
                return 1.0
            if n == 0:
                return 0.0
            return 1.0 - (_comb(n - c, k) / _comb(n, k))

        return cls(
            k=k,
            n_pass=c,
            pass_rate=pass_rate,
            pass_at_1=pass_at_k_estimate(n, c, 1),
            pass_at_5=pass_at_k_estimate(n, c, 5) if n >= 5 else None,
            pass_at_10=pass_at_k_estimate(n, c, 10) if n >= 10 else None,
        )

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n_pass": self.n_pass,
            "pass_rate": self.pass_rate,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "pass_at_10": self.pass_at_10,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PassAtKMetrics":
        return cls(
            k=data["k"],
            n_pass=data["n_pass"],
            pass_rate=data["pass_rate"],
            pass_at_1=data["pass_at_1"],
            pass_at_5=data.get("pass_at_5"),
            pass_at_10=data.get("pass_at_10"),
        )


@dataclass
class EvalResult:
    """
    Complete result for one EvalCase execution.

    Contains all sample results, aggregated metrics, and metadata.
    """
    # Identity
    eval_case_id: str
    run_id: str

    # Configuration used
    model: str
    model_temperature: float = 0.0

    # Sample results
    sample_results: list[SampleResult] = field(default_factory=list)

    # Aggregated status
    status: EvalStatus = EvalStatus.FAIL

    # Metrics
    pass_at_k: PassAtKMetrics = None

    # Timing
    total_duration_sec: float = 0.0
    started_at: datetime = None
    completed_at: datetime = None

    # Error tracking
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Environment
    python_version: Optional[str] = None
    java_version: Optional[str] = None
    maven_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "eval_case_id": self.eval_case_id,
            "run_id": self.run_id,
            "model": self.model,
            "model_temperature": self.model_temperature,
            "status": self.status.value,
            "sample_results": [s.to_dict() for s in self.sample_results],
            "pass_at_k": self.pass_at_k.to_dict() if self.pass_at_k else None,
            "total_duration_sec": self.total_duration_sec,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "python_version": self.python_version,
            "java_version": self.java_version,
            "maven_version": self.maven_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        return cls(
            eval_case_id=data["eval_case_id"],
            run_id=data["run_id"],
            model=data["model"],
            model_temperature=data.get("model_temperature", 0.0),
            sample_results=[SampleResult.from_dict(s) for s in data.get("sample_results", [])],
            status=EvalStatus(data["status"]),
            pass_at_k=PassAtKMetrics.from_dict(data["pass_at_k"]) if data.get("pass_at_k") else None,
            total_duration_sec=data.get("total_duration_sec", 0.0),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error_message=data.get("error_message"),
            error_traceback=data.get("error_traceback"),
            python_version=data.get("python_version"),
            java_version=data.get("java_version"),
            maven_version=data.get("maven_version"),
        )
