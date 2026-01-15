"""
Evaluation runner.

Main orchestrator for running evaluations with pass@k metrics.
"""

import uuid
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Optional

from .models.eval_case import EvalCase, EvalLevel
from .models.eval_result import (
    EvalResult,
    EvalStatus,
    SampleResult,
    PassAtKMetrics,
)
from .models.eval_run import EvalRun
from .graders.base import BaseGrader
from .graders.compilation_grader import CompilationGrader
from .graders.execution_grader import ExecutionGrader
from .levels.l3_agent import L3AgentEvaluator


class EvalRunner:
    """
    Main orchestrator for running evaluations.

    Supports:
    - Running single eval cases
    - Running full datasets
    - Parallel execution (future)
    - Timeout handling
    - pass@k metrics
    """

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-haiku",
        api_key: Optional[str] = None,
        graders: Optional[list[BaseGrader]] = None,
        verbose: bool = True,
        max_methods: int = 0,  # 0 = no limit, >0 = faster evals
        generator_type: str = "spec",  # "spec" or "direct"
        self_verify: bool = True,  # For direct generator
    ):
        self.model = model
        self.api_key = api_key
        self.verbose = verbose
        self.max_methods = max_methods
        self.generator_type = generator_type
        self.self_verify = self_verify

        # Default graders
        self.graders = graders or [
            CompilationGrader(),
            ExecutionGrader(),
        ]

        # Level evaluators
        self.level_evaluators = {
            EvalLevel.L3_AGENT: L3AgentEvaluator(
                model, api_key, max_methods,
                generator_type=generator_type,
                self_verify=self_verify
            ),
        }

    def run_case(self, case: EvalCase) -> EvalResult:
        """
        Run a single eval case with k samples.

        Args:
            case: The eval case to run

        Returns:
            EvalResult with all sample results and pass@k metrics
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.now()

        sample_results = []

        # Get appropriate evaluator for level
        evaluator = self.level_evaluators.get(case.level)
        if not evaluator:
            return EvalResult(
                eval_case_id=case.id,
                run_id=run_id,
                model=self.model,
                status=EvalStatus.ERROR,
                error_message=f"No evaluator for level: {case.level}",
                started_at=started_at,
                completed_at=datetime.now(),
            )

        for sample_idx in range(case.k_samples):
            if self.verbose:
                print(f"  Sample {sample_idx + 1}/{case.k_samples}...", end=" ", flush=True)

            try:
                # Run with timeout
                sample_result = self._run_sample_with_timeout(
                    evaluator, case, sample_idx
                )
                sample_result.sample_index = sample_idx
                sample_results.append(sample_result)

                if self.verbose:
                    if sample_result.status == EvalStatus.SKIPPED:
                        print("SKIP (no methods to test)")
                    else:
                        status = "PASS" if sample_result.passed else "FAIL"
                        print(f"{status} ({sample_result.tests_passed}/{sample_result.tests_generated} tests)")

            except FuturesTimeout:
                sample_results.append(SampleResult(
                    sample_index=sample_idx,
                    status=EvalStatus.TIMEOUT,
                    grader_outputs=[],
                    duration_sec=case.timeout_sec,
                ))
                if self.verbose:
                    print("TIMEOUT")

            except Exception as e:
                sample_results.append(SampleResult(
                    sample_index=sample_idx,
                    status=EvalStatus.ERROR,
                    grader_outputs=[],
                    error_log=traceback.format_exc(),
                ))
                if self.verbose:
                    print(f"ERROR: {str(e)[:50]}")

        completed_at = datetime.now()

        # Compute pass@k metrics
        pass_at_k = PassAtKMetrics.compute(sample_results, case.k_samples)

        # Determine overall status
        if pass_at_k.n_pass > 0:
            status = EvalStatus.PASS
        elif all(s.status == EvalStatus.SKIPPED for s in sample_results):
            status = EvalStatus.SKIPPED
        elif any(s.status == EvalStatus.ERROR for s in sample_results):
            status = EvalStatus.ERROR
        elif any(s.status == EvalStatus.TIMEOUT for s in sample_results):
            status = EvalStatus.TIMEOUT
        else:
            status = EvalStatus.FAIL

        return EvalResult(
            eval_case_id=case.id,
            run_id=run_id,
            model=self.model,
            sample_results=sample_results,
            status=status,
            pass_at_k=pass_at_k,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            started_at=started_at,
            completed_at=completed_at,
        )

    def _run_sample_with_timeout(
        self,
        evaluator,
        case: EvalCase,
        sample_idx: int
    ) -> SampleResult:
        """Run a single sample with timeout protection."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(evaluator.evaluate, case, self.graders)
            try:
                return future.result(timeout=case.timeout_sec)
            except FuturesTimeout:
                future.cancel()
                raise

    def run_dataset(
        self,
        cases: list[EvalCase],
        run_name: str = None,
        level_filter: Optional[EvalLevel] = None,
    ) -> EvalRun:
        """
        Run evaluation on a full dataset.

        Args:
            cases: List of eval cases to run
            run_name: Optional name for this run
            level_filter: Optional filter by level

        Returns:
            EvalRun with all results and summary
        """
        run_id = str(uuid.uuid4())[:8]
        run_name = run_name or f"eval-{run_id}"

        # Filter by level if specified
        if level_filter:
            cases = [c for c in cases if c.level == level_filter]

        started_at = datetime.now()
        results = []

        for i, case in enumerate(cases):
            if self.verbose:
                print(f"[{i+1}/{len(cases)}] {case.name}...")

            result = self.run_case(case)
            results.append(result)

            if self.verbose:
                if result.status == EvalStatus.SKIPPED:
                    print(f"  -> SKIP (no methods to test)\n")
                else:
                    status = "PASS" if result.status == EvalStatus.PASS else "FAIL"
                    print(f"  -> {status} (pass@{result.pass_at_k.k}={result.pass_at_k.pass_rate:.2f})\n")

        completed_at = datetime.now()

        eval_run = EvalRun(
            run_id=run_id,
            run_name=run_name,
            dataset_name=cases[0].id.split("-")[0] if cases else "unknown",
            model=self.model,
            k_samples=cases[0].k_samples if cases else 1,
            results=results,
            started_at=started_at,
            completed_at=completed_at,
        )

        eval_run.compute_summary()

        return eval_run
