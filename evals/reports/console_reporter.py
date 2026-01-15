"""
Console reporter.

Prints evaluation results to the terminal in a readable format.
"""

from ..models.eval_run import EvalRun
from ..models.eval_result import EvalStatus


class ConsoleReporter:
    """Prints evaluation results to the console."""

    def report(self, run: EvalRun) -> None:
        """
        Print a full eval run report to the console.

        Args:
            run: The eval run to report on
        """
        print()
        print("=" * 70)
        print(f"EVAL RUN SUMMARY: {run.run_name}")
        print("=" * 70)
        print(f"Model: {run.model}")
        print(f"Dataset: {run.dataset_name}")
        print(f"Samples per case: {run.k_samples}")
        print()

        # Results table
        print("RESULTS:")
        print("-" * 70)
        print(f"{'Case':<35} {'Status':<8} {'Pass@k':<10} {'Tests':<12} {'Time':<8}")
        print("-" * 70)

        for result in run.results:
            if result.status == EvalStatus.SKIPPED:
                status = "SKIP"
            elif result.status == EvalStatus.PASS:
                status = "PASS"
            else:
                status = "FAIL"

            # Get test counts from first passing sample or last sample
            tests_str = "0/0"
            for sample in result.sample_results:
                if sample.tests_generated > 0:
                    tests_str = f"{sample.tests_passed}/{sample.tests_generated}"
                    break

            pass_at_k = f"{result.pass_at_k.n_pass}/{result.pass_at_k.k}" if result.pass_at_k else "0/0"
            time_str = f"{result.total_duration_sec:.1f}s"

            # Truncate case ID for display
            case_id = result.eval_case_id[:33] + ".." if len(result.eval_case_id) > 35 else result.eval_case_id

            print(f"{case_id:<35} {status:<8} {pass_at_k:<10} {tests_str:<12} {time_str:<8}")

        print("-" * 70)
        print()

        # Summary
        if run.summary:
            s = run.summary
            print("AGGREGATE METRICS:")
            print("-" * 40)
            print(f"Total Cases: {s.total_cases}")
            print(f"Passed: {s.passed} ({s.overall_pass_rate*100:.1f}%)")
            print(f"Failed: {s.failed}")
            if s.skipped > 0:
                print(f"Skipped: {s.skipped}")
            if s.errors > 0:
                print(f"Errors: {s.errors}")
            if s.timeouts > 0:
                print(f"Timeouts: {s.timeouts}")
            print()

            # pass@k metrics
            if run.results and run.results[0].pass_at_k:
                avg_pass_at_1 = sum(
                    r.pass_at_k.pass_at_1 for r in run.results if r.pass_at_k
                ) / len(run.results)
                print(f"Average Pass@1: {avg_pass_at_1:.2f}")

                if run.k_samples >= 5:
                    avg_pass_at_5 = sum(
                        r.pass_at_k.pass_at_5 or 0 for r in run.results if r.pass_at_k
                    ) / len(run.results)
                    print(f"Average Pass@5: {avg_pass_at_5:.2f}")

            print()
            print(f"Avg Tests Generated: {s.avg_tests_generated:.1f}")
            print(f"Avg Duration: {s.avg_duration_sec:.1f}s")
            if s.avg_react_iterations > 0:
                print(f"Avg React Iterations: {s.avg_react_iterations:.1f}")

        print("=" * 70)
        print()

    def report_comparison(self, run1: EvalRun, run2: EvalRun) -> None:
        """
        Print a comparison of two eval runs.

        Args:
            run1: First run (baseline)
            run2: Second run (comparison)
        """
        print()
        print("=" * 70)
        print("EVAL RUN COMPARISON")
        print("=" * 70)
        print(f"Baseline: {run1.run_name} ({run1.model})")
        print(f"Compare:  {run2.run_name} ({run2.model})")
        print()

        if run1.summary and run2.summary:
            s1, s2 = run1.summary, run2.summary

            print("METRICS COMPARISON:")
            print("-" * 50)
            print(f"{'Metric':<25} {'Baseline':<12} {'Compare':<12} {'Delta':<10}")
            print("-" * 50)

            # Pass rate
            delta = s2.overall_pass_rate - s1.overall_pass_rate
            sign = "+" if delta >= 0 else ""
            print(f"{'Pass Rate':<25} {s1.overall_pass_rate*100:.1f}%{'':<7} {s2.overall_pass_rate*100:.1f}%{'':<7} {sign}{delta*100:.1f}%")

            # Duration
            delta = s2.avg_duration_sec - s1.avg_duration_sec
            sign = "+" if delta >= 0 else ""
            print(f"{'Avg Duration':<25} {s1.avg_duration_sec:.1f}s{'':<8} {s2.avg_duration_sec:.1f}s{'':<8} {sign}{delta:.1f}s")

            # Tests
            delta = s2.avg_tests_generated - s1.avg_tests_generated
            sign = "+" if delta >= 0 else ""
            print(f"{'Avg Tests Generated':<25} {s1.avg_tests_generated:.1f}{'':<9} {s2.avg_tests_generated:.1f}{'':<9} {sign}{delta:.1f}")

            print("-" * 50)

        print("=" * 70)
        print()
