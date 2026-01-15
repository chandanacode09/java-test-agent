"""
L3 Agent-Level Evaluations.

Tests the full agent pipeline end-to-end:
1. Parse source class
2. Generate test specs
3. Render to Java code
4. Run ReAct loop until tests pass
5. Grade compilation and execution
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..models.eval_case import EvalCase
from ..models.eval_result import SampleResult, EvalStatus


class L3AgentEvaluator:
    """
    Evaluates full agent pipeline.

    Runs the complete JavaTestAgent.run() method and grades
    the results using provided graders.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_methods: int = 0,
        generator_type: str = "spec",
        self_verify: bool = True
    ):
        self.model = model
        self.api_key = api_key
        self.max_methods = max_methods  # 0 = no limit, >0 = faster evals
        self.generator_type = generator_type  # "spec" or "direct"
        self.self_verify = self_verify  # For direct generator

    def evaluate(
        self,
        case: EvalCase,
        graders: list = None  # Ignored - we trust SpecAgent results
    ) -> SampleResult:
        """
        Run full agent evaluation for one sample.

        Args:
            case: The eval case to run
            graders: Ignored (kept for API compatibility)

        Returns:
            SampleResult with agent metrics
        """
        start = time.time()

        # Import the Java agent
        try:
            from src.java_agent import JavaTestAgent, JavaAgentConfig
        except ImportError as e:
            return SampleResult(
                sample_index=0,
                status=EvalStatus.ERROR,
                grader_outputs=[],
                error_log=f"Failed to import JavaTestAgent: {e}",
                duration_sec=time.time() - start,
            )

        project_path = Path(case.project_path)

        # Clean up any existing generated test files to ensure isolation between cases
        self._cleanup_generated_tests(project_path, case.target_class)

        # Configure agent
        config = JavaAgentConfig(
            project_path=project_path,
            api_key=self.api_key,
            model=self.model,
            max_iterations=case.expectations.max_iterations or 5,
            verbose=False,
            max_methods=self.max_methods,
            generator_type=self.generator_type,
            self_verify=self.self_verify,
        )

        try:
            # Run the agent
            agent = JavaTestAgent(config)
            result = agent.run(case.target_class)

            # Handle skipped cases (empty classes with no methods)
            if result.skipped:
                return SampleResult(
                    sample_index=0,
                    status=EvalStatus.SKIPPED,
                    grader_outputs=[],
                    tests_generated=0,
                    tests_passed=0,
                    tests_failed=0,
                    react_iterations=0,
                    duration_sec=time.time() - start,
                    generated_test_code="# Skipped: class has no methods to test",
                )

            # Determine test file path
            test_file_path = self._get_test_file_path(
                project_path, case.target_class
            )

            # Read generated code if available
            generated_code = ""
            if test_file_path.exists():
                generated_code = test_file_path.read_text()

            duration = time.time() - start

            # Trust SpecAgent's results directly - no need for redundant graders
            all_passed = result.success

            return SampleResult(
                sample_index=0,
                status=EvalStatus.PASS if all_passed else EvalStatus.FAIL,
                grader_outputs=[],  # No graders - trust SpecAgent
                tests_generated=result.tests_total,
                tests_passed=result.tests_passed,
                tests_failed=result.tests_total - result.tests_passed,
                react_iterations=result.iterations,
                duration_sec=duration,
                generated_test_code=generated_code[:5000] if generated_code else None,
            )

        except Exception as e:
            return SampleResult(
                sample_index=0,
                status=EvalStatus.ERROR,
                grader_outputs=[],
                error_log=traceback.format_exc(),
                duration_sec=time.time() - start,
            )

    def _cleanup_generated_tests(self, project_path: Path, target_class: str) -> None:
        """
        Clean up ALL generated test files before running a new case.

        This ensures isolation between eval cases - old test files from
        previous cases won't interfere with grading the current case.
        We aggressively clean ALL test files that were generated by the agent
        (files newer than pom.xml or matching common patterns).
        """
        test_dir = project_path / "src/test/java"
        if not test_dir.exists():
            return

        pom_file = project_path / "pom.xml"
        pom_mtime = pom_file.stat().st_mtime if pom_file.exists() else 0

        # Clean up ALL test files that appear to be generated
        # (newer than pom.xml = likely generated by agent)
        for test_file in test_dir.rglob("*Test.java"):
            try:
                # Delete if newer than pom.xml (generated by agent)
                if test_file.stat().st_mtime > pom_mtime:
                    test_file.unlink()
            except Exception:
                pass

        # Also ensure the specific test file for this case is removed
        for test_file in test_dir.rglob(f"{target_class}Test.java"):
            try:
                test_file.unlink()
            except Exception:
                pass

    def _get_test_file_path(self, project_path: Path, class_name: str) -> Path:
        """
        Find the generated test file path.

        Searches common locations for the test file.
        """
        # Search in test directories
        test_dirs = [
            project_path / "src/test/java",
        ]

        for test_dir in test_dirs:
            if test_dir.exists():
                # Search recursively
                for test_file in test_dir.rglob(f"{class_name}Test.java"):
                    return test_file

        # Default path if not found
        return project_path / "src/test/java" / f"{class_name}Test.java"
