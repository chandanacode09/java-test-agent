"""
Execution grader.

Runs generated tests via Maven and parses Surefire XML reports.
"""

import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .base import BaseGrader
from ..models.eval_result import GraderOutput, GradeResult

# Import FAST_FLAGS to skip formatting plugins
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.maven_utils import FAST_FLAGS, get_maven_cmd


class ExecutionGrader(BaseGrader):
    """Grades test execution results via Maven/Surefire."""

    name = "execution"

    def __init__(self, timeout_sec: int = 180):
        self.timeout_sec = timeout_sec

    def grade(
        self,
        generated_code: str,
        project_path: str,
        test_file_path: str,
        test_class_name: Optional[str] = None,
        **kwargs
    ) -> GraderOutput:
        """
        Run mvn test and parse Surefire XML reports.

        Returns:
            GraderOutput with PASS if all tests pass,
            FAIL with test counts and failure details if any fail.
        """
        start = time.time()
        project = Path(project_path)

        # Clean up old Surefire reports to avoid stale data
        surefire_dir = project / "target" / "surefire-reports"
        if surefire_dir.exists() and test_class_name:
            # Only remove reports for the specific test class
            for old_report in surefire_dir.glob(f"*{test_class_name}*.xml"):
                old_report.unlink()

        # Determine Maven wrapper or command
        mvn_cmd = get_maven_cmd(project)

        # Build test filter command with FAST_FLAGS to skip formatters
        cmd = [mvn_cmd, "test", "-q"]
        cmd.extend(FAST_FLAGS)  # Skip spring-javaformat, checkstyle, etc.
        if test_class_name:
            cmd.append(f"-Dtest={test_class_name}")

        try:
            result = subprocess.run(
                cmd,
                cwd=project,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            duration_ms = (time.time() - start) * 1000

            # Parse Surefire reports
            surefire_dir = project / "target" / "surefire-reports"
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            tests_error = 0
            failure_details = []

            if surefire_dir.exists():
                # Filter to specific test class if provided
                pattern = f"TEST-*{test_class_name}*.xml" if test_class_name else "TEST-*.xml"

                for xml_file in surefire_dir.glob(pattern):
                    try:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()

                        tests_run += int(root.get("tests", 0))
                        tests_failed += int(root.get("failures", 0))
                        tests_error += int(root.get("errors", 0))

                        # Collect failure details
                        for testcase in root.findall(".//testcase"):
                            failure = testcase.find("failure")
                            error = testcase.find("error")
                            if failure is not None or error is not None:
                                elem = failure if failure is not None else error
                                failure_details.append({
                                    "test": testcase.get("name"),
                                    "class": testcase.get("classname"),
                                    "type": elem.get("type"),
                                    "message": (elem.get("message", "") or "")[:200],
                                })
                    except ET.ParseError:
                        continue

                tests_passed = tests_run - tests_failed - tests_error

            # Determine result
            all_passed = tests_failed == 0 and tests_error == 0 and tests_run > 0

            if all_passed:
                return GraderOutput(
                    grader_name=self.name,
                    result=GradeResult.PASS,
                    score=1.0,
                    message=f"All {tests_run} tests passed",
                    details={
                        "tests_run": tests_run,
                        "tests_passed": tests_passed,
                        "tests_failed": 0,
                        "tests_error": 0,
                    },
                    duration_ms=duration_ms,
                )
            elif tests_run == 0:
                return GraderOutput(
                    grader_name=self.name,
                    result=GradeResult.FAIL,
                    score=0.0,
                    message="No tests were executed",
                    details={
                        "stderr": result.stderr[-500:] if result.stderr else "",
                        "stdout": result.stdout[-500:] if result.stdout else "",
                    },
                    duration_ms=duration_ms,
                )
            else:
                score = tests_passed / tests_run if tests_run > 0 else 0.0
                return GraderOutput(
                    grader_name=self.name,
                    result=GradeResult.FAIL,
                    score=score,
                    message=f"{tests_passed}/{tests_run} tests passed ({tests_failed} failed, {tests_error} errors)",
                    details={
                        "tests_run": tests_run,
                        "tests_passed": tests_passed,
                        "tests_failed": tests_failed,
                        "tests_error": tests_error,
                        "failures": failure_details[:5],
                    },
                    duration_ms=duration_ms,
                )

        except subprocess.TimeoutExpired:
            duration_ms = (time.time() - start) * 1000
            return GraderOutput(
                grader_name=self.name,
                result=GradeResult.ERROR,
                score=0.0,
                message=f"Test execution timed out after {self.timeout_sec}s",
                details={},
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return GraderOutput(
                grader_name=self.name,
                result=GradeResult.ERROR,
                score=0.0,
                message=f"Test execution error: {str(e)}",
                details={"exception": str(e)},
                duration_ms=duration_ms,
            )
