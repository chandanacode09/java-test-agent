"""
Compilation grader.

Checks whether generated tests compile successfully using Maven.
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import BaseGrader
from ..models.eval_result import GraderOutput, GradeResult

# Import FAST_FLAGS to skip formatting plugins
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.maven_utils import FAST_FLAGS, get_maven_cmd


class CompilationGrader(BaseGrader):
    """Grades whether generated tests compile successfully."""

    name = "compilation"

    def __init__(self, timeout_sec: int = 120):
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
        Run mvn test-compile and check for compilation errors.

        Returns:
            GraderOutput with PASS if compilation succeeds,
            FAIL with error count and details if it fails.
        """
        start = time.time()
        project = Path(project_path)

        # Determine Maven wrapper or command
        mvn_cmd = get_maven_cmd(project)

        # Build command with FAST_FLAGS to skip formatters
        cmd = [mvn_cmd, "test-compile", "-q"]
        cmd.extend(FAST_FLAGS)

        try:
            result = subprocess.run(
                cmd,
                cwd=project,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )

            duration_ms = (time.time() - start) * 1000

            if result.returncode == 0:
                return GraderOutput(
                    grader_name=self.name,
                    result=GradeResult.PASS,
                    score=1.0,
                    message="Compilation successful",
                    details={"stdout": result.stdout[-500:] if result.stdout else ""},
                    duration_ms=duration_ms,
                )
            else:
                # Parse compilation errors
                output = result.stdout + result.stderr
                error_pattern = r'\[ERROR\].*\.java:\[(\d+),(\d+)\]'
                errors = re.findall(error_pattern, output)
                error_count = len(errors)

                # Extract first few error messages
                error_lines = [
                    line for line in output.split('\n')
                    if '[ERROR]' in line and '.java:' in line
                ][:5]

                return GraderOutput(
                    grader_name=self.name,
                    result=GradeResult.FAIL,
                    score=0.0,
                    message=f"Compilation failed with {error_count} errors",
                    details={
                        "error_count": error_count,
                        "error_lines": error_lines,
                        "stderr": result.stderr[-1000:] if result.stderr else "",
                    },
                    duration_ms=duration_ms,
                )

        except subprocess.TimeoutExpired:
            duration_ms = (time.time() - start) * 1000
            return GraderOutput(
                grader_name=self.name,
                result=GradeResult.ERROR,
                score=0.0,
                message=f"Compilation timed out after {self.timeout_sec}s",
                details={},
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return GraderOutput(
                grader_name=self.name,
                result=GradeResult.ERROR,
                score=0.0,
                message=f"Compilation error: {str(e)}",
                details={"exception": str(e)},
                duration_ms=duration_ms,
            )
