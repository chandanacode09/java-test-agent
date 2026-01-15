"""
Base grader interface.

All graders inherit from this class and implement the grade() method.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..models.eval_result import GraderOutput


class BaseGrader(ABC):
    """
    Base class for all graders.

    Graders evaluate specific aspects of generated test code:
    - CompilationGrader: Does the code compile?
    - ExecutionGrader: Do the tests pass?
    - CoverageGrader: What's the code coverage?
    """

    name: str = "base"

    @abstractmethod
    def grade(
        self,
        generated_code: str,
        project_path: str,
        test_file_path: str,
        test_class_name: Optional[str] = None,
        **kwargs
    ) -> GraderOutput:
        """
        Grade the generated test code.

        Args:
            generated_code: The generated Java test code
            project_path: Path to the Maven project
            test_file_path: Path where test file was written
            test_class_name: Name of the test class (for filtering)
            **kwargs: Additional grader-specific arguments

        Returns:
            GraderOutput with result, score, and details
        """
        pass
