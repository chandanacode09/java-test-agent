"""
Evaluation case data models.

Defines the input structure for evaluation cases - what classes to test
and what expectations to validate against.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EvalLevel(str, Enum):
    """Evaluation levels."""
    L1_COMPONENT = "l1_component"    # Individual component testing
    L2_PIPELINE = "l2_pipeline"      # Pipeline integration testing
    L3_AGENT = "l3_agent"            # Full agent aggregate testing


class ComponentType(str, Enum):
    """Component types for L1 evals."""
    AST_PARSER = "ast_parser"
    SPEC_GENERATOR = "spec_generator"
    VALIDATOR = "validator"
    TEMPLATE_RENDERER = "template_renderer"
    REACT_LOOP = "react_loop"


class ClassComplexity(str, Enum):
    """Complexity classification for Java classes."""
    POJO = "pojo"                    # Plain objects, getters/setters
    ENTITY = "entity"                # JPA entities with relationships
    SERVICE = "service"              # Business logic, dependencies
    CONTROLLER = "controller"        # Spring MVC controllers
    REPOSITORY = "repository"        # Data access layer
    UTILITY = "utility"              # Static utility classes


@dataclass
class EvalCaseMetadata:
    """Metadata describing the Java class under test."""
    class_name: str
    fully_qualified_name: str
    complexity: ClassComplexity
    method_count: int
    has_dependencies: bool           # Requires mocking
    has_inheritance: bool
    uses_generics: bool
    framework_annotations: list[str] # e.g., ["@Entity", "@Controller"]
    expected_test_count: int         # Baseline expected tests
    source_file: str                 # Relative path to source
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "fully_qualified_name": self.fully_qualified_name,
            "complexity": self.complexity.value,
            "method_count": self.method_count,
            "has_dependencies": self.has_dependencies,
            "has_inheritance": self.has_inheritance,
            "uses_generics": self.uses_generics,
            "framework_annotations": self.framework_annotations,
            "expected_test_count": self.expected_test_count,
            "source_file": self.source_file,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCaseMetadata":
        return cls(
            class_name=data["class_name"],
            fully_qualified_name=data["fully_qualified_name"],
            complexity=ClassComplexity(data["complexity"]),
            method_count=data["method_count"],
            has_dependencies=data["has_dependencies"],
            has_inheritance=data["has_inheritance"],
            uses_generics=data["uses_generics"],
            framework_annotations=data["framework_annotations"],
            expected_test_count=data["expected_test_count"],
            source_file=data["source_file"],
            tags=data.get("tags", []),
        )


@dataclass
class EvalCaseExpectation:
    """Expected outcomes for an eval case."""
    should_compile: bool = True      # Generated tests must compile
    should_pass: bool = True         # Generated tests must pass
    min_test_count: Optional[int] = None
    max_iterations: Optional[int] = None
    min_coverage: Optional[float] = None
    max_execution_time_sec: Optional[float] = None
    expected_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "should_compile": self.should_compile,
            "should_pass": self.should_pass,
            "min_test_count": self.min_test_count,
            "max_iterations": self.max_iterations,
            "min_coverage": self.min_coverage,
            "max_execution_time_sec": self.max_execution_time_sec,
            "expected_errors": self.expected_errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCaseExpectation":
        return cls(
            should_compile=data.get("should_compile", True),
            should_pass=data.get("should_pass", True),
            min_test_count=data.get("min_test_count"),
            max_iterations=data.get("max_iterations"),
            min_coverage=data.get("min_coverage"),
            max_execution_time_sec=data.get("max_execution_time_sec"),
            expected_errors=data.get("expected_errors", []),
        )


@dataclass
class EvalCase:
    """
    Single evaluation case representing a Java class to test.

    This is the atomic unit of evaluation - one class, one eval run.
    """
    # Identity (required)
    id: str                          # Unique case ID
    name: str                        # Human-readable name
    description: str

    # Classification (required)
    level: EvalLevel

    # Target specification (required)
    project_path: str                # Path to Maven project
    target_class: str                # Class name to generate tests for

    # Classification (optional)
    component: Optional[ComponentType] = None  # Required for L1 evals

    # Metadata about the class
    metadata: Optional[EvalCaseMetadata] = None

    # Expected outcomes
    expectations: EvalCaseExpectation = field(default_factory=EvalCaseExpectation)

    # Input overrides (for L1 testing specific inputs)
    input_override: Optional[dict[str, Any]] = None

    # Reference data
    reference_tests: Optional[str] = None
    golden_output: Optional[dict] = None

    # Eval configuration
    k_samples: int = 1               # Number of samples for pass@k
    timeout_sec: float = 300.0       # Max time per run

    # Version tracking
    dataset_version: str = "1.0"
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "component": self.component.value if self.component else None,
            "project_path": self.project_path,
            "target_class": self.target_class,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "expectations": self.expectations.to_dict(),
            "input_override": self.input_override,
            "reference_tests": self.reference_tests,
            "golden_output": self.golden_output,
            "k_samples": self.k_samples,
            "timeout_sec": self.timeout_sec,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        """Deserialize from dictionary."""
        metadata = None
        if data.get("metadata"):
            metadata = EvalCaseMetadata.from_dict(data["metadata"])

        expectations = EvalCaseExpectation()
        if data.get("expectations"):
            expectations = EvalCaseExpectation.from_dict(data["expectations"])

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            level=EvalLevel(data["level"]),
            component=ComponentType(data["component"]) if data.get("component") else None,
            project_path=data["project_path"],
            target_class=data["target_class"],
            metadata=metadata,
            expectations=expectations,
            input_override=data.get("input_override"),
            reference_tests=data.get("reference_tests"),
            golden_output=data.get("golden_output"),
            k_samples=data.get("k_samples", 1),
            timeout_sec=data.get("timeout_sec", 300.0),
            dataset_version=data.get("dataset_version", "1.0"),
            created_at=data.get("created_at"),
        )
