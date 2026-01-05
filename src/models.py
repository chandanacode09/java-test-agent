"""
Core data models for the test generation pipeline.

These models define the contracts between different phases:
- Context extraction → LLM
- LLM → Template engine
- Template engine → Test runner
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pathlib import Path


# ============================================================================
# Enums
# ============================================================================

class TestType(str, Enum):
    """Types of tests we can generate."""
    UNIT_PURE = "unit_pure"           # Pure functions, no side effects
    UNIT_CLASS = "unit_class"         # Class methods
    UNIT_MOCKED = "unit_mocked"       # Functions requiring mocks
    INTEGRATION_API = "integration_api"
    INTEGRATION_DB = "integration_db"
    EDGE_CASE = "edge_case"
    CUSTOM = "custom"                  # Fallback to LLM generation


class TestCategory(str, Enum):
    """Categories for test cases."""
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    BOUNDARY = "boundary"
    NULL_EMPTY = "null_empty"
    PERFORMANCE = "performance"


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"


class CoverageStatus(str, Enum):
    """Coverage threshold status."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class ErrorCategory(str, Enum):
    """Categories of test errors for specialized fix strategies."""
    ASSERTION_FAILURE = "assertion_failure"      # assertEquals failed, expected vs actual mismatch
    NULL_POINTER = "null_pointer"                # NullPointerException
    MISSING_METHOD = "missing_method"            # Method does not exist
    WRONG_ARGUMENTS = "wrong_arguments"          # Constructor/method argument mismatch
    CLASS_NOT_FOUND = "class_not_found"          # Import/class resolution error
    TYPE_MISMATCH = "type_mismatch"              # Incompatible types
    ILLEGAL_ARGUMENT = "illegal_argument"        # IllegalArgumentException
    COMPILATION_ERROR = "compilation_error"      # General compilation error
    UNKNOWN = "unknown"                          # Fallback


# ============================================================================
# Context Models (Phase 1 Output)
# ============================================================================

@dataclass
class CodeLocation:
    """Location of code in a file."""
    file_path: Path
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0


@dataclass
class Parameter:
    """Function/method parameter."""
    name: str
    type_hint: str | None = None
    default_value: str | None = None
    is_optional: bool = False


class ReturnSemantics(str, Enum):
    """Semantic meaning of return type."""
    VALUE = "value"                 # Returns a concrete value
    GENERATOR = "generator"         # Uses yield, returns iterator
    ITERATOR = "iterator"           # Returns Iterator/Iterable type
    NONE_SIDEEFFECT = "none_sideeffect"  # Returns None, has side effects
    CONTEXT_MANAGER = "context_manager"  # Returns a context manager


@dataclass
class FunctionContext:
    """Extracted context for a function."""
    name: str
    location: CodeLocation
    parameters: list[Parameter]
    return_type: str | None
    docstring: str | None
    is_async: bool
    is_method: bool
    class_name: str | None
    decorators: list[str]
    source_code: str

    # Dependencies extracted via LSP/static analysis
    calls: list[str] = field(default_factory=list)           # Functions this calls
    imports: list[str] = field(default_factory=list)         # Required imports
    references: list[str] = field(default_factory=list)      # External references

    # NEW: Semantic understanding fields
    is_generator: bool = False                               # Has yield statements
    return_semantics: ReturnSemantics = ReturnSemantics.VALUE  # How to handle return
    requires_context: list[str] = field(default_factory=list)  # ["flask_app", "request", "db_session"]
    mutates_args: bool = False                               # Modifies input arguments
    framework_hints: list[str] = field(default_factory=list)   # ["flask", "werkzeug", "sqlalchemy"]


@dataclass
class ClassContext:
    """Extracted context for a class."""
    name: str
    location: CodeLocation
    methods: list[FunctionContext]
    base_classes: list[str]
    docstring: str | None
    decorators: list[str]
    attributes: list[tuple[str, str | None]]  # (name, type_hint)
    constructors: list[FunctionContext] = field(default_factory=list)  # Constructors for this class


@dataclass
class FixAttempt:
    """Track a fix attempt for self-reflection in ReAct loops."""
    iteration: int
    error_category: ErrorCategory
    original_code: str
    attempted_fix: str
    result: str  # "compiled", "compile_error", "test_failed", "test_passed"
    error_after_fix: str | None = None


@dataclass
class FileChange:
    """A changed file in the PR diff."""
    file_path: Path
    change_type: str  # "added", "modified", "deleted"
    added_lines: list[int]
    removed_lines: list[int]
    diff_content: str


@dataclass
class PRContext:
    """Complete context extracted from a PR."""
    pr_number: int | None
    branch_name: str
    base_branch: str
    changed_files: list[FileChange]
    functions: list[FunctionContext]
    classes: list[ClassContext]
    language: Language

    # Metadata
    total_lines_added: int = 0
    total_lines_removed: int = 0


# ============================================================================
# Pattern Models (Phase 2 Output)
# ============================================================================

@dataclass
class TestPattern:
    """A pattern extracted from existing tests."""
    pattern_type: str              # e.g., "fixture_usage", "mock_pattern"
    example_code: str
    frequency: int                 # How often this pattern appears
    associated_imports: list[str]
    description: str


@dataclass
class ProjectTestPatterns:
    """All test patterns found in the project."""
    fixtures: list[str]
    common_mocks: list[str]
    assertion_patterns: list[str]
    test_utilities: list[str]
    patterns: list[TestPattern]


# ============================================================================
# Test Specification Models (Phase 3 Output - LLM generates these)
# ============================================================================

@dataclass
class ExpectedOutput:
    """Expected output for a test case."""
    returns: Any | None = None
    raises: str | None = None
    raises_message: str | None = None
    side_effects: list[dict] | None = None  # e.g., [{"call": "mock.method", "args": [...]}]


@dataclass
class MockSpec:
    """Specification for a mock."""
    target: str                    # What to mock (e.g., "requests.get")
    return_value: Any | None = None
    side_effect: Any | None = None
    assert_called_with: list[Any] | None = None


@dataclass
class TestCase:
    """A single test case specification."""
    name: str
    category: TestCategory
    description: str
    inputs: dict[str, Any]
    expected: ExpectedOutput
    mocks: list[MockSpec] = field(default_factory=list)
    setup: list[str] = field(default_factory=list)    # Setup code lines
    teardown: list[str] = field(default_factory=list) # Teardown code lines


@dataclass
class TestSpec:
    """Complete test specification for a function/class (LLM output)."""
    test_type: TestType
    target_file: str
    target_name: str                # Function or class name
    target_class: str | None        # If testing a method
    language: Language
    test_cases: list[TestCase]

    # Optional enrichments
    fixtures_needed: list[str] = field(default_factory=list)
    imports_needed: list[str] = field(default_factory=list)
    parametrize: bool = False       # Use pytest.mark.parametrize

    # Metadata
    complexity_score: int = 1       # 1-10, affects template selection
    requires_custom_generation: bool = False  # If True, use LLM fallback

    # NEW: Semantic understanding for smart template selection
    return_semantics: ReturnSemantics = ReturnSemantics.VALUE
    requires_context: list[str] = field(default_factory=list)  # ["flask_app", "request"]
    framework_hints: list[str] = field(default_factory=list)   # ["flask", "werkzeug"]
    is_generator: bool = False                                  # Function uses yield
    mutates_args: bool = False                                  # Function modifies inputs


# ============================================================================
# Execution Models (Phase 5 Output)
# ============================================================================

@dataclass
class TestResult:
    """Result of running a single test."""
    test_name: str
    passed: bool
    error_message: str | None = None
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


@dataclass
class CoverageReport:
    """Coverage report for generated tests."""
    total_statements: int
    covered_statements: int
    missing_lines: list[int]
    coverage_percent: float
    file_path: str

    @property
    def meets_threshold(self) -> bool:
        return self.coverage_percent >= 80.0


@dataclass
class ExecutionResult:
    """Complete execution result for a test file."""
    test_file: str
    target_file: str
    test_results: list[TestResult]
    coverage: CoverageReport | None
    total_tests: int
    passed_tests: int
    failed_tests: int

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0

    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100


# ============================================================================
# Quality Gate Models (Phase 6 Output)
# ============================================================================

@dataclass
class SonarIssue:
    """A SonarQube-style issue."""
    rule_id: str
    severity: str  # "critical", "major", "minor", "info"
    message: str
    file_path: str
    line: int
    issue_type: str  # "bug", "vulnerability", "code_smell"


@dataclass
class QualityReport:
    """Complete quality gate report."""
    passed: bool
    coverage_met: bool
    coverage_percent: float
    sonar_issues: list[SonarIssue]
    critical_issues: int
    complexity_issues: int
    test_results: list[ExecutionResult]

    # Summary
    total_tests_generated: int
    total_tests_passed: int
    files_analyzed: int

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"""
Quality Gate: {status}
Coverage: {self.coverage_percent:.1f}% (threshold: 80%)
Tests: {self.total_tests_passed}/{self.total_tests_generated} passed
Critical Issues: {self.critical_issues}
Files Analyzed: {self.files_analyzed}
"""


# ============================================================================
# Agent State Models (ReAct Loop)
# ============================================================================

@dataclass
class AgentState:
    """State of the test generation agent."""
    iteration: int = 0
    max_iterations: int = 3
    current_phase: str = "init"

    # Accumulated data
    context: PRContext | None = None
    patterns: ProjectTestPatterns | None = None
    test_specs: list[TestSpec] = field(default_factory=list)
    execution_results: list[ExecutionResult] = field(default_factory=list)
    quality_report: QualityReport | None = None

    # Tracking
    coverage_history: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def should_continue(self) -> bool:
        """Check if we should continue iterating."""
        if self.iteration >= self.max_iterations:
            return False
        if self.quality_report and self.quality_report.passed:
            return False
        return True
