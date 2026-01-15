"""Models for TestSpec DSL and core pipeline models."""

# Re-export DSL models with Dsl prefix to avoid conflict with existing TestSpec
from .test_spec import (
    TestSpec as DslTestSpec,
    TestCaseSpec as DslTestCaseSpec,
    MockSpec as DslMockSpec,
    MockSetup,
    Assertion,
    MethodCall,
    get_testspec_prompt_schema,
)

# Re-export all original models from models.py (now in _core.py)
# These are the models used throughout the pipeline
import importlib.util
import sys
from pathlib import Path

# Load the original models.py as a module
_models_path = Path(__file__).parent.parent / "models.py"
if _models_path.exists():
    _spec = importlib.util.spec_from_file_location("_core_models", _models_path)
    _core_models = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_core_models)

    # Export all public names from original models
    from types import ModuleType
    for name in dir(_core_models):
        if not name.startswith('_'):
            obj = getattr(_core_models, name)
            if not isinstance(obj, ModuleType):
                globals()[name] = obj

__all__ = [
    # DSL models (prefixed to avoid conflict)
    "DslTestSpec",
    "DslTestCaseSpec",
    "DslMockSpec",
    "MockSetup",
    "Assertion",
    "MethodCall",
    "get_testspec_prompt_schema",
    # Core models (from original models.py)
    "TestType",
    "TestCategory",
    "Language",
    "CoverageStatus",
    "ErrorCategory",
    "CodeLocation",
    "Parameter",
    "ReturnSemantics",
    "FunctionContext",
    "ClassContext",
    "FixAttempt",
    "FileChange",
    "PRContext",
    "TestPattern",
    "ProjectTestPatterns",
    "ExpectedOutput",
    "MockSpec",
    "TestCase",
    "TestSpec",
    "TestResult",
    "CoverageReport",
    "ExecutionResult",
    "SonarIssue",
    "QualityReport",
    "AgentState",
]
