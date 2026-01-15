"""
Unit tests for the SpecValidator component.

Tests validation and sanitization of TestSpec data before rendering.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validator import SpecValidator, validate_and_sanitize, ValidationResult
from src.models import (
    Language, TestType, TestCategory,
    TestSpec, TestCase, ExpectedOutput, MockSpec
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def validator():
    return SpecValidator()


@pytest.fixture
def valid_java_spec():
    """A valid Java TestSpec."""
    return TestSpec(
        test_type=TestType.UNIT_CLASS,
        target_file="com/example/Example.java",
        target_name="getValue",
        target_class="Example",
        language=Language.JAVA,
        test_cases=[
            TestCase(
                name="test_getValue_returns_value",
                category=TestCategory.HAPPY_PATH,
                description="Test getValue returns the value",
                inputs={"value": "42"},
                expected=ExpectedOutput(returns="42")
            )
        ],
        imports_needed=[
            "import org.junit.jupiter.api.Test",
            "import static org.junit.jupiter.api.Assertions.*"
        ]
    )


# ============================================================================
# Structure Validation Tests
# ============================================================================

class TestValidatorStructure:
    """Tests for basic structure validation."""

    def test_valid_spec_passes(self, validator, valid_java_spec):
        """Test that a valid spec passes validation."""
        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.sanitized_spec is not None

    def test_missing_target_name_fails(self, validator, valid_java_spec):
        """Test that missing target_name fails validation."""
        valid_java_spec.target_name = ""

        result = validator.validate(valid_java_spec)

        assert result.is_valid is False
        assert any("target_name" in issue for issue in result.issues)

    def test_missing_target_file_fails(self, validator, valid_java_spec):
        """Test that missing target_file fails validation."""
        valid_java_spec.target_file = ""

        result = validator.validate(valid_java_spec)

        assert result.is_valid is False
        assert any("target_file" in issue for issue in result.issues)

    def test_none_test_cases_fails(self, validator, valid_java_spec):
        """Test that None test_cases fails validation."""
        valid_java_spec.test_cases = None

        result = validator.validate(valid_java_spec)

        assert result.is_valid is False
        assert any("test_cases" in issue for issue in result.issues)

    def test_empty_test_cases_fails(self, validator, valid_java_spec):
        """Test that empty test_cases fails validation."""
        valid_java_spec.test_cases = []

        result = validator.validate(valid_java_spec)

        assert result.is_valid is False
        assert any("empty" in issue for issue in result.issues)

    def test_test_case_missing_name_fails(self, validator, valid_java_spec):
        """Test that test case without name fails validation."""
        valid_java_spec.test_cases[0].name = ""

        result = validator.validate(valid_java_spec)

        assert result.is_valid is False
        assert any("name" in issue for issue in result.issues)


# ============================================================================
# Python Syntax Sanitization Tests
# ============================================================================

class TestValidatorPythonSanitization:
    """Tests for Python→Java syntax sanitization."""

    def test_sanitize_none_to_null(self, validator, valid_java_spec):
        """Test that Python None is converted to Java null."""
        valid_java_spec.test_cases[0].inputs = {"value": "None"}
        valid_java_spec.test_cases[0].expected.returns = "None"

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].inputs["value"] == "null"
        assert result.sanitized_spec.test_cases[0].expected.returns == "null"
        assert any("Sanitized" in w for w in result.warnings)

    def test_sanitize_true_to_lowercase(self, validator, valid_java_spec):
        """Test that Python True is converted to Java true."""
        valid_java_spec.test_cases[0].inputs = {"flag": "True"}
        valid_java_spec.test_cases[0].expected.returns = "True"

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].inputs["flag"] == "true"
        assert result.sanitized_spec.test_cases[0].expected.returns == "true"

    def test_sanitize_false_to_lowercase(self, validator, valid_java_spec):
        """Test that Python False is converted to Java false."""
        valid_java_spec.test_cases[0].inputs = {"flag": "False"}

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].inputs["flag"] == "false"

    def test_sanitize_python_dict_to_null(self, validator, valid_java_spec):
        """Test that Python dict syntax is converted to null."""
        valid_java_spec.test_cases[0].inputs = {"data": "{'key': 'value'}"}

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].inputs["data"] == "null"

    def test_sanitize_preserves_valid_java(self, validator, valid_java_spec):
        """Test that valid Java values are preserved."""
        valid_java_spec.test_cases[0].inputs = {
            "str": '"hello"',
            "num": "42",
            "obj": "new Example()"
        }

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].inputs["str"] == '"hello"'
        assert result.sanitized_spec.test_cases[0].inputs["num"] == "42"
        assert result.sanitized_spec.test_cases[0].inputs["obj"] == "new Example()"


# ============================================================================
# Import Validation Tests
# ============================================================================

class TestValidatorImports:
    """Tests for import validation and filtering."""

    def test_valid_java_imports_preserved(self, validator, valid_java_spec):
        """Test that valid Java imports are preserved."""
        valid_java_spec.imports_needed = [
            "import org.junit.jupiter.api.Test",
            "import static org.junit.jupiter.api.Assertions.*",
            "import java.util.List",
            "import com.example.Model"
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert len(result.sanitized_spec.imports_needed) == 4

    def test_python_from_import_removed(self, validator, valid_java_spec):
        """Test that Python 'from x import y' is removed."""
        valid_java_spec.imports_needed = [
            "import org.junit.jupiter.api.Test",
            "from flask import Flask",  # Python import
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert len(result.sanitized_spec.imports_needed) == 1
        assert "flask" not in str(result.sanitized_spec.imports_needed)
        assert any("Removed invalid import" in w for w in result.warnings)

    def test_python_pytest_import_removed(self, validator, valid_java_spec):
        """Test that Python pytest import is removed."""
        valid_java_spec.imports_needed = [
            "import org.junit.jupiter.api.Test",
            "import pytest",
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert "pytest" not in str(result.sanitized_spec.imports_needed)

    def test_python_unittest_import_removed(self, validator, valid_java_spec):
        """Test that Python unittest import is removed."""
        valid_java_spec.imports_needed = [
            "import org.junit.jupiter.api.Test",
            "import unittest",
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert "unittest" not in str(result.sanitized_spec.imports_needed)

    def test_mixed_valid_invalid_imports(self, validator, valid_java_spec):
        """Test filtering of mixed imports."""
        valid_java_spec.imports_needed = [
            "import org.junit.jupiter.api.Test",       # Valid
            "from flask import Flask",                  # Invalid
            "import static org.mockito.Mockito.*",     # Valid
            "import numpy as np",                       # Invalid
            "import java.util.ArrayList",               # Valid
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        imports = result.sanitized_spec.imports_needed
        assert len(imports) == 3
        assert any("junit" in i for i in imports)
        assert any("mockito" in i for i in imports)
        assert any("ArrayList" in i for i in imports)


# ============================================================================
# Mock Sanitization Tests
# ============================================================================

class TestValidatorMocks:
    """Tests for mock specification sanitization."""

    def test_sanitize_mock_return_value(self, validator, valid_java_spec):
        """Test that mock return values are sanitized."""
        valid_java_spec.test_cases[0].mocks = [
            MockSpec(target="service.findById", return_value="None")
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert result.sanitized_spec.test_cases[0].mocks[0].return_value == "null"

    def test_preserve_valid_mock(self, validator, valid_java_spec):
        """Test that valid mocks are preserved."""
        valid_java_spec.test_cases[0].mocks = [
            MockSpec(target="repository.findById", return_value="new User()")
        ]

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        mock = result.sanitized_spec.test_cases[0].mocks[0]
        assert mock.target == "repository.findById"
        assert mock.return_value == "new User()"


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestValidateAndSanitize:
    """Tests for the convenience function."""

    def test_convenience_function_works(self, valid_java_spec):
        """Test validate_and_sanitize convenience function."""
        result = validate_and_sanitize(valid_java_spec)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.sanitized_spec is not None

    def test_convenience_function_catches_errors(self):
        """Test that convenience function catches validation errors."""
        invalid_spec = TestSpec(
            test_type=TestType.UNIT_CLASS,
            target_file="",
            target_name="",
            target_class=None,
            language=Language.JAVA,
            test_cases=[]
        )

        result = validate_and_sanitize(invalid_spec)

        assert result.is_valid is False
        assert len(result.issues) > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestValidatorEdgeCases:
    """Edge case tests for validator."""

    def test_none_inputs_handled(self, validator, valid_java_spec):
        """Test that None inputs are handled gracefully."""
        valid_java_spec.test_cases[0].inputs = None

        result = validator.validate(valid_java_spec)

        # Should not fail - inputs will be converted to empty dict
        assert result.is_valid is True
        # Sanitized spec should have empty dict for inputs
        assert result.sanitized_spec.test_cases[0].inputs == {}

    def test_none_expected_handled(self, validator, valid_java_spec):
        """Test that None expected is handled gracefully."""
        valid_java_spec.test_cases[0].expected = None

        result = validator.validate(valid_java_spec)

        # Should warn but not fail
        assert result.is_valid is True

    def test_deep_copy_preserves_structure(self, validator, valid_java_spec):
        """Test that sanitization creates a copy, not modifying original."""
        original_inputs = dict(valid_java_spec.test_cases[0].inputs)
        valid_java_spec.test_cases[0].inputs["value"] = "None"

        result = validator.validate(valid_java_spec)

        # Original should be unchanged
        assert valid_java_spec.test_cases[0].inputs["value"] == "None"
        # Sanitized should be changed
        assert result.sanitized_spec.test_cases[0].inputs["value"] == "null"

    def test_multiple_test_cases(self, validator, valid_java_spec):
        """Test validation with multiple test cases."""
        valid_java_spec.test_cases.append(
            TestCase(
                name="test_second",
                category=TestCategory.EDGE_CASE,
                description="Second test",
                inputs={"x": "True"},
                expected=ExpectedOutput(returns="False")
            )
        )

        result = validator.validate(valid_java_spec)

        assert result.is_valid is True
        assert len(result.sanitized_spec.test_cases) == 2
        assert result.sanitized_spec.test_cases[1].inputs["x"] == "true"
        assert result.sanitized_spec.test_cases[1].expected.returns == "false"
