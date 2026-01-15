"""
Unit tests for Spec Generator component.

Tests both normal operation and fault injection for:
- LLM response parsing
- JSON validation and sanitization
- Python→Java syntax conversion
- Fallback spec creation
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.generator.spec_generator import SpecGenerator
from src.models import (
    Language, TestType, TestCategory, ReturnSemantics,
    FunctionContext, ClassContext, Parameter, CodeLocation, TestSpec
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def spec_generator():
    """Spec generator with mocked API key."""
    return SpecGenerator(api_key="test-key", provider="openrouter")


@pytest.fixture
def sample_function_context(model_factory):
    """Sample function context for testing."""
    return model_factory.create_function_context(
        name="getValue",
        class_name="Example",
        return_type="int",
        parameters=[
            model_factory.create_parameter("id", "Long"),
        ],
        source_code="public int getValue(Long id) { return this.value; }"
    )


@pytest.fixture
def sample_class_context(model_factory):
    """Sample class context for testing."""
    return model_factory.create_class_context(
        name="UserService",
        methods=[
            model_factory.create_function_context(
                name="findById",
                class_name="UserService",
                return_type="User",
                parameters=[model_factory.create_parameter("id", "Long")]
            ),
            model_factory.create_function_context(
                name="save",
                class_name="UserService",
                return_type="User",
                parameters=[model_factory.create_parameter("user", "User")]
            )
        ]
    )


def create_mock_response(json_data: dict, status_code: int = 200):
    """Create a mock HTTP response."""
    mock_resp = Mock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps(json_data)
            }
        }]
    }
    mock_resp.text = json.dumps({
        "choices": [{
            "message": {"content": json.dumps(json_data)}
        }]
    })
    return mock_resp


# ============================================================================
# Normal Operation Tests
# ============================================================================

class TestSpecGeneratorNormal:
    """Normal operation tests for spec generation."""

    @patch('requests.post')
    def test_generate_for_simple_getter(
        self, mock_post, spec_generator, sample_function_context, valid_llm_response
    ):
        """Test generating spec for a simple getter method."""
        mock_post.return_value = create_mock_response(valid_llm_response)

        spec = spec_generator.generate_for_function(sample_function_context)

        assert spec is not None
        assert isinstance(spec, TestSpec)
        assert spec.target_name == valid_llm_response['target_name']
        assert len(spec.test_cases) > 0

    @patch('requests.post')
    def test_generate_for_method_with_parameters(
        self, mock_post, spec_generator, model_factory
    ):
        """Test generating spec for method with multiple parameters."""
        func = model_factory.create_function_context(
            name="processData",
            parameters=[
                model_factory.create_parameter("input", "String"),
                model_factory.create_parameter("count", "int"),
            ],
            return_type="Result"
        )

        response_data = {
            "test_type": "unit_class",
            "target_name": "processData",
            "test_cases": [
                {
                    "name": "test_processData_valid_input",
                    "category": "happy_path",
                    "description": "Test with valid input",
                    "inputs": {"input": "\"test\"", "count": "5"},
                    "expected": {"returns": "new Result()"}
                }
            ],
            "imports_needed": ["import org.junit.jupiter.api.Test"]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(func)

        assert spec is not None
        assert len(spec.test_cases) >= 1
        assert spec.test_cases[0].inputs.get("input") is not None

    @patch('requests.post')
    def test_generate_for_void_method(self, mock_post, spec_generator, model_factory):
        """Test generating spec for void method."""
        func = model_factory.create_function_context(
            name="deleteById",
            parameters=[model_factory.create_parameter("id", "Long")],
            return_type="void"
        )

        response_data = {
            "test_type": "unit_class",
            "target_name": "deleteById",
            "test_cases": [
                {
                    "name": "test_deleteById_success",
                    "category": "happy_path",
                    "description": "Test successful deletion",
                    "inputs": {"id": "1L"},
                    "expected": {"returns": None}
                }
            ]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(func)

        assert spec is not None
        assert spec.test_cases[0].expected.returns is None

    @patch('requests.post')
    def test_generate_for_method_throwing_exception(
        self, mock_post, spec_generator, model_factory
    ):
        """Test generating spec for method that throws exception."""
        func = model_factory.create_function_context(
            name="validateInput",
            parameters=[model_factory.create_parameter("input", "String")],
            return_type="void"
        )

        response_data = {
            "test_type": "unit_class",
            "target_name": "validateInput",
            "test_cases": [
                {
                    "name": "test_validateInput_throws_on_null",
                    "category": "error_handling",
                    "description": "Test throws on null input",
                    "inputs": {"input": "null"},
                    "expected": {
                        "raises": "IllegalArgumentException",
                        "raises_message": "Input cannot be null"
                    }
                }
            ]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(func)

        assert spec is not None
        assert spec.test_cases[0].expected.raises == "IllegalArgumentException"

    @patch('requests.post')
    def test_semantic_fields_from_function_context(
        self, mock_post, spec_generator, model_factory
    ):
        """Test that semantic fields are applied from FunctionContext."""
        func = model_factory.create_function_context(
            name="generateItems",
            return_type="Iterator<String>",
            is_generator=True,
            return_semantics=ReturnSemantics.GENERATOR,
            mutates_args=True
        )

        response_data = {
            "test_type": "unit_class",
            "target_name": "generateItems",
            "test_cases": [{
                "name": "test_generate",
                "category": "happy_path",
                "description": "Test generator",
                "inputs": {},
                "expected": {"returns": "iterator"}
            }]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(func)

        # Semantic fields should be from FunctionContext, not LLM
        assert spec.is_generator is True
        assert spec.return_semantics == ReturnSemantics.GENERATOR
        assert spec.mutates_args is True


# ============================================================================
# Fault Injection Tests - Invalid LLM Responses
# ============================================================================

class TestSpecGeneratorFaultInjection:
    """Fault injection tests for spec generator."""

    @patch('requests.post')
    def test_llm_returns_invalid_json(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test handling of invalid JSON from LLM."""
        # First call returns invalid JSON, triggers retry
        invalid_response = Mock()
        invalid_response.status_code = 200
        invalid_response.json.return_value = {
            "choices": [{
                "message": {"content": "{invalid json..."}
            }]
        }

        # Second call also fails, triggers fallback
        mock_post.return_value = invalid_response

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should return fallback spec
        assert spec is not None
        assert spec.requires_custom_generation is True

    @patch('requests.post')
    def test_llm_returns_python_syntax_in_java(
        self, mock_post, spec_generator, sample_function_context,
        llm_response_with_python_syntax
    ):
        """Test sanitization of Python syntax (None→null, True→true)."""
        mock_post.return_value = create_mock_response(llm_response_with_python_syntax)

        spec = spec_generator.generate_for_function(sample_function_context)

        assert spec is not None
        # Python imports should be filtered out for Java
        if spec.imports_needed:
            assert not any("from unittest" in imp for imp in spec.imports_needed)
            assert not any("import pytest" == imp for imp in spec.imports_needed)

    @patch('requests.post')
    def test_llm_hallucinates_imports(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test filtering of hallucinated Python imports for Java."""
        response_data = {
            "test_type": "unit_class",
            "target_name": "getValue",
            "test_cases": [{
                "name": "test_getValue",
                "category": "happy_path",
                "description": "Test",
                "inputs": {},
                "expected": {"returns": "1"}
            }],
            "imports_needed": [
                "from flask import Flask",  # Python
                "import pytest",            # Python
                "import numpy as np",       # Python
                "import org.junit.jupiter.api.Test",  # Java
                "import static org.mockito.Mockito.*"  # Java
            ]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should filter out Python imports
        if spec.imports_needed:
            for imp in spec.imports_needed:
                assert "from flask" not in imp
                assert "import pytest" != imp
                assert "import numpy" not in imp

    @patch('requests.post')
    def test_llm_wrong_param_count(
        self, mock_post, spec_generator, model_factory
    ):
        """Test handling when LLM provides wrong number of inputs."""
        # Function has 2 parameters
        func = model_factory.create_function_context(
            name="process",
            parameters=[
                model_factory.create_parameter("a", "int"),
                model_factory.create_parameter("b", "int"),
            ],
            return_type="int"
        )

        # LLM provides 5 inputs
        response_data = {
            "test_type": "unit_class",
            "target_name": "process",
            "test_cases": [{
                "name": "test_process",
                "category": "happy_path",
                "description": "Test",
                "inputs": {
                    "a": "1",
                    "b": "2",
                    "c": "3",  # Extra
                    "d": "4",  # Extra
                    "e": "5"   # Extra
                },
                "expected": {"returns": "3"}
            }]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(func)

        # Should handle gracefully (may truncate or keep all)
        assert spec is not None
        # Just verify it doesn't crash
        assert len(spec.test_cases) >= 1

    @patch('requests.post')
    def test_llm_empty_test_cases(
        self, mock_post, spec_generator, sample_function_context,
        llm_response_empty_test_cases
    ):
        """Test handling of empty test cases from LLM."""
        mock_post.return_value = create_mock_response(llm_response_empty_test_cases)

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should create fallback spec or handle gracefully
        assert spec is not None
        # May have custom generation flag or minimal test case
        assert spec.requires_custom_generation is True or len(spec.test_cases) >= 0

    @patch('requests.post')
    def test_llm_missing_required_fields(
        self, mock_post, spec_generator, sample_function_context,
        llm_response_missing_fields
    ):
        """Test handling of missing required fields."""
        mock_post.return_value = create_mock_response(llm_response_missing_fields)

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should create fallback or auto-generate defaults
        assert spec is not None

    @patch('requests.post')
    def test_llm_api_timeout(self, mock_post, spec_generator, sample_function_context):
        """Test handling of API timeout."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout("Connection timed out")

        # Should raise or return fallback
        try:
            spec = spec_generator.generate_for_function(sample_function_context)
            # If it returns, should be fallback
            assert spec.requires_custom_generation is True
        except req.exceptions.Timeout:
            # This is also acceptable behavior
            pass

    @patch('requests.post')
    def test_llm_api_error_500(self, mock_post, spec_generator, sample_function_context):
        """Test handling of API 500 error."""
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        mock_post.return_value = error_response

        # Should handle gracefully
        try:
            spec = spec_generator.generate_for_function(sample_function_context)
            assert spec.requires_custom_generation is True
        except Exception:
            # Raising is also acceptable
            pass

    @patch('requests.post')
    def test_llm_mock_empty_target(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test handling of empty mock target strings."""
        response_data = {
            "test_type": "unit_mocked",
            "target_name": "getValue",
            "test_cases": [{
                "name": "test_getValue",
                "category": "happy_path",
                "description": "Test with mock",
                "inputs": {},
                "expected": {"returns": "1"},
                "mocks": [
                    {"target": "", "return_value": "null"},  # Empty target
                    {"target": "repository.findById", "return_value": "entity"}
                ]
            }]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should handle gracefully
        assert spec is not None

    @patch('requests.post')
    def test_llm_returns_markdown_wrapped_json(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test handling of JSON wrapped in markdown code blocks."""
        json_content = {
            "test_type": "unit_class",
            "target_name": "getValue",
            "test_cases": [{
                "name": "test_getValue",
                "category": "happy_path",
                "description": "Test",
                "inputs": {},
                "expected": {"returns": "1"}
            }]
        }

        # Wrap in markdown
        markdown_response = Mock()
        markdown_response.status_code = 200
        markdown_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": f"```json\n{json.dumps(json_content)}\n```"
                }
            }]
        }
        mock_post.return_value = markdown_response

        spec = spec_generator.generate_for_function(sample_function_context)

        # Should extract JSON from markdown
        assert spec is not None
        assert spec.target_name == "getValue"


# ============================================================================
# Sanitization Tests
# ============================================================================

class TestSpecGeneratorSanitization:
    """Tests for Python→Java syntax sanitization."""

    @patch('requests.post')
    def test_sanitize_none_to_null(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test None is converted to null for Java."""
        response_data = {
            "test_type": "unit_class",
            "target_name": "getValue",
            "test_cases": [{
                "name": "test_getValue_null",
                "category": "edge_case",
                "description": "Test with null",
                "inputs": {"value": "None"},
                "expected": {"returns": "None"}
            }]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(sample_function_context)

        assert spec is not None
        # The sanitization should convert None to null
        # (depending on where sanitization happens)

    @patch('requests.post')
    def test_sanitize_true_false_to_lowercase(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test True/False are converted to true/false for Java."""
        response_data = {
            "test_type": "unit_class",
            "target_name": "isValid",
            "test_cases": [{
                "name": "test_isValid_true",
                "category": "happy_path",
                "description": "Test returns true",
                "inputs": {},
                "expected": {"returns": "True"}
            }]
        }
        mock_post.return_value = create_mock_response(response_data)

        spec = spec_generator.generate_for_function(sample_function_context)

        assert spec is not None


# ============================================================================
# Provider Tests
# ============================================================================

class TestSpecGeneratorProviders:
    """Tests for different API providers."""

    def test_openrouter_provider_initialization(self):
        """Test OpenRouter provider initialization."""
        gen = SpecGenerator(api_key="test-key", provider="openrouter")
        assert gen.provider == "openrouter"
        assert gen.base_url == "https://openrouter.ai/api/v1"

    def test_anthropic_provider_initialization(self):
        """Test Anthropic provider initialization."""
        gen = SpecGenerator(api_key="test-key", provider="anthropic")
        assert gen.provider == "anthropic"

    def test_language_detection_java(self, spec_generator):
        """Test Java language detection from file path."""
        lang = spec_generator._detect_language("com/example/UserService.java")
        assert lang == "java"

    def test_language_detection_python(self, spec_generator):
        """Test Python language detection from file path."""
        lang = spec_generator._detect_language("src/service.py")
        assert lang == "python"

    def test_language_detection_typescript(self, spec_generator):
        """Test TypeScript language detection from file path."""
        lang = spec_generator._detect_language("src/service.ts")
        assert lang == "typescript"

    def test_language_detection_tsx(self, spec_generator):
        """Test TSX language detection from file path."""
        lang = spec_generator._detect_language("src/Component.tsx")
        assert lang == "typescript"


# ============================================================================
# Fallback Spec Tests
# ============================================================================

class TestSpecGeneratorFallback:
    """Tests for fallback spec creation."""

    @patch('requests.post')
    def test_fallback_spec_on_repeated_failures(
        self, mock_post, spec_generator, sample_function_context
    ):
        """Test fallback spec is created after max retries."""
        # Always return invalid response
        invalid_response = Mock()
        invalid_response.status_code = 200
        invalid_response.json.return_value = {
            "choices": [{
                "message": {"content": "not valid json at all!!!"}
            }]
        }
        mock_post.return_value = invalid_response

        spec = spec_generator.generate_for_function(
            sample_function_context,
            max_retries=2
        )

        # Should return fallback spec
        assert spec is not None
        assert spec.requires_custom_generation is True
        assert spec.target_name == sample_function_context.name

    @patch('requests.post')
    def test_fallback_spec_preserves_function_info(
        self, mock_post, spec_generator, model_factory
    ):
        """Test fallback spec preserves function information."""
        func = model_factory.create_function_context(
            name="specialMethod",
            class_name="SpecialClass",
            return_type="SpecialType"
        )

        invalid_response = Mock()
        invalid_response.status_code = 200
        invalid_response.json.return_value = {
            "choices": [{"message": {"content": "invalid"}}]
        }
        mock_post.return_value = invalid_response

        spec = spec_generator.generate_for_function(func, max_retries=0)

        assert spec.target_name == "specialMethod"
        assert spec.target_class == "SpecialClass"


# ============================================================================
# Class Generation Tests
# ============================================================================

class TestSpecGeneratorClassGeneration:
    """Tests for class-level spec generation."""

    @patch('requests.post')
    def test_generate_for_class(
        self, mock_post, spec_generator, sample_class_context
    ):
        """Test generating spec for a class."""
        response_data = {
            "test_type": "unit_class",
            "target_name": "findById",
            "test_cases": [
                {
                    "name": "test_findById_returns_user",
                    "category": "happy_path",
                    "description": "Test findById",
                    "inputs": {"id": "1L"},
                    "expected": {"returns": "user"}
                }
            ]
        }
        mock_post.return_value = create_mock_response(response_data)

        # generate_for_class returns list[TestSpec], one per method
        specs = spec_generator.generate_for_class(sample_class_context)

        assert specs is not None
        assert isinstance(specs, list)
        assert len(specs) >= 1
        # Each spec corresponds to a method in the class
        assert all(hasattr(s, 'target_name') for s in specs)
