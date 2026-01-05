"""
Spec Generator - LLM-powered test specification generation.

Uses structured output to ensure valid JSON that matches our schema.
Supports both Anthropic API and OpenRouter API.
"""

import json
import os
import re
import requests
from pathlib import Path

from ..models import (
    FunctionContext, ClassContext, TestSpec, TestCase, TestType,
    TestCategory, ExpectedOutput, MockSpec, Language, ProjectTestPatterns,
    ReturnSemantics
)
from .prompts import TestPromptBuilder


class SpecGenerator:
    """
    Generate test specifications using LLM.

    The LLM outputs structured JSON, which is validated and converted
    to TestSpec objects for template rendering.

    Supports:
    - Anthropic API (default)
    - OpenRouter API (for cheaper models like Claude Haiku)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic"  # "anthropic" or "openrouter"
    ):
        self.provider = provider
        self.model = model
        self._client = None

        # Get appropriate API key
        if provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.base_url = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client (only for anthropic provider)."""
        if self.provider == "openrouter":
            return None  # OpenRouter uses requests directly

        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def _detect_language(self, file_path: str) -> str:
        """Detect the programming language from the file extension."""
        path = file_path.lower()
        if path.endswith('.java'):
            return 'java'
        elif path.endswith('.ts') or path.endswith('.tsx'):
            return 'typescript'
        elif path.endswith('.js') or path.endswith('.jsx'):
            return 'javascript'
        else:
            return 'python'

    def generate_for_function(
        self,
        func: FunctionContext,
        patterns: ProjectTestPatterns | None = None,
        max_retries: int = 2
    ) -> TestSpec:
        """
        Generate test specification for a function.

        Args:
            func: The function context to generate tests for
            patterns: Optional project patterns for context
            max_retries: Number of retries on validation failure

        Returns:
            TestSpec object ready for template rendering
        """
        prompt = TestPromptBuilder.build_function_prompt(func, patterns)

        # Detect language from file path
        language = self._detect_language(str(func.location.file_path))

        # Store param count for input validation
        param_count = len(func.parameters)

        for attempt in range(max_retries + 1):
            try:
                response = self._call_llm(prompt, language)
                spec_dict = self._parse_json_response(response, language, param_count)
                spec = self._dict_to_test_spec(spec_dict)

                # Ensure semantic fields from FunctionContext override LLM response
                # (FunctionContext detection is more reliable)
                spec = self._apply_function_semantics(spec, func)

                return spec
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries:
                    # Add error context to prompt for retry
                    prompt = self._add_error_context(prompt, str(e))
                else:
                    # Return a minimal spec that triggers custom generation
                    return self._create_fallback_spec(func, str(e))

    def _apply_function_semantics(self, spec: TestSpec, func: FunctionContext) -> TestSpec:
        """Apply semantic information from FunctionContext to TestSpec.

        FunctionContext detection is more reliable than LLM guessing,
        so we override the LLM's semantic fields with detected values.
        """
        # Override with detected values (more reliable)
        spec.is_generator = func.is_generator
        spec.return_semantics = func.return_semantics
        spec.mutates_args = func.mutates_args

        # Merge context requirements (keep LLM's additions but include detected)
        detected_contexts = set(func.requires_context)
        llm_contexts = set(spec.requires_context)
        spec.requires_context = list(detected_contexts | llm_contexts)

        # Merge framework hints
        detected_frameworks = set(func.framework_hints)
        llm_frameworks = set(spec.framework_hints)
        spec.framework_hints = list(detected_frameworks | llm_frameworks)

        return spec

    def generate_for_class(
        self,
        cls: ClassContext,
        patterns: ProjectTestPatterns | None = None
    ) -> list[TestSpec]:
        """Generate test specifications for all methods in a class."""
        specs = []

        for method in cls.methods:
            spec = self.generate_for_function(method, patterns)
            specs.append(spec)

        return specs

    def generate_additional_tests(
        self,
        func: FunctionContext,
        current_coverage: float,
        missing_lines: list[int],
        existing_spec: TestSpec
    ) -> list[TestCase]:
        """Generate additional test cases to improve coverage."""
        prompt = TestPromptBuilder.build_coverage_improvement_prompt(
            func, current_coverage, missing_lines,
            self._test_spec_to_dict(existing_spec)
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)
            additional_cases = result.get('additional_test_cases', [])
            return [self._dict_to_test_case(tc) for tc in additional_cases]
        except Exception:
            return []

    def generate_custom(self, func: FunctionContext, reason: str) -> str:
        """
        Generate custom test code when templates don't fit.

        Returns raw Python test code instead of a spec.
        """
        prompt = TestPromptBuilder.build_custom_generation_prompt(func, reason)

        response = self._call_llm(prompt)

        # Extract code from response
        if '```python' in response:
            code = response.split('```python')[1].split('```')[0]
        else:
            code = response

        return code.strip()

    def _call_llm(self, prompt: str, language: str = "python") -> str:
        """Call the LLM API (supports Anthropic and OpenRouter)."""
        if self.provider == "openrouter":
            return self._call_openrouter(prompt, language)
        else:
            return self._call_anthropic(prompt, language)

    def _call_anthropic(self, prompt: str, language: str = "python") -> str:
        """Call Anthropic API directly."""
        system_prompt = TestPromptBuilder.get_system_prompt(language)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

    def _call_openrouter(self, prompt: str, language: str = "python") -> str:
        """Call OpenRouter API for access to various models."""
        system_prompt = TestPromptBuilder.get_system_prompt(language)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pipeline-test-agent",
            "X-Title": "Pipeline Test Agent"
        }

        payload = {
            "model": self.model,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]

    # Known hallucinated class names to replace with Mock()
    HALLUCINATED_PATTERNS = [
        'MockLoader', 'FakeLoader', 'TestLoader', 'StubLoader',
        'MockRequest', 'FakeRequest', 'TestRequest', 'StubRequest',
        'MockResponse', 'FakeResponse', 'TestResponse', 'StubResponse',
        'MockSession', 'FakeSession', 'TestSession', 'StubSession',
        'MockClient', 'FakeClient', 'TestClient', 'StubClient',
        'MockHandler', 'FakeHandler', 'TestHandler', 'StubHandler',
        'ComplexLoader', 'SimpleLoader', 'CustomLoader',
    ]

    def _parse_json_response(self, response: str, language: str = "python", param_count: int = 0) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Clean up response
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith('```'):
            # Find the end of the first line (language specifier)
            first_newline = text.find('\n')
            text = text[first_newline + 1:]
            # Remove trailing ```
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

        # Try to parse
        data = json.loads(text)

        # Post-process to fix hallucinations (Option 4)
        data = self._sanitize_llm_output(data, language, param_count)

        # Validate and fix JSON structure (Option A)
        data, validation_errors = self._validate_and_fix_json(data)

        # Log validation errors for debugging (non-fatal)
        if validation_errors:
            print(f"  [WARN] Validation fixes applied: {len(validation_errors)}")

        return data

    # Invalid imports that LLM hallucinates
    INVALID_IMPORTS = [
        'from flask import App',  # Should be Flask, not App
        'from flask import app',  # lowercase app is an instance, not class
    ]

    # Import corrections
    IMPORT_FIXES = {
        'from flask import App': 'from flask import Flask',
        'from flask import app': 'from flask import Flask',
    }

    def _sanitize_llm_output(self, data: dict, language: str = "python", param_count: int = 0) -> dict:
        """Sanitize LLM output to fix common hallucinations."""
        # Sanitize test cases
        if 'test_cases' in data:
            data['test_cases'] = [
                self._sanitize_test_case(tc, language, param_count)
                for tc in data['test_cases']
            ]

        # Language-specific import handling
        imports = data.get('imports_needed', [])

        if language == "java":
            # Remove any Python imports that leaked through
            imports = [imp for imp in imports if not imp.startswith('from ') and not imp.startswith('import ') or imp.startswith('import org.') or imp.startswith('import java.') or imp.startswith('import static ')]

            # Check if we need Mockito imports
            for tc in data.get('test_cases', []):
                inputs_str = str(tc.get('inputs', {}))
                if 'mock(' in inputs_str.lower():
                    if not any('mockito' in imp.lower() for imp in imports):
                        imports.append('import static org.mockito.Mockito.*')
                    break
        else:
            # Python sanitization
            sanitized_imports = []
            for imp in imports:
                # Fix known bad imports
                if imp in self.IMPORT_FIXES:
                    imp = self.IMPORT_FIXES[imp]
                # Skip completely invalid imports
                if imp not in self.INVALID_IMPORTS:
                    sanitized_imports.append(imp)
            imports = sanitized_imports

            # Ensure imports_needed includes Mock if we detect Mock usage
            if not any('mock' in imp.lower() for imp in imports):
                # Check if any test case uses Mock
                for tc in data.get('test_cases', []):
                    inputs_str = str(tc.get('inputs', {}))
                    if 'Mock(' in inputs_str or any(h in inputs_str for h in self.HALLUCINATED_PATTERNS):
                        imports.append('from unittest.mock import Mock, MagicMock')
                        break

        # Remove duplicates
        seen = set()
        unique_imports = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
        imports = unique_imports

        # Sanitize fixtures_needed - remove duplicates and invalid ones
        fixtures = data.get('fixtures_needed', [])
        valid_fixtures = []
        seen_fixtures = set()
        for f in fixtures:
            if f not in seen_fixtures and f not in ['flask_app']:  # flask_app doesn't exist
                seen_fixtures.add(f)
                valid_fixtures.append(f)
        data['fixtures_needed'] = valid_fixtures

        data['imports_needed'] = imports
        return data

    def _sanitize_test_case(self, tc: dict, language: str = "python", param_count: int = 0) -> dict:
        """Sanitize a single test case to fix hallucinations."""
        # Fix inputs - replace hallucinated class names with Mock()
        if 'inputs' in tc:
            tc['inputs'] = self._sanitize_inputs(tc['inputs'], language, param_count)

        # Fix expected returns - ensure strings are properly quoted
        if 'expected' in tc and 'returns' in tc['expected']:
            tc['expected']['returns'] = self._sanitize_value(tc['expected']['returns'], language)

        return tc

    def _sanitize_inputs(self, inputs, language: str = "python", param_count: int = 0) -> dict:
        """Sanitize input values to replace hallucinated classes with Mock()."""
        # Handle case where LLM returns a list instead of dict
        if isinstance(inputs, list):
            # Convert list to dict with arg0, arg1, etc.
            inputs = {f"arg{i}": v for i, v in enumerate(inputs)}

        if not isinstance(inputs, dict):
            return {}

        sanitized = {}
        for key, value in inputs.items():
            sanitized[key] = self._sanitize_value(value, language)

        # Trim extra inputs if LLM hallucinated more than needed
        if param_count > 0 and len(sanitized) > param_count:
            # Keep only the first param_count inputs
            keys = list(sanitized.keys())[:param_count]
            sanitized = {k: sanitized[k] for k in keys}

        return sanitized

    def _sanitize_value(self, value, language: str = "python"):
        """Sanitize a single value, replacing hallucinated classes with appropriate mocks."""
        if value is None:
            return None

        # Handle Python booleans for Java
        if language == "java":
            if value is True:
                return "true"
            if value is False:
                return "false"

        if isinstance(value, str):
            # Java-specific sanitization
            if language == "java":
                # Replace Python None with null
                if value == "None":
                    return "null"

                # Replace Python booleans with Java booleans
                if value == "True":
                    return "true"
                if value == "False":
                    return "false"

                # Replace Python Mock() with Mockito mock()
                if value.startswith("Mock("):
                    # Mock(spec=Pet) -> mock(Pet.class)
                    spec_match = re.search(r'Mock\(spec=(\w+)\)', value)
                    if spec_match:
                        class_name = spec_match.group(1)
                        return f"mock({class_name}.class)"
                    return "mock(Object.class)"

                # Replace Python dict syntax with Java
                if value.startswith('{') and ':' in value:
                    # This is a Python dict, needs Java conversion
                    # For now, mark it as needing manual fix
                    return f"/* TODO: convert to Java */ {value}"

                # Fix hallucinated Python patterns
                for pattern in self.HALLUCINATED_PATTERNS:
                    if pattern in value:
                        return "mock(Object.class)"

                return value

            # Python sanitization (existing logic)
            # Check for hallucinated class instantiation patterns
            for pattern in self.HALLUCINATED_PATTERNS:
                if pattern in value:
                    # Replace with Mock()
                    # e.g., "MockLoader(name='test')" -> "Mock()"
                    return "Mock()"

            # Fix Mock(spec=ClassName) - extract class name and track for import
            # But simplify to just Mock() to avoid import issues
            mock_spec_match = re.search(r'Mock\(spec=(\w+)', value)
            if mock_spec_match:
                # Keep it simple - just use Mock() without spec to avoid import issues
                return "Mock()"

            # Check for unquoted strings that look like they should be quoted
            # e.g., "Hello, World!" without quotes in JSON context
            if '(' not in value and '.' not in value and not value.startswith('"'):
                # It's probably meant to be a string literal
                # Keep as-is, template will handle quoting
                pass

            return value

        if isinstance(value, dict):
            return {k: self._sanitize_value(v, language) for k, v in value.items()}

        if isinstance(value, list):
            return [self._sanitize_value(v, language) for v in value]

        return value

    # =========================================================================
    # JSON SCHEMA VALIDATION AND FIXING (Option A)
    # =========================================================================

    # Required fields for a valid test spec
    REQUIRED_FIELDS = ['test_type', 'target_name', 'test_cases']
    REQUIRED_TEST_CASE_FIELDS = ['name', 'inputs', 'expected']

    # Known class names that need imports (framework-specific)
    CLASS_IMPORTS = {
        'BaseLoader': 'from jinja2 import BaseLoader',
        'FileSystemLoader': 'from jinja2 import FileSystemLoader',
        'Response': 'from flask import Response',
        'Request': 'from flask import Request',
        'Flask': 'from flask import Flask',
        'Blueprint': 'from flask import Blueprint',
        'MultiDict': 'from werkzeug.datastructures import MultiDict',
        'ImmutableMultiDict': 'from werkzeug.datastructures import ImmutableMultiDict',
        'DebugFilesKeyError': 'from flask.debughelpers import DebugFilesKeyError',
    }

    def _validate_and_fix_json(self, data: dict) -> tuple[dict, list[str]]:
        """
        Validate JSON structure and fix common issues.

        Returns:
            tuple: (fixed_data, list of validation errors that couldn't be fixed)
        """
        errors = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Validate and fix test cases
        if 'test_cases' in data:
            fixed_cases = []
            for i, tc in enumerate(data['test_cases']):
                fixed_tc, tc_errors = self._validate_and_fix_test_case(tc, i)
                fixed_cases.append(fixed_tc)
                errors.extend(tc_errors)
            data['test_cases'] = fixed_cases

        # Auto-add missing imports based on usage
        data = self._auto_add_imports(data)

        return data, errors

    def _validate_and_fix_test_case(self, tc: dict, index: int) -> tuple[dict, list[str]]:
        """Validate and fix a single test case."""
        errors = []

        # Ensure required fields exist
        if 'name' not in tc:
            tc['name'] = f'test_case_{index}'
            errors.append(f"Test case {index}: Missing 'name', auto-generated")

        if 'inputs' not in tc:
            tc['inputs'] = {}

        if 'expected' not in tc:
            tc['expected'] = {'returns': None}

        # Fix inputs - ensure all string values are properly quoted for Python
        tc['inputs'] = self._fix_input_values(tc['inputs'])

        # Fix expected values
        if 'expected' in tc:
            tc['expected'] = self._fix_expected_values(tc['expected'])

        return tc, errors

    def _fix_input_values(self, inputs: dict) -> dict:
        """Fix input values for Python compatibility."""
        fixed = {}
        for key, value in inputs.items():
            fixed[key] = self._fix_value_for_python(value)
        return fixed

    def _fix_expected_values(self, expected: dict) -> dict:
        """Fix expected values for Python compatibility."""
        fixed = {}
        for key, value in expected.items():
            if key == 'returns':
                fixed[key] = self._fix_value_for_python(value)
            else:
                fixed[key] = value
        return fixed

    def _fix_value_for_python(self, value):
        """
        Fix a value to be valid Python.

        Handles:
        - JSON booleans (true/false) - these come as Python True/False from json.loads
        - Unquoted strings that should be quoted
        - Mock(spec=X) patterns
        """
        if value is None:
            return None

        # JSON booleans are already converted by json.loads to Python True/False
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            # If it looks like a function call, keep it as-is
            if '(' in value and ')' in value:
                return value

            # If it's a simple identifier (could be a variable reference)
            # that's not a known Python builtin, quote it
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value):
                # Known Python values that shouldn't be quoted
                python_keywords = {'True', 'False', 'None', 'Mock', 'MagicMock'}
                if value not in python_keywords:
                    # This is likely meant to be a string, quote it
                    return f'"{value}"'

            # If it starts with / or http, or contains special chars, it's likely a path/URL - quote it
            if value.startswith('/') or value.startswith('http') or '?' in value or '#' in value:
                return f'"{value}"'

            # If it contains spaces, it's definitely a string that needs quoting
            if ' ' in value and not value.startswith('"') and not value.startswith("'"):
                return f'"{value}"'

            return value

        if isinstance(value, dict):
            return {k: self._fix_value_for_python(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._fix_value_for_python(v) for v in value]

        return value

    def _auto_add_imports(self, data: dict) -> dict:
        """Auto-add missing imports based on class usage in test cases."""
        imports = set(data.get('imports_needed', []))

        # Always ensure Mock is imported if used
        all_values_str = json.dumps(data.get('test_cases', []))
        if 'Mock(' in all_values_str or 'MagicMock(' in all_values_str:
            imports.add('from unittest.mock import Mock, MagicMock, patch')

        # Check for class names that need imports
        for class_name, import_stmt in self.CLASS_IMPORTS.items():
            if class_name in all_values_str:
                imports.add(import_stmt)

        data['imports_needed'] = list(imports)
        return data

    def _dict_to_test_spec(self, data: dict) -> TestSpec:
        """Convert a dictionary to a TestSpec object."""
        # Map test type string to enum
        test_type_map = {
            'unit_pure': TestType.UNIT_PURE,
            'unit_class': TestType.UNIT_CLASS,
            'unit_mocked': TestType.UNIT_MOCKED,
            'integration_api': TestType.INTEGRATION_API,
            'integration_db': TestType.INTEGRATION_DB,
            'edge_case': TestType.EDGE_CASE,
            'custom': TestType.CUSTOM,
        }

        test_type = test_type_map.get(
            data.get('test_type', 'unit_pure'),
            TestType.UNIT_PURE
        )

        # Parse test cases
        test_cases = [
            self._dict_to_test_case(tc)
            for tc in data.get('test_cases', [])
        ]

        # Parse return semantics
        return_semantics_map = {
            'value': ReturnSemantics.VALUE,
            'generator': ReturnSemantics.GENERATOR,
            'iterator': ReturnSemantics.ITERATOR,
            'none_sideeffect': ReturnSemantics.NONE_SIDEEFFECT,
            'context_manager': ReturnSemantics.CONTEXT_MANAGER,
        }
        return_semantics = return_semantics_map.get(
            data.get('return_semantics', 'value'),
            ReturnSemantics.VALUE
        )

        return TestSpec(
            test_type=test_type,
            target_file=data.get('target_file', ''),
            target_name=data.get('target_name', ''),
            target_class=data.get('target_class'),
            language=Language(data.get('language', 'python')),
            test_cases=test_cases,
            fixtures_needed=data.get('fixtures_needed', []),
            imports_needed=data.get('imports_needed', []),
            parametrize=data.get('parametrize', False),
            complexity_score=data.get('complexity_score', 1),
            requires_custom_generation=data.get('requires_custom_generation', False),
            # NEW: Semantic fields
            return_semantics=return_semantics,
            requires_context=data.get('requires_context', []),
            framework_hints=data.get('framework_hints', []),
            is_generator=data.get('is_generator', False),
            mutates_args=data.get('mutates_args', False),
        )

    def _dict_to_test_case(self, data: dict) -> TestCase:
        """Convert a dictionary to a TestCase object."""
        category_map = {
            'happy_path': TestCategory.HAPPY_PATH,
            'edge_case': TestCategory.EDGE_CASE,
            'error_handling': TestCategory.ERROR_HANDLING,
            'boundary': TestCategory.BOUNDARY,
            'null_empty': TestCategory.NULL_EMPTY,
            'performance': TestCategory.PERFORMANCE,
        }

        category = category_map.get(
            data.get('category', 'happy_path'),
            TestCategory.HAPPY_PATH
        )

        # Parse expected output
        expected_data = data.get('expected', {})
        expected = ExpectedOutput(
            returns=expected_data.get('returns'),
            raises=expected_data.get('raises'),
            raises_message=expected_data.get('raises_message'),
            side_effects=expected_data.get('side_effects')
        )

        # Parse mocks
        mocks = []
        for mock_data in data.get('mocks', []):
            mocks.append(MockSpec(
                target=mock_data.get('target', ''),
                return_value=mock_data.get('return_value'),
                side_effect=mock_data.get('side_effect'),
                assert_called_with=mock_data.get('assert_called_with')
            ))

        return TestCase(
            name=data.get('name', 'unnamed_test'),
            category=category,
            description=data.get('description', ''),
            inputs=data.get('inputs', {}),
            expected=expected,
            mocks=mocks,
            setup=data.get('setup', []),
            teardown=data.get('teardown', [])
        )

    def _test_spec_to_dict(self, spec: TestSpec) -> dict:
        """Convert TestSpec back to dict for prompts."""
        return {
            'test_type': spec.test_type.value,
            'target_file': spec.target_file,
            'target_name': spec.target_name,
            'target_class': spec.target_class,
            'test_cases': [
                {
                    'name': tc.name,
                    'category': tc.category.value,
                    'description': tc.description,
                    'inputs': tc.inputs,
                    'expected': {
                        'returns': tc.expected.returns,
                        'raises': tc.expected.raises,
                    }
                }
                for tc in spec.test_cases
            ]
        }

    def _add_error_context(self, prompt: str, error: str) -> str:
        """Add error context for retry."""
        return f"""{prompt}

PREVIOUS ATTEMPT FAILED WITH ERROR:
{error}

Please fix the JSON output and try again. Ensure:
1. Valid JSON syntax (no trailing commas, proper quotes)
2. All required fields are present
3. Enum values match exactly (e.g., "happy_path" not "happyPath")
"""

    def _create_fallback_spec(self, func: FunctionContext, error: str) -> TestSpec:
        """Create a fallback spec when LLM generation fails."""
        return TestSpec(
            test_type=TestType.CUSTOM,
            target_file=str(func.location.file_path),
            target_name=func.name,
            target_class=func.class_name,
            language=Language.PYTHON,
            test_cases=[
                TestCase(
                    name=f"test_{func.name}_basic",
                    category=TestCategory.HAPPY_PATH,
                    description=f"Basic test for {func.name} (auto-generated fallback)",
                    inputs={},
                    expected=ExpectedOutput()
                )
            ],
            requires_custom_generation=True,
            complexity_score=10  # High complexity = needs custom handling
        )


class OpenRouterSpecGenerator(SpecGenerator):
    """
    Spec Generator using OpenRouter API with Claude Haiku 3.5.

    This is the most cost-effective option for test generation.
    Claude Haiku 3.5 on OpenRouter: ~$0.25/million input tokens
    """

    # OpenRouter model IDs for Claude models
    MODELS = {
        "haiku": "anthropic/claude-3.5-haiku",
        "haiku-3.5": "anthropic/claude-3.5-haiku",
        "sonnet": "anthropic/claude-3.5-sonnet",
        "sonnet-3.5": "anthropic/claude-3.5-sonnet",
    }

    def __init__(self, api_key: str | None = None, model: str = "haiku"):
        # Map friendly name to OpenRouter model ID
        model_id = self.MODELS.get(model, model)

        super().__init__(
            api_key=api_key,
            model=model_id,
            provider="openrouter"
        )

    @classmethod
    def from_env(cls, model: str = "haiku") -> "OpenRouterSpecGenerator":
        """Create generator from environment variable."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get your key from https://openrouter.ai/keys"
            )
        return cls(api_key=api_key, model=model)


class MockSpecGenerator:
    """
    Offline/mock spec generator for testing without LLM.

    Uses heuristics to generate basic test specs.
    """

    def generate_for_function(
        self,
        func: FunctionContext,
        patterns: ProjectTestPatterns | None = None
    ) -> TestSpec:
        """Generate a basic test spec using heuristics."""
        test_cases = []

        # Happy path test
        inputs = self._generate_sample_inputs(func)
        test_cases.append(TestCase(
            name=f"{func.name}_returns_expected_value",
            category=TestCategory.HAPPY_PATH,
            description=f"Test {func.name} with valid inputs",
            inputs=inputs,
            expected=ExpectedOutput(returns=None)  # Unknown without running
        ))

        # Edge case for optional parameters
        optional_params = [p for p in func.parameters if p.is_optional]
        if optional_params:
            test_cases.append(TestCase(
                name=f"{func.name}_with_defaults",
                category=TestCategory.EDGE_CASE,
                description=f"Test {func.name} using default values",
                inputs={p.name: None for p in func.parameters if not p.is_optional},
                expected=ExpectedOutput(returns=None)
            ))

        # Determine test type
        if func.calls and any('.' in call for call in func.calls):
            test_type = TestType.UNIT_MOCKED
        elif func.is_method:
            test_type = TestType.UNIT_CLASS
        else:
            test_type = TestType.UNIT_PURE

        return TestSpec(
            test_type=test_type,
            target_file=str(func.location.file_path),
            target_name=func.name,
            target_class=func.class_name,
            language=Language.PYTHON,
            test_cases=test_cases,
            complexity_score=len(func.parameters) + len(func.calls)
        )

    def _generate_sample_inputs(self, func: FunctionContext) -> dict:
        """Generate sample input values based on type hints."""
        inputs = {}

        type_samples = {
            'str': '"test_value"',
            'int': '42',
            'float': '3.14',
            'bool': 'True',
            'list': '[]',
            'dict': '{}',
            'None': 'None',
            'Optional': 'None',
        }

        for param in func.parameters:
            if param.type_hint:
                for type_name, sample in type_samples.items():
                    if type_name in param.type_hint:
                        inputs[param.name] = sample
                        break
                else:
                    inputs[param.name] = '"test"'
            else:
                inputs[param.name] = '"test"'

        return inputs
