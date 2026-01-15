"""
Shared fixtures and utilities for testing the Java Test Agent pipeline.

Provides:
- Sample Java code fixtures
- Mock LLM response factories
- FaultInjector for boundary fault injection
- Data model factories for testing
"""

import json
import pytest
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    Language, TestType, TestCategory, ErrorCategory, ReturnSemantics,
    CodeLocation, Parameter, FunctionContext, ClassContext,
    TestSpec, TestCase, ExpectedOutput, MockSpec, FixAttempt
)


# ============================================================================
# FaultInjector - Helper for boundary fault injection
# ============================================================================

class FaultInjector:
    """Helper for injecting faults at component boundaries."""

    # --- JSON Corruption ---

    @staticmethod
    def corrupt_json(valid_json: dict, corruption: str) -> str:
        """Corrupt valid JSON in specific ways.

        Args:
            valid_json: A valid JSON dict
            corruption: Type of corruption to apply
                - 'truncate': Cut off the JSON
                - 'invalid_syntax': Add invalid characters
                - 'missing_brace': Remove closing brace
                - 'empty': Return empty object
                - 'null': Return null
                - 'array_instead': Return array instead of object
        """
        json_str = json.dumps(valid_json)

        if corruption == 'truncate':
            return json_str[:len(json_str)//2]
        elif corruption == 'invalid_syntax':
            return json_str.replace('"', "'")  # Invalid JSON quotes
        elif corruption == 'missing_brace':
            return json_str.rstrip('}')
        elif corruption == 'empty':
            return '{}'
        elif corruption == 'null':
            return 'null'
        elif corruption == 'array_instead':
            return '[' + json_str + ']'
        else:
            return json_str

    # --- Java Code Malformation ---

    @staticmethod
    def malform_java(valid_java: str, fault_type: str) -> str:
        """Introduce syntax errors in Java code.

        Args:
            valid_java: Valid Java source code
            fault_type: Type of malformation
                - 'unbalanced_braces': Remove closing braces
                - 'mixed_language': Insert Python syntax
                - 'incomplete_method': Cut off method body
                - 'unicode': Add unicode identifiers
                - 'binary': Add binary garbage
                - 'duplicate_class': Duplicate class declaration
        """
        if fault_type == 'unbalanced_braces':
            # Remove last 3 closing braces
            count = 0
            result = []
            for char in reversed(valid_java):
                if char == '}' and count < 3:
                    count += 1
                    continue
                result.append(char)
            return ''.join(reversed(result))

        elif fault_type == 'mixed_language':
            # Insert Python code in the middle
            lines = valid_java.split('\n')
            mid = len(lines) // 2
            lines.insert(mid, "    def python_method(self):")
            lines.insert(mid + 1, "        return None")
            return '\n'.join(lines)

        elif fault_type == 'incomplete_method':
            # Find a method and truncate it
            idx = valid_java.find('{', valid_java.find('public'))
            if idx > 0:
                return valid_java[:idx + 1] + '\n        // incomplete'
            return valid_java

        elif fault_type == 'unicode':
            return valid_java.replace('public void', 'public void méthodé')

        elif fault_type == 'binary':
            return valid_java + '\x00\x01\x02\x03'

        elif fault_type == 'duplicate_class':
            return valid_java + '\n' + valid_java

        return valid_java

    # --- Invalid TestSpec Creation ---

    @staticmethod
    def create_invalid_spec(**overrides) -> TestSpec:
        """Create TestSpec with specific invalid fields.

        Pass field overrides to inject specific invalid values.
        """
        defaults = {
            'test_type': TestType.UNIT_CLASS,
            'target_file': 'Example.java',
            'target_name': 'exampleMethod',
            'target_class': 'Example',
            'language': Language.JAVA,
            'test_cases': [
                TestCase(
                    name='test_example',
                    category=TestCategory.HAPPY_PATH,
                    description='Test example',
                    inputs={},
                    expected=ExpectedOutput(returns=None)
                )
            ]
        }
        defaults.update(overrides)
        return TestSpec(**defaults)

    # --- Corrupt Surefire XML ---

    @staticmethod
    def create_corrupt_surefire_xml(fault_type: str) -> str:
        """Generate malformed Surefire XML.

        Args:
            fault_type: Type of corruption
                - 'invalid_xml': Not valid XML
                - 'missing_testcase': Missing testcase element
                - 'empty': Empty file
                - 'wrong_root': Wrong root element
                - 'truncated': Truncated XML
        """
        valid_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ExampleTest" tests="2" failures="1" errors="0">
    <testcase name="testSuccess" classname="com.example.ExampleTest" time="0.001"/>
    <testcase name="testFailure" classname="com.example.ExampleTest" time="0.002">
        <failure message="expected 5 but was 3" type="AssertionError">
            java.lang.AssertionError: expected 5 but was 3
                at com.example.ExampleTest.testFailure(ExampleTest.java:15)
        </failure>
    </testcase>
</testsuite>'''

        if fault_type == 'invalid_xml':
            return '<not valid xml <<<'
        elif fault_type == 'missing_testcase':
            return '''<?xml version="1.0"?>
<testsuite name="ExampleTest" tests="0" failures="0" errors="0">
</testsuite>'''
        elif fault_type == 'empty':
            return ''
        elif fault_type == 'wrong_root':
            return '<?xml version="1.0"?><notatestsuite></notatestsuite>'
        elif fault_type == 'truncated':
            return valid_xml[:len(valid_xml)//2]

        return valid_xml


# ============================================================================
# Data Model Factories
# ============================================================================

class ModelFactory:
    """Factory for creating test data models."""

    @staticmethod
    def create_code_location(
        file_path: str = "/test/Example.java",
        start_line: int = 1,
        end_line: int = 10
    ) -> CodeLocation:
        return CodeLocation(
            file_path=Path(file_path),
            start_line=start_line,
            end_line=end_line,
            start_col=0,
            end_col=0
        )

    @staticmethod
    def create_parameter(
        name: str = "param",
        type_hint: str = "String",
        default_value: str = None
    ) -> Parameter:
        return Parameter(
            name=name,
            type_hint=type_hint,
            default_value=default_value,
            is_optional=default_value is not None
        )

    @staticmethod
    def create_function_context(
        name: str = "exampleMethod",
        class_name: str = "Example",
        parameters: list = None,
        return_type: str = "void",
        source_code: str = None,
        **kwargs
    ) -> FunctionContext:
        if parameters is None:
            parameters = []
        if source_code is None:
            source_code = f"public {return_type} {name}() {{}}"

        defaults = {
            'name': name,
            'location': ModelFactory.create_code_location(),
            'parameters': parameters,
            'return_type': return_type,
            'docstring': None,
            'is_async': False,
            'is_method': True,
            'class_name': class_name,
            'decorators': [],
            'source_code': source_code,
            'calls': [],
            'imports': [],
            'references': [],
            'is_generator': False,
            'return_semantics': ReturnSemantics.VALUE,
            'requires_context': [],
            'mutates_args': False,
            'framework_hints': []
        }
        defaults.update(kwargs)
        return FunctionContext(**defaults)

    @staticmethod
    def create_class_context(
        name: str = "Example",
        methods: list = None,
        base_classes: list = None,
        **kwargs
    ) -> ClassContext:
        if methods is None:
            methods = [ModelFactory.create_function_context(class_name=name)]
        if base_classes is None:
            base_classes = []

        defaults = {
            'name': name,
            'location': ModelFactory.create_code_location(),
            'methods': methods,
            'base_classes': base_classes,
            'docstring': None,
            'decorators': [],
            'attributes': [],
            'constructors': []
        }
        defaults.update(kwargs)
        return ClassContext(**defaults)

    @staticmethod
    def create_test_spec(
        target_name: str = "exampleMethod",
        target_class: str = "Example",
        test_cases: list = None,
        **kwargs
    ) -> TestSpec:
        if test_cases is None:
            test_cases = [ModelFactory.create_test_case()]

        defaults = {
            'test_type': TestType.UNIT_CLASS,
            'target_file': 'Example.java',
            'target_name': target_name,
            'target_class': target_class,
            'language': Language.JAVA,
            'test_cases': test_cases,
            'fixtures_needed': [],
            'imports_needed': [],
            'parametrize': False,
            'complexity_score': 1,
            'requires_custom_generation': False,
            'return_semantics': ReturnSemantics.VALUE,
            'requires_context': [],
            'framework_hints': [],
            'is_generator': False,
            'mutates_args': False
        }
        defaults.update(kwargs)
        return TestSpec(**defaults)

    @staticmethod
    def create_test_case(
        name: str = "test_example",
        category: TestCategory = TestCategory.HAPPY_PATH,
        inputs: dict = None,
        expected_returns: Any = None,
        **kwargs
    ) -> TestCase:
        if inputs is None:
            inputs = {}

        defaults = {
            'name': name,
            'category': category,
            'description': f'Test case: {name}',
            'inputs': inputs,
            'expected': ExpectedOutput(returns=expected_returns),
            'mocks': [],
            'setup': [],
            'teardown': []
        }
        defaults.update(kwargs)
        return TestCase(**defaults)

    @staticmethod
    def create_mock_spec(
        target: str = "repository.findById",
        return_value: Any = None,
        **kwargs
    ) -> MockSpec:
        defaults = {
            'target': target,
            'return_value': return_value,
            'side_effect': None,
            'assert_called_with': None
        }
        defaults.update(kwargs)
        return MockSpec(**defaults)


# ============================================================================
# Sample Java Code Fixtures
# ============================================================================

@pytest.fixture
def valid_java_class():
    """A valid Java class with various method types."""
    return '''package com.example;

import java.util.List;
import java.util.Map;

/**
 * Example class for testing.
 */
public class Example {
    private String name;
    private int value;

    public Example() {
        this.name = "";
        this.value = 0;
    }

    public Example(String name, int value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Gets the name.
     * @return the name
     */
    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getValue() {
        return this.value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public List<String> getItems() {
        return List.of("a", "b", "c");
    }

    public Map<String, Integer> getMapping() {
        return Map.of("one", 1, "two", 2);
    }

    public void processWithParams(String input, int count, boolean flag) {
        // Process logic
    }
}
'''


@pytest.fixture
def valid_spring_controller():
    """A valid Spring MVC controller."""
    return '''package com.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.example.service.UserService;
import com.example.model.User;
import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @GetMapping
    public List<User> getAllUsers() {
        return userService.findAll();
    }

    @GetMapping("/{id}")
    public User getUserById(@PathVariable Long id) {
        return userService.findById(id);
    }

    @PostMapping
    public User createUser(@RequestBody User user) {
        return userService.save(user);
    }

    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        userService.delete(id);
    }
}
'''


@pytest.fixture
def java_class_with_generics():
    """Java class with complex generics."""
    return '''package com.example;

import java.util.*;

public class GenericExample<T, K extends Comparable<K>> {
    private Map<K, List<T>> items;
    private Set<Map.Entry<String, T>> entries;

    public Map<K, List<T>> getItems() {
        return items;
    }

    public void setItems(Map<K, List<T>> items) {
        this.items = items;
    }

    public <R> R transform(T input, java.util.function.Function<T, R> transformer) {
        return transformer.apply(input);
    }

    public Map<String, List<Map<K, T>>> getNestedGenerics() {
        return new HashMap<>();
    }
}
'''


@pytest.fixture
def malformed_java_unbalanced():
    """Java with unbalanced braces."""
    return '''package com.example;

public class Broken {
    public void method() {
        if (true) {
            // missing closing brace

    public void anotherMethod() {
        // this shouldn't parse correctly
    }
}
'''


@pytest.fixture
def empty_java_file():
    """Empty Java file."""
    return ''


@pytest.fixture
def java_with_abstract_methods():
    """Abstract class with abstract methods."""
    return '''package com.example;

public abstract class AbstractExample {

    public abstract void doSomething();

    public abstract String getName();

    public void concreteMethod() {
        System.out.println("concrete");
    }
}
'''


# ============================================================================
# Mock LLM Response Fixtures
# ============================================================================

@pytest.fixture
def valid_llm_response():
    """A valid LLM JSON response for spec generation."""
    return {
        "test_type": "unit_class",
        "target_name": "getName",
        "target_class": "Example",
        "test_cases": [
            {
                "name": "test_getName_returns_name",
                "category": "happy_path",
                "description": "Test that getName returns the name",
                "inputs": {},
                "expected": {
                    "returns": "\"test name\""
                }
            },
            {
                "name": "test_getName_returns_empty_when_not_set",
                "category": "edge_case",
                "description": "Test getName when name is not set",
                "inputs": {},
                "expected": {
                    "returns": "\"\""
                }
            }
        ],
        "imports_needed": [
            "import org.junit.jupiter.api.Test",
            "import static org.junit.jupiter.api.Assertions.*"
        ]
    }


@pytest.fixture
def llm_response_with_python_syntax():
    """LLM response containing Python syntax that needs conversion."""
    return {
        "test_type": "unit_class",
        "target_name": "getValue",
        "test_cases": [
            {
                "name": "test_getValue",
                "category": "happy_path",
                "description": "Test getValue",
                "inputs": {"value": "None"},  # Python None
                "expected": {
                    "returns": "None"  # Should be null
                }
            }
        ],
        "imports_needed": [
            "from unittest import TestCase",  # Python import
            "import org.junit.jupiter.api.Test"
        ]
    }


@pytest.fixture
def llm_response_empty_test_cases():
    """LLM response with empty test cases."""
    return {
        "test_type": "unit_class",
        "target_name": "method",
        "test_cases": []
    }


@pytest.fixture
def llm_response_missing_fields():
    """LLM response missing required fields."""
    return {
        "test_cases": [
            {
                "name": "test_something"
                # Missing: category, description, inputs, expected
            }
        ]
        # Missing: test_type, target_name
    }


# ============================================================================
# Surefire XML Fixtures
# ============================================================================

@pytest.fixture
def valid_surefire_xml():
    """Valid Surefire XML with passing and failing tests."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.ExampleTest" tests="3" failures="1" errors="0" skipped="0" time="0.123">
    <testcase name="testSuccess" classname="com.example.ExampleTest" time="0.010"/>
    <testcase name="testAnotherSuccess" classname="com.example.ExampleTest" time="0.008"/>
    <testcase name="testFailure" classname="com.example.ExampleTest" time="0.015">
        <failure message="expected: &lt;5&gt; but was: &lt;3&gt;" type="org.opentest4j.AssertionFailedError">
org.opentest4j.AssertionFailedError: expected: &lt;5&gt; but was: &lt;3&gt;
    at org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:55)
    at org.junit.jupiter.api.Assertions.fail(Assertions.java:118)
    at com.example.ExampleTest.testFailure(ExampleTest.java:25)
        </failure>
    </testcase>
</testsuite>'''


@pytest.fixture
def surefire_xml_null_pointer():
    """Surefire XML with NullPointerException."""
    return '''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.ExampleTest" tests="1" failures="0" errors="1" time="0.05">
    <testcase name="testNullPointer" classname="com.example.ExampleTest" time="0.020">
        <error message="Cannot invoke method on null" type="java.lang.NullPointerException">
java.lang.NullPointerException: Cannot invoke method on null
    at com.example.Example.getName(Example.java:15)
    at com.example.ExampleTest.testNullPointer(ExampleTest.java:30)
        </error>
    </testcase>
</testsuite>'''


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_java_file(tmp_path):
    """Create a temporary Java file."""
    def _create(content: str, filename: str = "Example.java") -> Path:
        java_file = tmp_path / filename
        java_file.write_text(content)
        return java_file
    return _create


@pytest.fixture
def mock_openrouter_response():
    """Factory for mocking OpenRouter API responses."""
    def _create_response(json_data: dict, status_code: int = 200):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(json_data)
                    }
                }
            ]
        }
        mock_response.text = json.dumps({"choices": [{"message": {"content": json.dumps(json_data)}}]})
        return mock_response
    return _create_response


@pytest.fixture
def fault_injector():
    """Instance of FaultInjector for use in tests."""
    return FaultInjector()


@pytest.fixture
def model_factory():
    """Instance of ModelFactory for use in tests."""
    return ModelFactory()
