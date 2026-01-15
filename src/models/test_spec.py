"""
TestSpec DSL - Structured test specification that compiles to Java.

This module defines the schema for test specifications. The LLM generates
TestSpec JSON, which is then compiled deterministically to Java code.

Key insight: By constraining LLM to output structured JSON instead of
freeform Java, we eliminate syntax hallucinations entirely.
"""

from dataclasses import dataclass, field
from typing import Literal
from pydantic import BaseModel, Field
import json


class MockSetup(BaseModel):
    """Setup for a mock expectation (when/thenReturn or doThrow/when)."""
    mock_name: str = Field(..., description="Name of the mock variable (e.g., 'productServices')")
    method: str = Field(..., description="Method to mock (e.g., 'addProduct')")
    args: list[str] = Field(default_factory=list, description="Arguments matcher (e.g., ['any(Product.class)'])")
    returns: str | None = Field(None, description="Return value (null for void methods)")
    throws: str | None = Field(None, description="Exception to throw (e.g., 'RuntimeException')")


class Assertion(BaseModel):
    """A single assertion in a test case."""
    type: Literal["equals", "not_null", "true", "false", "throws", "verify"] = Field(
        ..., description="Type of assertion"
    )
    expected: str | None = Field(None, description="Expected value for equals assertions")
    actual: str | None = Field(None, description="Actual value/variable to check")
    exception: str | None = Field(None, description="Exception class for assertThrows")
    mock_name: str | None = Field(None, description="Mock name for verify assertions")
    method: str | None = Field(None, description="Method name for verify assertions")
    args: list[str] | None = Field(None, description="Arguments for verify assertions")


class MethodCall(BaseModel):
    """A method call on the test subject."""
    object: str = Field("controller", description="Object to call method on")
    method: str = Field(..., description="Method name to call")
    args: list[str] = Field(default_factory=list, description="Arguments to pass")
    returns_to: str | None = Field(None, description="Variable name to store result (None for void)")


class TestCaseSpec(BaseModel):
    """Specification for a single test case."""
    name: str = Field(..., description="Test method name (e.g., 'addProduct_success')")
    description: str = Field(..., description="Human-readable description for @DisplayName")

    # Arrange phase
    mocks: list[MockSetup] = Field(default_factory=list, description="Mock setups (when/thenReturn)")
    arrange: list[str] = Field(default_factory=list, description="Setup statements (Java code lines)")

    # Act phase
    action: MethodCall = Field(..., description="The method call to test")

    # Assert phase
    assertions: list[Assertion] = Field(default_factory=list, description="Assertions to verify")


class MockSpec(BaseModel):
    """Specification for a mock dependency."""
    type: str = Field(..., description="Type name (e.g., 'ProductServices')")
    name: str = Field(..., description="Variable name (e.g., 'productServices')")


class TestSpec(BaseModel):
    """
    Complete test specification for a Java class.

    This is the top-level schema that the LLM generates.
    The SpecCompiler transforms this into Java test code.
    """
    class_under_test: str = Field(..., description="Name of the class being tested")
    package_name: str = Field(..., description="Package name for the test class")
    dependencies: list[MockSpec] = Field(default_factory=list, description="Dependencies to mock")
    test_cases: list[TestCaseSpec] = Field(default_factory=list, description="Test cases to generate")

    # Optional: additional imports needed
    additional_imports: list[str] = Field(default_factory=list, description="Extra import statements")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TestSpec":
        """Parse from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def get_json_schema(cls) -> str:
        """Get JSON schema for LLM prompt."""
        return json.dumps(cls.model_json_schema(), indent=2)


def get_testspec_prompt_schema() -> str:
    """
    Get a simplified schema description for the LLM prompt.

    This is more concise than the full JSON schema and easier for LLMs to follow.
    """
    return '''
{
  "test_cases": [
    {
      "name": "methodName_scenario",
      "description": "should do X when Y",
      "mocks": [
        {
          "mock_name": "serviceName",
          "method": "methodName",
          "args": ["any(Type.class)"],
          "returns": null,
          "throws": null
        }
      ],
      "arrange": [
        "Type variable = new Type()",
        "variable.setField(\\"value\\")"
      ],
      "action": {
        "object": "controller",
        "method": "methodName",
        "args": ["variable"],
        "returns_to": "result"
      },
      "assertions": [
        {"type": "equals", "expected": "\\"expectedValue\\"", "actual": "result"},
        {"type": "verify", "mock_name": "serviceName", "method": "methodName", "args": ["variable"]},
        {"type": "not_null", "actual": "result"},
        {"type": "throws", "exception": "RuntimeException", "actual": "() -> controller.method()"}
      ]
    }
  ]
}
'''
