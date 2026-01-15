"""
Validator - Validates and sanitizes TestSpec data before rendering.

This module sits between SpecGenerator and TemplateRenderer to:
1. Validate TestSpec structure and types
2. Sanitize Python syntax for Java code
3. Filter invalid imports
4. Report clear validation errors

This addresses the validation gap identified in PIPELINE_ANALYSIS.md.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .models import TestSpec, TestCase, ExpectedOutput, MockSpec, Language


@dataclass
class ValidationResult:
    """Result of validation with issues and sanitized spec."""
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_spec: TestSpec | None = None


class SpecValidator:
    """
    Validates and sanitizes TestSpec before rendering.

    Usage:
        validator = SpecValidator()
        result = validator.validate(spec)
        if result.is_valid:
            rendered = renderer.render(result.sanitized_spec)
        else:
            print("Errors:", result.issues)
    """

    # Python syntax patterns to detect
    PYTHON_PATTERNS = {
        'None': 'null',
        'True': 'true',
        'False': 'false',
    }

    # Invalid Java code patterns (LLM hallucinations)
    INVALID_JAVA_PATTERNS = [
        # Semicolons inside method call parentheses
        (r'\([^)]*;[^)]*\)', 'inline semicolons in method call'),
        # Variable declaration inside method call
        (r'\(\s*\w+\s+\w+\s*=\s*new\s+', 'variable declaration inside method call'),
        # Class name used as method (e.g., instance.Owner(...))
        (r'instance\.[A-Z][a-zA-Z]+\s*\(', 'class name used as method'),
    ]

    # Python import patterns (invalid for Java)
    PYTHON_IMPORT_PATTERNS = [
        r'^from\s+\w+',           # from x import y
        r'^import\s+\w+\s*$',     # import x (no dots - likely Python)
        r'import\s+pytest',
        r'import\s+unittest',
        r'import\s+flask',
        r'import\s+django',
        r'import\s+numpy',
        r'import\s+pandas',
    ]

    # Valid Java import patterns
    JAVA_IMPORT_PATTERN = re.compile(
        r'^import\s+(static\s+)?'  # import or import static
        r'[a-z][a-z0-9_]*'         # package start (lowercase)
        r'(\.[a-zA-Z][a-zA-Z0-9_]*)+' # more package parts
        r'(\.\*)?;?$'              # optional wildcard and semicolon
    )

    def validate(self, spec: TestSpec) -> ValidationResult:
        """
        Validate and sanitize a TestSpec.

        Returns ValidationResult with:
        - is_valid: True if spec is usable (may have warnings)
        - issues: Critical errors that prevent rendering
        - warnings: Non-critical issues that were auto-fixed
        - sanitized_spec: Clean spec ready for rendering (if valid)
        """
        issues = []
        warnings = []

        # 1. Validate basic structure
        structure_issues = self._validate_structure(spec)
        issues.extend(structure_issues)

        if issues:
            return ValidationResult(is_valid=False, issues=issues)

        # 2. Create a sanitized copy
        sanitized = self._copy_spec(spec)

        # 3. Language-specific sanitization
        if spec.language == Language.JAVA:
            sanitized, java_warnings = self._sanitize_for_java(sanitized)
            warnings.extend(java_warnings)

        # 4. Validate test cases
        for i, tc in enumerate(sanitized.test_cases):
            tc_issues, tc_warnings = self._validate_test_case(tc, i)
            issues.extend(tc_issues)
            warnings.extend(tc_warnings)

        if issues:
            return ValidationResult(is_valid=False, issues=issues, warnings=warnings)

        return ValidationResult(
            is_valid=True,
            issues=[],
            warnings=warnings,
            sanitized_spec=sanitized
        )

    def _validate_structure(self, spec: TestSpec) -> list[str]:
        """Validate basic TestSpec structure."""
        issues = []

        if not spec.target_name:
            issues.append("Missing target_name in TestSpec")

        if not spec.target_file:
            issues.append("Missing target_file in TestSpec")

        if spec.test_cases is None:
            issues.append("test_cases is None (should be empty list)")
        elif len(spec.test_cases) == 0:
            issues.append("test_cases is empty - no tests to generate")

        return issues

    def _validate_test_case(self, tc: TestCase, index: int) -> tuple[list[str], list[str]]:
        """Validate a single test case."""
        issues = []
        warnings = []

        if not tc.name:
            issues.append(f"TestCase[{index}] missing name")

        if tc.expected is None:
            warnings.append(f"TestCase[{index}] has None expected - will use default")

        if tc.inputs is None:
            warnings.append(f"TestCase[{index}] has None inputs - will use empty dict")

        # Check for invalid Java patterns in inputs
        if tc.inputs:
            for key, value in tc.inputs.items():
                if value:
                    invalid = self._detect_invalid_java_patterns(str(value))
                    if invalid:
                        issues.append(f"TestCase[{index}] input '{key}' has {invalid}")

        # Check expected returns for invalid patterns
        if tc.expected and tc.expected.returns:
            invalid = self._detect_invalid_java_patterns(str(tc.expected.returns))
            if invalid:
                issues.append(f"TestCase[{index}] expected.returns has {invalid}")

        return issues, warnings

    def _detect_invalid_java_patterns(self, code: str) -> str | None:
        """
        Detect invalid Java code patterns that LLMs commonly generate.

        Returns description of the issue if found, None otherwise.
        """
        for pattern, description in self.INVALID_JAVA_PATTERNS:
            if re.search(pattern, code):
                return description
        return None

    def _copy_spec(self, spec: TestSpec) -> TestSpec:
        """Create a deep copy of TestSpec for sanitization."""
        return TestSpec(
            test_type=spec.test_type,
            target_file=spec.target_file,
            target_name=spec.target_name,
            target_class=spec.target_class,
            language=spec.language,
            test_cases=[self._copy_test_case(tc) for tc in spec.test_cases],
            fixtures_needed=list(spec.fixtures_needed) if spec.fixtures_needed else [],
            imports_needed=list(spec.imports_needed) if spec.imports_needed else [],
            parametrize=spec.parametrize,
            complexity_score=spec.complexity_score,
            requires_custom_generation=spec.requires_custom_generation,
            return_semantics=spec.return_semantics,
            requires_context=list(spec.requires_context) if spec.requires_context else [],
            framework_hints=list(spec.framework_hints) if spec.framework_hints else [],
            is_generator=spec.is_generator,
            mutates_args=spec.mutates_args
        )

    def _copy_test_case(self, tc: TestCase) -> TestCase:
        """Create a copy of TestCase."""
        return TestCase(
            name=tc.name,
            category=tc.category,
            description=tc.description,
            inputs=dict(tc.inputs) if tc.inputs else {},
            expected=ExpectedOutput(
                returns=tc.expected.returns if tc.expected else None,
                raises=tc.expected.raises if tc.expected else None,
                raises_message=tc.expected.raises_message if tc.expected else None,
                side_effects=tc.expected.side_effects if tc.expected else None
            ),
            mocks=[self._copy_mock(m) for m in tc.mocks] if tc.mocks else [],
            setup=list(tc.setup) if tc.setup else [],
            teardown=list(tc.teardown) if tc.teardown else []
        )

    def _copy_mock(self, mock: MockSpec) -> MockSpec:
        """Create a copy of MockSpec."""
        return MockSpec(
            target=mock.target,
            return_value=mock.return_value,
            side_effect=mock.side_effect,
            assert_called_with=mock.assert_called_with
        )

    def _sanitize_for_java(self, spec: TestSpec) -> tuple[TestSpec, list[str]]:
        """Sanitize TestSpec for Java code generation."""
        warnings = []

        # 1. Filter imports
        valid_imports = []
        for imp in spec.imports_needed:
            if self._is_valid_java_import(imp):
                valid_imports.append(imp)
            else:
                warnings.append(f"Removed invalid import: {imp}")

        spec.imports_needed = valid_imports

        # 2. Sanitize setup - remove method calls that should be generated by template
        for tc in spec.test_cases:
            if tc.setup:
                clean_setup = []
                for line in tc.setup:
                    # Remove lines that look like method calls that should be generated
                    # Pattern: "var result = something.methodName(...)" or "result = ..."
                    if re.match(r'^\s*(?:var\s+)?result\s*=\s*', line):
                        warnings.append(f"Removed method call from setup: {line[:50]}...")
                        continue
                    # Also catch: "petType.parse(...)" or "formatter.print(...)" without var result
                    if re.match(r'^\s*\w+\s*\.\s*' + re.escape(spec.target_name) + r'\s*\(', line):
                        warnings.append(f"Removed target method call from setup: {line[:50]}...")
                        continue
                    clean_setup.append(line)
                tc.setup = clean_setup

        # 3. Sanitize test case values
        for tc in spec.test_cases:
            # Sanitize inputs - fix string-quoted variable names
            if tc.inputs:
                for key, value in tc.inputs.items():
                    sanitized, changed = self._sanitize_java_value(value)
                    if changed:
                        warnings.append(f"Sanitized input {key}: {value} -> {sanitized}")
                        tc.inputs[key] = sanitized

                    # Fix: LLM outputs "\"varName\"" which becomes a string literal
                    # If setup creates a variable with this name, convert to variable reference
                    if tc.setup and isinstance(tc.inputs[key], str):
                        fixed_value, fixed = self._fix_string_to_variable_ref(tc.inputs[key], key, tc.setup)
                        if fixed:
                            warnings.append(f"Fixed string->var reference: {tc.inputs[key]} -> {fixed_value}")
                            tc.inputs[key] = fixed_value

            # Sanitize expected returns
            if tc.expected and tc.expected.returns:
                sanitized, changed = self._sanitize_java_value(tc.expected.returns)
                if changed:
                    warnings.append(f"Sanitized expected.returns: {tc.expected.returns} -> {sanitized}")
                    tc.expected.returns = sanitized

            # Sanitize mock return values
            for mock in tc.mocks:
                if mock.return_value:
                    sanitized, changed = self._sanitize_java_value(str(mock.return_value))
                    if changed:
                        warnings.append(f"Sanitized mock return_value: {mock.return_value} -> {sanitized}")
                        mock.return_value = sanitized

        return spec, warnings

    def _is_valid_java_import(self, imp: str) -> bool:
        """Check if import is valid Java import."""
        imp = imp.strip()

        # Check for Python import patterns
        for pattern in self.PYTHON_IMPORT_PATTERNS:
            if re.match(pattern, imp):
                return False

        # Check if it matches Java import pattern
        if self.JAVA_IMPORT_PATTERN.match(imp):
            return True

        # Also accept common JUnit/Mockito imports that might not match perfectly
        java_keywords = ['org.junit', 'org.mockito', 'java.', 'javax.', 'static org.']
        return any(kw in imp for kw in java_keywords)

    def _fix_string_to_variable_ref(self, value: str, key: str, setup: list[str]) -> tuple[str, bool]:
        """
        Fix string literals that should be variable references.

        If the value is a quoted string like '"specialty"' and the setup contains
        a variable declaration like 'Specialty specialty = new Specialty()',
        convert the input to the variable name 'specialty'.

        This fixes LLM hallucination where it outputs:
            inputs: {"specialty": "\"specialty\""}  -> addSpecialty("specialty")
        When it should be:
            inputs: {"specialty": "specialty"}      -> addSpecialty(specialty)
        """
        # Check if value looks like a quoted string: "varname" or \"varname\"
        stripped = value.strip()
        if not stripped:
            return value, False

        # Pattern: the value is exactly a quoted string like "specialty" or \"specialty\"
        # JSON encoding: "\"specialty\"" becomes the string: "specialty" (with quotes)
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
            inner = stripped[1:-1]  # Extract the inner part without quotes

            # Check if this matches a variable name in setup
            var_pattern = rf'\b{re.escape(inner)}\s*='
            for line in setup:
                if re.search(var_pattern, line):
                    # Found a variable declaration with this name
                    return inner, True

            # Also check if the key name itself matches a variable in setup
            key_pattern = rf'\b{re.escape(key)}\s*='
            for line in setup:
                if re.search(key_pattern, line):
                    # The input key matches a setup variable, use the key directly
                    return key, True

        # Check if value equals the key name as a string (common pattern)
        # e.g., inputs: {"specialty": "specialty"} where setup has "Specialty specialty = ..."
        if stripped == key:
            # This is already correct - a variable reference
            return value, False

        # Check if the key itself is declared in setup
        key_pattern = rf'\b{re.escape(key)}\s*='
        for line in setup:
            if re.search(key_pattern, line):
                # The key is a setup variable, and the value should reference it
                if stripped != key:
                    return key, True

        return value, False

    def _sanitize_java_value(self, value: Any) -> tuple[Any, bool]:
        """
        Sanitize a value for Java code.

        Returns (sanitized_value, was_changed).
        """
        if value is None:
            return None, False

        value_str = str(value)
        original = value_str
        changed = False

        # Replace Python syntax with Java equivalents
        for py_syntax, java_syntax in self.PYTHON_PATTERNS.items():
            if py_syntax in value_str:
                value_str = re.sub(rf'\b{py_syntax}\b', java_syntax, value_str)
                changed = True

        # Remove Python-style dict/list if present
        if value_str.startswith("{'") or value_str.startswith("['"):
            # This is Python dict/list syntax - replace with null for safety
            value_str = "null"
            changed = True

        return value_str, (value_str != original)


def validate_and_sanitize(spec: TestSpec) -> ValidationResult:
    """
    Convenience function to validate and sanitize a TestSpec.

    Usage:
        result = validate_and_sanitize(spec)
        if result.is_valid:
            code = renderer.render(result.sanitized_spec)
    """
    validator = SpecValidator()
    return validator.validate(spec)
