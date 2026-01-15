"""
Direct Code Generator - Generate Java test code directly with self-verification.

This is an alternative to the spec-based approach. Instead of:
  LLM -> JSON Spec -> Template -> Java Code

This does:
  LLM -> Java Code -> LLM Self-Review -> Java Code

The hypothesis is that self-verification can catch semantic errors
that regex postprocessors cannot.
"""

import hashlib
import json
import os
import re
import time
import requests
from pathlib import Path

from ..models import FunctionContext, ClassContext, ProjectTestPatterns
from ..context.test_examples import TestExamplesFinder
from ..few_shot import FewShotExampleSelector, SelectedExample


def _validate_java_code_quality(code: str) -> tuple[bool, str]:
    """
    Basic quality validation for Java test code.

    Returns (is_valid, error_message).
    Checks for common structural issues that indicate broken code.
    """
    if not code or len(code.strip()) < 50:
        return False, "Code is empty or too short"

    # Must have a package declaration or import
    if "package " not in code and "import " not in code:
        return False, "Missing package/import declarations"

    # Must have a class declaration
    if "class " not in code:
        return False, "Missing class declaration"

    # Must have at least one @Test method
    if "@Test" not in code:
        return False, "No @Test methods found"

    # Check for balanced braces (basic check)
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces != close_braces:
        return False, f"Unbalanced braces: {open_braces} open, {close_braces} close"

    # Check for common syntax errors
    # Empty method bodies that look broken
    if re.search(r"@Test\s*\n\s*\n", code):
        return False, "Empty test method detected"

    # Method inside method (Java doesn't allow this)
    if re.search(r"void\s+\w+\s*\([^)]*\)\s*\{[^}]*void\s+\w+\s*\([^)]*\)\s*\{", code):
        return False, "Nested method definitions detected"

    return True, ""


# Reuse cache from spec_generator
_LLM_CACHE: dict[str, str] = {}
_CACHE_DIR = Path(__file__).parent.parent.parent / ".llm_cache"


def _get_cache_key(prompt: str, model: str) -> str:
    """Generate cache key from prompt and model."""
    content = f"{model}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_cache():
    """Load cache from disk."""
    global _LLM_CACHE
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                _LLM_CACHE[f.stem] = data["response"]
            except:
                pass


def _save_to_cache(key: str, response: str):
    """Save response to cache."""
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file = _CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps({"response": response}))
    _LLM_CACHE[key] = response


# Load cache on module import
_load_cache()


class DirectCodeGenerator:
    """
    Generate Java test code directly without JSON spec intermediate.

    Uses a two-phase approach:
    1. Generate: LLM writes Java test code directly
    2. Self-Verify: LLM reviews and fixes its own code

    This approach trades structure for flexibility, allowing the LLM
    to catch semantic errors that template-based approaches miss.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "openrouter",
        self_verify: bool = True
    ):
        self.provider = provider
        self.model = model
        self.self_verify = self_verify
        self.few_shot_selector = FewShotExampleSelector()

        # Get appropriate API key
        if provider == "openrouter":
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            self.base_url = None

    def generate_for_function(
        self,
        func: FunctionContext,
        class_context: ClassContext | None = None,
        patterns: ProjectTestPatterns | None = None,
        api_context: str | None = None
    ) -> str:
        """
        Generate test code for a single function.

        Args:
            func: The function to generate tests for
            class_context: Optional class context for additional info
            patterns: Optional project patterns
            api_context: Optional API context string

        Returns:
            Complete Java test code as a string
        """
        # Phase 1: Generate Java code directly
        prompt = self._build_generation_prompt(func, class_context, patterns, api_context)
        raw_code = self._call_llm(prompt)
        raw_code = self._extract_java_code(raw_code)

        # Phase 2: Self-verify (if enabled)
        if self.self_verify:
            verified_code = self._self_verify_code(raw_code, func, class_context)
            return verified_code

        return raw_code

    def generate_for_class(
        self,
        cls: ClassContext,
        project_context=None,
        patterns: ProjectTestPatterns | None = None,
        test_examples_finder: TestExamplesFinder | None = None
    ) -> str:
        """
        Generate test code for an entire class (all methods).

        Unlike spec generator which generates per-method specs,
        this generates a single test class covering all methods.

        Args:
            cls: The class context
            project_context: Optional ProjectContext for API information
            patterns: Optional test patterns
            test_examples_finder: Optional finder for few-shot examples

        Returns:
            Complete Java test class code as a string
        """
        # Build API context
        api_context = None
        if project_context is not None:
            api_context = project_context.build_api_context_for_class(cls.name)

        # Get curated few-shot examples based on component type
        source_code = ""
        if cls.location and cls.location.file_path:
            try:
                source_code = cls.location.file_path.read_text()
            except:
                pass
        curated_examples = self.few_shot_selector.select_examples(cls, source_code, max_examples=2)

        # Also get project-specific examples as supplementary
        project_examples = []
        if test_examples_finder is not None:
            project_examples = test_examples_finder.find_similar_tests(cls.name, max_examples=1)

        # Phase 1: Generate complete test class
        prompt = self._build_class_generation_prompt(cls, patterns, api_context, curated_examples, project_examples)
        raw_code = self._call_llm(prompt)
        raw_code = self._extract_java_code(raw_code)

        # Fix missing @Mock/@InjectMocks pattern if class has dependencies
        deps = self._extract_dependencies(cls)
        if deps:
            raw_code = self._fix_missing_mock_pattern(raw_code, cls.name, deps)

        # Phase 2: Self-verify with quality gate (if enabled)
        if self.self_verify:
            # Validate raw code first
            raw_valid, raw_error = _validate_java_code_quality(raw_code)

            verified_code = self._self_verify_class_code(raw_code, cls, api_context)

            # Validate verified code
            verified_valid, verified_error = _validate_java_code_quality(verified_code)

            # Quality gate: prefer original if verification broke the code
            if raw_valid and not verified_valid:
                print(f"  [QUALITY GATE] Verified code failed validation: {verified_error}")
                print(f"  [QUALITY GATE] Keeping original code instead")
                return raw_code

            return verified_code

        return raw_code

    def _build_generation_prompt(
        self,
        func: FunctionContext,
        class_context: ClassContext | None,
        patterns: ProjectTestPatterns | None,
        api_context: str | None
    ) -> str:
        """Build prompt for direct Java code generation."""

        # Build method signature
        params_str = ", ".join(f"{p.type_hint or 'Object'} {p.name}" for p in func.parameters)
        return_type = func.return_type or "void"
        method_sig = f"{return_type} {func.name}({params_str})"

        # Build class info
        class_info = ""
        if class_context:
            # Extract dependencies from constructor parameters
            deps = []
            if class_context.constructors:
                for constructor in class_context.constructors:
                    for param in constructor.parameters:
                        if param.type_hint:
                            deps.append(param.type_hint)
            deps_str = ", ".join(deps) if deps else "none"

            # Try to extract package from file path
            package_name = "unknown"
            if class_context.location and class_context.location.file_path:
                # Java convention: src/main/java/org/example/Foo.java -> org.example
                path_str = str(class_context.location.file_path)
                if "src/main/java/" in path_str:
                    pkg_path = path_str.split("src/main/java/")[1]
                    pkg_path = pkg_path.rsplit("/", 1)[0]  # Remove filename
                    package_name = pkg_path.replace("/", ".")

            class_info = f"""
Class: {class_context.name}
Package: {package_name}
Dependencies: {deps_str}
"""

        # Build API context section with strong enforcement
        api_section = ""
        if api_context:
            api_section = f"""
═══════════════════════════════════════════════════════════════════════════════
AVAILABLE APIs - USE ONLY THESE METHODS (from actual source code):
═══════════════════════════════════════════════════════════════════════════════
{api_context}

CRITICAL API RULES:
1. ONLY call methods listed above - do NOT invent methods that don't exist
2. Use EXACT import paths shown above - do NOT guess package names
3. If a method isn't listed, it DOES NOT EXIST - don't call it
═══════════════════════════════════════════════════════════════════════════════
"""

        # Build patterns section
        patterns_section = ""
        if patterns and patterns.test_patterns:
            patterns_section = """
## Project Test Patterns
Follow these patterns from the existing codebase:
"""
            for pattern in patterns.test_patterns[:3]:
                patterns_section += f"- {pattern}\n"

        prompt = f"""Generate a complete JUnit 5 test class for this Java method.

## Method Under Test
```java
{method_sig}
```

## Method Body
```java
{func.body or "// Method body not available"}
```
{class_info}
{api_section}
{patterns_section}

## Requirements
1. Use JUnit 5 annotations (@Test, @BeforeEach, @DisplayName)
2. Use Mockito for mocking dependencies (@Mock, @InjectMocks, @ExtendWith(MockitoExtension.class))
3. Include at least 3 test cases:
   - Happy path (normal operation)
   - Edge case (boundary conditions)
   - Error case (exceptions, null inputs)
4. Use descriptive @DisplayName annotations
5. Follow Java naming conventions (testMethodName_scenario_expectedResult)

## CRITICAL RULES - DO NOT VIOLATE
1. Mock variable names MUST match their declarations (if you declare `@Mock private Repo types;`, use `types.method()` NOT `repo.method()`)
2. NO inline method definitions inside test methods (this is invalid Java)
3. Method calls must be on the correct object (use `controller.method()` NOT `parameterObject.method()`)
4. All imports must be valid Java imports

## Output
Output ONLY the complete Java test class. No explanation, no markdown, just Java code:
"""
        return prompt

    def _build_class_generation_prompt(
        self,
        cls: ClassContext,
        patterns: ProjectTestPatterns | None,
        api_context: str | None,
        curated_examples: list[SelectedExample] | None = None,
        project_examples: list[str] | None = None
    ) -> str:
        """Build prompt for generating tests for an entire class."""

        # Build methods list
        methods_info = []
        for method in cls.methods:
            params_str = ", ".join(f"{p.type_hint or 'Object'} {p.name}" for p in method.parameters)
            return_type = method.return_type or "void"
            methods_info.append(f"  {return_type} {method.name}({params_str})")

        methods_section = "\n".join(methods_info)

        # Build dependencies section from constructor parameters AND @Autowired fields
        deps = []

        # From constructor parameters
        if cls.constructors:
            for constructor in cls.constructors:
                for param in constructor.parameters:
                    if param.type_hint:
                        deps.append(f"{param.type_hint} {param.name}")

        # Also detect @Autowired field injections from source code
        source_code_for_deps = ""
        if cls.location and cls.location.file_path:
            try:
                source_code_for_deps = cls.location.file_path.read_text()
            except:
                pass

        if source_code_for_deps:
            import re
            # Match @Autowired followed by field declaration
            autowired_pattern = r'@Autowired\s+(?:private\s+)?(\w+)\s+(\w+)\s*;'
            for match in re.finditer(autowired_pattern, source_code_for_deps):
                field_type, field_name = match.groups()
                dep = f"{field_type} {field_name}"
                if dep not in deps:
                    deps.append(dep)

        deps_section = ""
        if deps:
            deps_list = "\n".join(f"  @Mock\n  private {d};" for d in deps)
            deps_section = f"""
## Dependencies to Mock (MANDATORY PATTERN)
```java
@ExtendWith(MockitoExtension.class)
class {cls.name}Test {{

{deps_list}

  @InjectMocks
  private {cls.name} controller;
}}
```
CRITICAL: Use this EXACT pattern - do NOT use `new {cls.name}()` when there are dependencies!
"""

        # Build API context section with strong enforcement
        api_section = ""
        if api_context:
            api_section = f"""
═══════════════════════════════════════════════════════════════════════════════
AVAILABLE APIs - USE ONLY THESE METHODS (from actual source code):
═══════════════════════════════════════════════════════════════════════════════
{api_context}

CRITICAL API RULES:
1. ONLY call methods listed above - do NOT invent methods that don't exist
2. Use EXACT import paths shown above - do NOT guess package names
3. If a method isn't listed, it DOES NOT EXIST - don't call it
4. Check which CLASS has each method before calling it
═══════════════════════════════════════════════════════════════════════════════
"""

        # Try to extract package from file path
        package_name = "unknown"
        if cls.location and cls.location.file_path:
            path_str = str(cls.location.file_path)
            if "src/main/java/" in path_str:
                pkg_path = path_str.split("src/main/java/")[1]
                pkg_path = pkg_path.rsplit("/", 1)[0]  # Remove filename
                package_name = pkg_path.replace("/", ".")

        # Build examples section with curated gold-standard examples
        examples_section = ""
        if curated_examples:
            examples_section = """
## GOLD-STANDARD TEST EXAMPLES - STUDY THESE CAREFULLY
The following are curated examples showing CORRECT test patterns.
CRITICAL: Follow these patterns exactly, especially for:
- Mocking with Pageable: use `any(Pageable.class)` NOT `any()`
- Exception handling: use `assertThrows` when method uses `orElseThrow`
- Validators: set ALL required fields in valid test case
- Formatters: handle ParseException properly

"""
            for i, example in enumerate(curated_examples, 1):
                examples_section += f"### Pattern: {example.component_type.value}\n"
                examples_section += f"Relevance: {example.reason}\n"
                examples_section += f"```java\n{example.code.strip()}\n```\n\n"

        # Add project-specific examples if available (lower priority)
        if project_examples:
            examples_section += """
## Project-Specific Examples (supplementary reference)
"""
            for i, example in enumerate(project_examples, 1):
                examples_section += f"### Project Example {i}\n```java\n{example}\n```\n\n"

        prompt = f"""Generate a complete JUnit 5 test class for the following Java class.

## Class Under Test
Class: {cls.name}
Package: {package_name}
Test Class Name: {cls.name}Test (IMPORTANT: use exactly this name, NOT {cls.name}Tests)
{deps_section}

## Methods to Test
```java
{methods_section}
```
{api_section}
{examples_section}
## Requirements
1. Use JUnit 5 annotations (@Test, @BeforeEach, @DisplayName)
2. Use Mockito for mocking dependencies (@Mock, @InjectMocks, @ExtendWith(MockitoExtension.class))
3. For EACH method, include at least 2 test cases (happy path + edge/error case)
4. Use descriptive @DisplayName annotations
5. Declare the test instance variable as `private {cls.name} controller;` or `private {cls.name} instance;`

## CRITICAL RULES - DO NOT VIOLATE
1. NEVER use `new {cls.name}()` for classes with dependencies - use @InjectMocks instead
2. Mock variable names MUST match their declarations throughout the class
3. NO inline method definitions inside test methods
4. Method calls must be on the test instance (controller/instance), NOT on parameter objects
5. All imports must be valid
6. Classes with @Autowired dependencies MUST use @ExtendWith(MockitoExtension.class), @Mock, @InjectMocks

## Output
Output ONLY the complete Java test class. No explanation, no markdown, just Java code:
"""
        return prompt

    def _self_verify_code(
        self,
        code: str,
        func: FunctionContext,
        class_context: ClassContext | None
    ) -> str:
        """Have LLM review and fix its own generated code."""

        prompt = f"""Review and fix this Java test code. Check for common errors and fix them.

## Generated Test Code
```java
{code}
```

## Common Errors to Check
1. **Mock variable consistency**: If `@Mock private TypeRepository types;` is declared,
   all usages must be `types.method()`, NOT `typeRepository.method()`

2. **Inline method definitions**: Java does NOT allow method definitions inside other methods.
   This is INVALID: `PetType createPetType(String name) {{ ... }};`

3. **Method calls on wrong objects**: The method under test should be called on the test instance
   (controller/instance), NOT on parameter objects like `petType.parse(...)`.

4. **Import completeness**: All used classes must be imported.

5. **Test logic correctness**: Mock setup must match how the method actually uses dependencies.
   For example, if testing `owner.getPet(petId)`, the owner mock must return a pet with that ID.

## Task
Fix any errors you find. If the code is correct, return it unchanged.

## Output
Output ONLY the corrected Java code. No explanation:
"""

        verified = self._call_llm(prompt, use_cache=False)  # Don't cache verification
        return self._extract_java_code(verified)

    def _self_verify_class_code(self, code: str, cls: ClassContext, api_context: str | None = None) -> str:
        """Have LLM review and fix generated class test code."""

        # Extract dependencies from constructor parameters
        deps = []
        if cls.constructors:
            for constructor in cls.constructors:
                for param in constructor.parameters:
                    if param.type_hint:
                        deps.append(param.type_hint)
        deps_list = ", ".join(deps) if deps else "none"

        # Build API validation section if context available
        api_validation = ""
        if api_context:
            api_validation = f"""
6. **API Validation - CRITICAL**: Check that ALL method calls exist in this API list:
═══════════════════════════════════════════════════════════════════════════════
{api_context}
═══════════════════════════════════════════════════════════════════════════════
   - If you see a method call that's NOT in the list above, REMOVE that test method entirely
   - If you see an import that doesn't match the paths above, FIX the import path
   - Do NOT invent methods - only use what's listed above
"""

        prompt = f"""Review and fix this Java test class. The tests are for class `{cls.name}`.

## Generated Test Code
```java
{code}
```

## Class Being Tested
- Name: {cls.name}
- Dependencies to mock: {deps_list}

## Common Errors to Check and Fix
1. **Mock variable consistency**: Declared mock names must match usages everywhere
2. **No inline method definitions**: Java doesn't allow methods inside methods
3. **Correct object for method calls**: Call methods on controller/instance, not parameters
4. **Complete imports**: All classes used must be imported
5. **Semantic correctness**: Mock setups should match how the real code uses dependencies
{api_validation}

## Output
Output ONLY the corrected Java code. No explanation:
"""

        verified = self._call_llm(prompt, use_cache=False)
        return self._extract_java_code(verified)

    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response, removing markdown if present."""
        text = response.strip()

        # Remove markdown code blocks
        if "```java" in text:
            match = re.search(r"```java\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        if "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()

        return text

    def _extract_dependencies(self, cls: ClassContext) -> list[tuple[str, str]]:
        """
        Extract dependencies from class (constructor params and @Autowired fields).

        Returns list of (type_name, var_name) tuples.
        """
        deps = []

        # From constructor parameters
        if cls.constructors:
            for constructor in cls.constructors:
                for param in constructor.parameters:
                    if param.type_hint:
                        deps.append((param.type_hint, param.name))

        # From @Autowired fields
        if cls.location and cls.location.file_path:
            try:
                source_code = cls.location.file_path.read_text()
                autowired_pattern = r'@Autowired\s+(?:private\s+)?(\w+)\s+(\w+)\s*;'
                for match in re.finditer(autowired_pattern, source_code):
                    field_type, field_name = match.groups()
                    if (field_type, field_name) not in deps:
                        deps.append((field_type, field_name))
            except:
                pass

        return deps

    def _fix_missing_mock_pattern(self, code: str, class_name: str, deps: list[tuple[str, str]]) -> str:
        """
        Fix test code that uses new ClassName() when it should use @Mock/@InjectMocks.

        Detects when:
        - Code has @ExtendWith(MockitoExtension.class) but no @Mock declarations
        - setUp() uses new ClassName()

        And rewrites to proper pattern.
        """
        # Check if code already has @Mock declarations
        if '@Mock' in code:
            return code  # Already has mocks, nothing to fix

        # Check if code has @ExtendWith(MockitoExtension.class)
        if '@ExtendWith(MockitoExtension.class)' not in code:
            return code  # Not using Mockito, leave as-is

        # Check if setUp uses new ClassName()
        if f'new {class_name}()' not in code:
            return code  # Not using direct instantiation

        # Need to fix: replace direct instantiation with @Mock/@InjectMocks pattern
        lines = code.split('\n')
        fixed_lines = []
        in_class = False
        added_mocks = False
        found_instance_var = False

        for i, line in enumerate(lines):
            # Track when we enter the class body
            if f'class {class_name}Test' in line:
                in_class = True
                fixed_lines.append(line)
                continue

            # After class declaration and before any method, add mocks
            if in_class and not added_mocks:
                # Check if this line is the start of class body (first non-empty after class)
                stripped = line.strip()

                # If we see 'private ClassName' variable, replace it
                if re.match(rf'^\s*private\s+{class_name}\s+\w+\s*;', line):
                    # Skip this line, we'll add proper @InjectMocks version
                    found_instance_var = True
                    # Add @Mock declarations first
                    fixed_lines.append('')
                    for dep_type, dep_name in deps:
                        fixed_lines.append(f'    @Mock')
                        fixed_lines.append(f'    private {dep_type} {dep_name};')
                    fixed_lines.append('')
                    # Add @InjectMocks
                    fixed_lines.append('    @InjectMocks')
                    fixed_lines.append(f'    private {class_name} controller;')
                    added_mocks = True
                    continue

                # If we see @BeforeEach without having found the instance var, add mocks here
                if '@BeforeEach' in line and not found_instance_var:
                    # Add mocks before @BeforeEach
                    fixed_lines.append('')
                    for dep_type, dep_name in deps:
                        fixed_lines.append(f'    @Mock')
                        fixed_lines.append(f'    private {dep_type} {dep_name};')
                    fixed_lines.append('')
                    fixed_lines.append('    @InjectMocks')
                    fixed_lines.append(f'    private {class_name} controller;')
                    fixed_lines.append('')
                    added_mocks = True
                    # Now add the @BeforeEach and skip the setUp method entirely
                    # Since @InjectMocks handles instantiation, we may not need setUp
                    # But keep it for now if it has other setup logic

            # Skip lines that create new instance in setUp
            if f'new {class_name}()' in line:
                # Check if this is an assignment to controller/instance
                if re.match(rf'^\s*(controller|instance)\s*=\s*new\s+{class_name}\s*\(\s*\)\s*;', line.strip()):
                    continue  # Skip this line, @InjectMocks handles it

            # Replace 'instance' with 'controller' if we added @InjectMocks
            if added_mocks and 'instance.' in line:
                line = line.replace('instance.', 'controller.')

            fixed_lines.append(line)

        # Add import for Mock and InjectMocks if needed
        result = '\n'.join(fixed_lines)
        if added_mocks:
            if 'import org.mockito.Mock;' not in result:
                # Add import after other mockito imports
                result = result.replace(
                    'import org.mockito.junit.jupiter.MockitoExtension;',
                    'import org.mockito.InjectMocks;\nimport org.mockito.Mock;\nimport org.mockito.junit.jupiter.MockitoExtension;'
                )

        return result

    def _is_complete_java_code(self, code: str) -> bool:
        """Check if Java code appears complete (not truncated)."""
        if not code or len(code.strip()) < 50:
            return False
        stripped = code.rstrip()
        # Must end with closing brace
        if not stripped.endswith('}'):
            return False
        # Braces must be balanced
        if stripped.count('{') != stripped.count('}'):
            return False
        return True

    def _call_llm(self, prompt: str, use_cache: bool = True) -> str:
        """Call the LLM API with caching and truncation handling."""
        # Check cache first
        cache_key = _get_cache_key(prompt, self.model)
        if use_cache and cache_key in _LLM_CACHE:
            return _LLM_CACHE[cache_key]

        # Call LLM
        if self.provider == "openrouter":
            response, was_truncated = self._call_openrouter(prompt)
        else:
            response, was_truncated = self._call_anthropic(prompt)

        # Check for truncation or incomplete code
        code = self._extract_java_code(response)
        is_incomplete = was_truncated or not self._is_complete_java_code(code)

        if is_incomplete:
            # Re-request with explicit brevity instruction (don't try to stitch)
            retry_prompt = f"""Your previous response was truncated. Generate a COMPLETE but CONCISE test class.

IMPORTANT: Keep tests SHORT to avoid truncation:
- Maximum 4 test methods
- Each test should be 5-10 lines max
- No verbose comments
- Use simple assertions

Generate the COMPLETE test class from scratch. Must end with closing brace.

{prompt}"""

            if self.provider == "openrouter":
                response, _ = self._call_openrouter(retry_prompt)
            else:
                response, _ = self._call_anthropic(retry_prompt)

        # Cache response
        if use_cache:
            _save_to_cache(cache_key, response)

        return response

    def _call_anthropic(self, prompt: str) -> tuple[str, bool]:
        """Call Anthropic API directly.

        Returns:
            tuple: (response_text, was_truncated)
        """
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        system_prompt = self._get_system_prompt()
        message = client.messages.create(
            model=self.model,
            max_tokens=8000,  # Increased from 4000
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        # Check if response was truncated
        was_truncated = message.stop_reason == "max_tokens"
        return message.content[0].text, was_truncated

    def _call_openrouter(self, prompt: str) -> tuple[str, bool]:
        """Call OpenRouter API.

        Returns:
            tuple: (response_text, was_truncated)
        """
        system_prompt = self._get_system_prompt()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/java-test-agent",
            "X-Title": "Java Test Agent"
        }

        payload = {
            "model": self.model,
            "max_tokens": 8000,  # Increased from 4000
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        # Retry with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120  # Increased timeout for larger responses
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    # Check if response was truncated (finish_reason == "length")
                    finish_reason = result["choices"][0].get("finish_reason", "stop")
                    was_truncated = finish_reason == "length"
                    return content, was_truncated

                # Retry on rate limit or server errors
                if response.status_code in (429, 502, 503, 504):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        time.sleep(wait_time)
                        continue

                raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

            except requests.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise RuntimeError("OpenRouter API timeout after retries")

            except requests.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise RuntimeError(f"OpenRouter connection error: {e}")

        raise RuntimeError("OpenRouter API failed after max retries")

    def _get_system_prompt(self) -> str:
        """Get system prompt for Java test generation."""
        return """You are an expert Java developer specializing in writing JUnit 5 tests.

Your task is to generate high-quality, compilable Java test code.

CRITICAL RULES:
1. Output ONLY valid Java code - no explanations, no markdown formatting
2. Mock variable names must be consistent (if declared as `types`, use `types` everywhere)
3. Never define methods inside other methods (this is invalid Java)
4. Call methods under test on the test instance (controller/instance), not on parameter objects
5. Include all necessary imports

Follow Java best practices and JUnit 5 conventions."""
