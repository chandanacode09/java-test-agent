"""
ReAct Loop for Java/Maven/JUnit - Iterative test improvement through execution feedback.

This module implements the Reason-Act-Observe loop for Java tests:
1. Reason: Analyze JUnit test failures with error categorization
2. Act: Fix tests using LLM with strategy-specific prompts
3. Observe: Run Maven tests and check results
4. Reflect: Learn from failed fix attempts
5. Repeat until success or max iterations
"""

import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import xml.etree.ElementTree as ET

# Import AST parser for dynamic API discovery
from src.context.ast_parser import ASTParser
from src.models import ErrorCategory, FixAttempt, ClassContext, FunctionContext, CodeLocation


@dataclass
class JavaTestFailure:
    """Represents a single JUnit test failure."""
    test_name: str
    test_class: str
    test_file: Path
    error_type: str
    error_message: str
    stack_trace: str


@dataclass
class JavaReactResult:
    """Result of a ReAct iteration."""
    iteration: int
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    failures: list[JavaTestFailure]
    fixed_in_this_iteration: int
    success: bool


class JavaReActLoop:
    """
    Implements the ReAct loop for iterative Java test improvement.

    Flow:
    1. Run Maven tests
    2. Parse Surefire reports for failures
    3. Ask LLM to fix each failure
    4. Update test files
    5. Repeat until all pass or max iterations
    """

    # NOTE: FIX_PROMPT_TEMPLATE removed - now using _get_fix_prompt() which provides:
    # - Dynamic API context from AST parsing
    # - Error categorization with strategy-specific instructions
    # - Self-reflection from previous fix attempts

    def __init__(
        self,
        project_path: Path,
        api_key: str,
        test_classes: list[str] = None,
        max_iterations: int = 3,
        verbose: bool = True,
        model: str = "anthropic/claude-3.5-haiku"
    ):
        self.project_path = Path(project_path)
        self.api_key = api_key
        self.test_classes = test_classes or []
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.model = model

        # Dynamic API discovery cache
        self._discovered_classes: dict[str, ClassContext] = {}

        # Self-reflection: track fix attempts per test
        self._fix_history: dict[str, list[FixAttempt]] = {}

        # Test method cache - parsed test files using AST
        self._test_method_cache: dict[Path, dict[str, FunctionContext]] = {}

    def run(self) -> list[JavaReactResult]:
        """
        Run the ReAct loop on Java tests.

        Returns:
            List of JavaReactResult for each iteration
        """
        results = []

        # Step 0: Discover project APIs dynamically (no hardcoding!)
        self._discover_project_apis()

        # Show discovered classes summary
        if self._discovered_classes:
            for cls_name in list(self._discovered_classes.keys())[:5]:
                cls = self._discovered_classes[cls_name]
                self._log(f"    - {cls_name}: {len(cls.constructors)} ctors, {len(cls.methods)} methods")
            if len(self._discovered_classes) > 5:
                self._log(f"    ... and {len(self._discovered_classes) - 5} more classes")

        # Step 1: Fix any compilation errors first
        compiles, error_msg = self._verify_compilation()
        if not compiles:
            self._log("\n[PRE-STEP] Fixing compilation errors...")
            compile_fixes = self._fix_compilation_errors(error_msg)
            self._log(f"  Applied {compile_fixes} compilation fixes")

        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n{'='*60}")
            self._log(f"Java ReAct Iteration {iteration}/{self.max_iterations}")
            self._log(f"{'='*60}")

            # Step 2: Run Maven tests
            self._log("\n[OBSERVE] Running Maven tests...")
            test_output, exit_code = self._run_maven_tests()

            # Step 3: Parse Surefire results
            total, passed, failed, errors, failures = self._parse_surefire_reports()

            self._log(f"Results: {passed}/{total} passed, {failed} failed, {errors} errors")

            if failed == 0 and errors == 0:
                self._log("\n[SUCCESS] All tests passing!")
                results.append(JavaReactResult(
                    iteration=iteration,
                    total_tests=total,
                    passed_tests=passed,
                    failed_tests=0,
                    error_tests=0,
                    failures=[],
                    fixed_in_this_iteration=0,
                    success=True
                ))
                break

            # Step 4: Reason about failures WITH CATEGORIZATION
            self._log(f"\n[REASON] Analyzing {len(failures)} failures with error categorization...")
            for f in failures:
                category = self._categorize_error(f)
                self._log(f"  - {f.test_class}.{f.test_name}: [{category.value}] {f.error_type[:50]}")

            # Step 5: Act - fix each failure with strategy-specific prompts
            self._log(f"\n[ACT] Fixing failures with targeted strategies...")
            fixed_count = 0

            for failure in failures:
                self._log(f"  Fixing: {failure.test_class}.{failure.test_name}")
                fixed = self._fix_failure(failure)
                if fixed:
                    fixed_count += 1
                    self._log(f"    [OK] Fixed")
                else:
                    self._log(f"    [FAIL] Could not fix")

            results.append(JavaReactResult(
                iteration=iteration,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                error_tests=errors,
                failures=failures,
                fixed_in_this_iteration=fixed_count,
                success=False
            ))

            if fixed_count == 0:
                self._log("\n[STOP] No fixes applied, stopping iteration")
                break

        return results

    def _run_maven_tests(self) -> tuple[str, int]:
        """Run Maven tests and capture output."""
        # Build test filter if specific classes provided
        test_filter = ""
        if self.test_classes:
            test_filter = f"-Dtest={','.join(self.test_classes)}"

        cmd = ["./mvnw", "test", "-q"]
        if test_filter:
            cmd.append(test_filter)

        self._log(f"  Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=120
        )

        return result.stdout + result.stderr, result.returncode

    def _parse_surefire_reports(self) -> tuple[int, int, int, int, list[JavaTestFailure]]:
        """Parse Surefire XML reports to extract test results and failures."""
        surefire_dir = self.project_path / "target" / "surefire-reports"

        if not surefire_dir.exists():
            self._log(f"  Warning: Surefire reports directory not found: {surefire_dir}")
            return 0, 0, 0, 0, []

        total = 0
        passed = 0
        failed = 0
        errors = 0
        failures = []

        # Filter to only our test classes if specified
        filter_classes = set(self.test_classes) if self.test_classes else None

        for xml_file in surefire_dir.glob("TEST-*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                class_name = root.get("name", "")

                # Skip if we have a filter and this class isn't in it
                if filter_classes:
                    short_name = class_name.split(".")[-1]
                    if short_name not in filter_classes and class_name not in filter_classes:
                        continue

                tests = int(root.get("tests", 0))
                test_failures = int(root.get("failures", 0))
                test_errors = int(root.get("errors", 0))

                total += tests
                failed += test_failures
                errors += test_errors
                passed += tests - test_failures - test_errors

                # Parse individual test cases for failure details
                for testcase in root.findall(".//testcase"):
                    test_name = testcase.get("name", "")

                    # Check for failure
                    failure_elem = testcase.find("failure")
                    error_elem = testcase.find("error")

                    if failure_elem is not None:
                        failures.append(JavaTestFailure(
                            test_name=test_name,
                            test_class=class_name,
                            test_file=self._find_test_file(class_name),
                            error_type=failure_elem.get("type", "AssertionError"),
                            error_message=failure_elem.get("message", "")[:300],
                            stack_trace=(failure_elem.text or "")[:500]
                        ))
                    elif error_elem is not None:
                        failures.append(JavaTestFailure(
                            test_name=test_name,
                            test_class=class_name,
                            test_file=self._find_test_file(class_name),
                            error_type=error_elem.get("type", "Error"),
                            error_message=error_elem.get("message", "")[:300],
                            stack_trace=(error_elem.text or "")[:500]
                        ))

            except ET.ParseError as e:
                self._log(f"  Warning: Could not parse {xml_file}: {e}")
                continue

        return total, passed, failed, errors, failures

    def _find_test_file(self, class_name: str) -> Path:
        """Find the Java test file for a given class name."""
        # Convert package.ClassName to path
        # e.g., org.springframework.samples.petclinic.eval.OwnerGetpetTest
        #    -> org/springframework/samples/petclinic/eval/OwnerGetpetTest.java
        parts = class_name.split(".")
        class_file = parts[-1] + ".java"
        package_path = "/".join(parts[:-1])  # Convert package to path

        # Search in test directories - prefer exact package match
        for test_dir in [
            self.project_path / "src" / "test" / "java",
            self.project_path / "src" / "test",
        ]:
            if test_dir.exists():
                # First, try exact package path match
                exact_path = test_dir / package_path / class_file
                if exact_path.exists():
                    return exact_path

                # Fallback: search for the file (less precise)
                for java_file in test_dir.rglob(class_file):
                    # Verify the package matches by checking the path
                    relative = java_file.relative_to(test_dir)
                    if package_path in str(relative):
                        return java_file

        return Path(class_file)

    def _parse_test_file(self, test_file: Path, force_refresh: bool = False) -> dict[str, FunctionContext]:
        """
        Parse test file to get test methods.

        Uses a custom pattern that handles package-private methods (no visibility modifier)
        which are common in JUnit 5 tests.

        Results are cached to avoid re-parsing the same file, but cache is invalidated
        when force_refresh=True (e.g., after file modifications).
        """
        # Check cache first (unless force refresh)
        if not force_refresh and test_file in self._test_method_cache:
            return self._test_method_cache[test_file]

        if not test_file.exists():
            self._test_method_cache[test_file] = {}
            return {}

        try:
            content = test_file.read_text()
            lines = content.split('\n')

            # Pattern for test methods - handles package-private (no visibility modifier)
            # Matches: void methodName() { or public void methodName() {
            method_pattern = re.compile(
                r'^[ \t]*'                                    # Leading whitespace
                r'(?:public\s+|private\s+|protected\s+)?'     # Optional visibility
                r'(?:static\s+)?'                             # Optional static
                r'void\s+'                                    # Return type (void for tests)
                r'(\w+)\s*'                                   # Method name (captured)
                r'\([^)]*\)\s*'                               # Parameters
                r'(?:throws\s+[\w,\s]+)?\s*'                  # Optional throws
                r'\{',                                        # Opening brace
                re.MULTILINE
            )

            test_methods = {}

            for match in method_pattern.finditer(content):
                method_name = match.group(1)
                method_start_pos = match.start()
                method_start_line = content[:method_start_pos].count('\n') + 1

                # Find the method end by counting braces
                brace_count = 1
                pos = match.end()
                while pos < len(content) and brace_count > 0:
                    if content[pos] == '{':
                        brace_count += 1
                    elif content[pos] == '}':
                        brace_count -= 1
                    pos += 1

                method_end_pos = pos
                method_end_line = content[:method_end_pos].count('\n') + 1

                # Extract full method source including annotations
                # Look back for @Test and @DisplayName annotations
                annotation_start = method_start_pos
                search_start = max(0, method_start_pos - 200)
                before_method = content[search_start:method_start_pos]

                # Find the last annotation line before the method
                annotation_lines = []
                for line in reversed(before_method.split('\n')):
                    stripped = line.strip()
                    if stripped.startswith('@'):
                        annotation_lines.insert(0, line)
                    elif stripped and not stripped.startswith('//') and not stripped.startswith('*'):
                        break

                # Build source code
                method_source = content[method_start_pos:method_end_pos]
                if annotation_lines:
                    method_source = '\n'.join(annotation_lines) + '\n' + method_source

                # Only include methods with @Test annotation
                if '@Test' in method_source or method_name.startswith('test'):
                    test_methods[method_name] = FunctionContext(
                        name=method_name,
                        location=CodeLocation(
                            file_path=test_file,
                            start_line=method_start_line,
                            end_line=method_end_line
                        ),
                        parameters=[],
                        return_type=None,
                        docstring=None,
                        is_async=False,
                        is_method=True,
                        class_name=test_file.stem,
                        decorators=[],
                        source_code=method_source,
                        calls=[],
                        imports=[]
                    )

            self._test_method_cache[test_file] = test_methods
            return test_methods

        except Exception as e:
            self._log(f"    Warning: Could not parse test file {test_file}: {e}")
            self._test_method_cache[test_file] = {}
            return {}

    # =========================================================================
    # DYNAMIC API DISCOVERY (replaces hardcoded class lists)
    # =========================================================================

    def _discover_project_apis(self) -> None:
        """
        Dynamically discover all APIs in the project source code using AST parser.

        Parses all Java files in src/main/java to extract:
        - Class names and their inheritance
        - Constructors with parameters
        - Public methods with signatures

        Results are cached in self._discovered_classes.
        """
        source_dir = self.project_path / "src" / "main" / "java"

        if not source_dir.exists():
            self._log(f"  Warning: Source directory not found: {source_dir}")
            return

        self._log(f"\n[DISCOVER] Parsing source code in {source_dir}...")

        parser = ASTParser()

        for java_file in source_dir.rglob("*.java"):
            try:
                content = java_file.read_text()
                functions, classes = parser._parse_java_regex(content, java_file)

                for cls in classes:
                    self._discovered_classes[cls.name] = cls

            except Exception as e:
                self._log(f"  Warning: Could not parse {java_file.name}: {e}")

        self._log(f"  Discovered {len(self._discovered_classes)} classes")

    def _build_api_context_for_test(self, test_class: str) -> str:
        """
        Build API context dynamically based on test class name.

        Instead of hardcoding which classes to include, we:
        1. Parse the test class name to identify the target class
        2. Include that class's APIs (constructors + methods)
        3. Include related classes (via inheritance, method parameters/returns)
        """
        if not self._discovered_classes:
            return "// No API information discovered"

        # Extract target class from test name dynamically
        short_name = test_class.split(".")[-1]  # e.g., OwnerAddpetTest

        # Remove common test suffixes
        target_class = short_name
        for suffix in ["Test", "Tests", "IT", "IntegrationTest"]:
            if target_class.endswith(suffix):
                target_class = target_class[:-len(suffix)]
                break

        # Try to find the main class being tested by checking prefixes
        # This is dynamic - works with any class name pattern
        found_target = None
        for cls_name in self._discovered_classes.keys():
            if target_class.startswith(cls_name):
                found_target = cls_name
                break

        if not found_target:
            # Fallback: try to match the first word (assumes ClassNameMethodTest pattern)
            match = re.match(r'^([A-Z][a-z]+)', target_class)
            if match:
                potential = match.group(1)
                if potential in self._discovered_classes:
                    found_target = potential

        # Collect relevant classes
        relevant_classes: set[str] = set()

        if found_target:
            relevant_classes.add(found_target)

            # Add parent classes (inheritance chain)
            to_process = [found_target]
            while to_process:
                current = to_process.pop()
                if current in self._discovered_classes:
                    cls = self._discovered_classes[current]
                    for parent in cls.base_classes:
                        if parent in self._discovered_classes:
                            relevant_classes.add(parent)
                            to_process.append(parent)

            # Add classes referenced in method signatures
            if found_target in self._discovered_classes:
                cls = self._discovered_classes[found_target]
                for method in cls.methods:
                    # Check return type
                    if method.return_type:
                        clean_type = re.sub(r'[<>\[\]].*', '', method.return_type)
                        if clean_type in self._discovered_classes:
                            relevant_classes.add(clean_type)
                    # Check parameter types
                    for param in method.parameters:
                        if param.type_hint:
                            clean_type = re.sub(r'[<>\[\]].*', '', param.type_hint)
                            if clean_type in self._discovered_classes:
                                relevant_classes.add(clean_type)

        # Build context string
        context_lines = []
        for class_name in sorted(relevant_classes):
            if class_name not in self._discovered_classes:
                continue

            cls = self._discovered_classes[class_name]
            context_lines.append(f"\n// === {class_name} ===")

            # Constructors
            for ctor in cls.constructors:
                params = ", ".join([f"{p.type_hint or 'Object'} {p.name}" for p in ctor.parameters])
                context_lines.append(f"Constructor: {class_name}({params})")

            # Methods
            for method in cls.methods:
                params = ", ".join([f"{p.type_hint or 'Object'} {p.name}" for p in method.parameters])
                ret = method.return_type or "void"
                context_lines.append(f"Method: {ret} {method.name}({params})")

        if not context_lines:
            return f"// No API information found for {test_class}"

        return "\n".join(context_lines)

    # =========================================================================
    # ERROR CATEGORIZATION
    # =========================================================================

    def _categorize_error(self, failure: JavaTestFailure) -> ErrorCategory:
        """
        Categorize the error type to select appropriate fix strategy.

        Analyzes error type, message, and stack trace to classify the failure.
        """
        error_type = failure.error_type.lower()
        error_msg = failure.error_message.lower()
        stack_trace = failure.stack_trace.lower()

        # NPE detection
        if "nullpointerexception" in error_type or "nullpointerexception" in stack_trace:
            return ErrorCategory.NULL_POINTER

        # IllegalArgumentException
        if "illegalargumentexception" in error_type or "illegalargumentexception" in stack_trace:
            return ErrorCategory.ILLEGAL_ARGUMENT

        # Assertion failures
        if "assertionerror" in error_type or "assertionfailure" in error_type:
            return ErrorCategory.ASSERTION_FAILURE
        if "expected" in error_msg and ("but was" in error_msg or "but got" in error_msg):
            return ErrorCategory.ASSERTION_FAILURE
        if "expected" in error_msg and "to be thrown" in error_msg:
            return ErrorCategory.ASSERTION_FAILURE
        if "expected: not <null>" in error_msg or "expected not <null>" in error_msg:
            return ErrorCategory.ASSERTION_FAILURE

        # Method not found / wrong arguments (compilation errors)
        if "cannot find symbol" in error_msg or "cannot resolve" in error_msg:
            if "method" in error_msg:
                return ErrorCategory.MISSING_METHOD
            if "class" in error_msg:
                return ErrorCategory.CLASS_NOT_FOUND

        # Constructor argument mismatch
        if "no suitable constructor" in error_msg or "constructor" in error_msg:
            return ErrorCategory.WRONG_ARGUMENTS

        # Method argument mismatch
        if "cannot be applied" in error_msg or "incompatible types" in error_msg:
            return ErrorCategory.WRONG_ARGUMENTS

        # Type mismatch
        if "incompatible types" in error_msg or "type mismatch" in error_msg:
            return ErrorCategory.TYPE_MISMATCH

        # General compilation error
        if "error" in error_type and ("compile" in error_type or "compile" in error_msg):
            return ErrorCategory.COMPILATION_ERROR

        return ErrorCategory.UNKNOWN

    def _get_strategy_instructions(self, error_category: ErrorCategory) -> str:
        """Get strategy-specific instructions based on error category."""

        strategies = {
            ErrorCategory.NULL_POINTER: """
FIX STRATEGY FOR NULL POINTER EXCEPTION:
1. Identify which variable is null from the stack trace
2. Trace back to find where that variable should have been initialized
3. Common causes:
   - Method returns null when object not found - ADD the object first before calling getter
   - Getter called before setter - initialize the property first
   - Collection not initialized - create empty collection
4. Add proper setup/initialization before the failing line

SPECIAL CASE - "Expected NullPointerException to be thrown, but nothing was thrown":
   The test uses assertThrows(NullPointerException.class, ...) but the method does NOT throw.
   FIX: Replace assertThrows with assertDoesNotThrow or just call the method directly:
   ```java
   // WRONG - method doesn't throw NPE
   assertThrows(NullPointerException.class, () -> instance.addX(null));

   // CORRECT - just verify it doesn't crash
   assertDoesNotThrow(() -> instance.addX(null));
   ```
""",

            ErrorCategory.ASSERTION_FAILURE: """
FIX STRATEGY FOR ASSERTION FAILURE:
1. Compare expected vs actual values in the error message
2. Determine if the test expectation is wrong OR the setup is incomplete

WORKING EXAMPLE - Test that retrieves a pet by name:
```java
@Test
void testGetPetByName() {
    Owner owner = new Owner();

    // STEP 1: Create and configure the pet
    Pet pet = new Pet();
    pet.setName("Fluffy");
    PetType type = new PetType();
    type.setName("Dog");
    pet.setType(type);

    // STEP 2: Add pet to owner FIRST
    owner.addPet(pet);

    // STEP 3: Now retrieve - this will NOT be null
    Pet result = owner.getPet("Fluffy");
    assertNotNull(result);
    assertEquals("Fluffy", result.getName());
}
```

3. Common fixes:
   - "expected not <null>" = object not added before retrieval. ADD IT FIRST!
   - "expected X to be thrown" = method doesn't throw. Remove assertThrows, just call method.
4. assertEquals(expected, actual) - expected comes FIRST
""",

            ErrorCategory.ILLEGAL_ARGUMENT: """
FIX STRATEGY FOR ILLEGAL ARGUMENT EXCEPTION:
1. The method is REJECTING your input due to validation
2. Read the error message to see WHICH argument is invalid

WORKING EXAMPLE - Test addVisit with proper setup:
```java
@Test
void testAddVisitHappyPath() {
    Owner owner = new Owner();

    // STEP 1: Create pet with an ID (simulates persisted entity)
    Pet pet = new Pet();
    pet.setName("Buddy");
    pet.setId(1);  // CRITICAL: Set ID before adding

    // STEP 2: Add pet to owner
    owner.addPet(pet);

    // STEP 3: Now addVisit works because pet with ID=1 exists
    Visit visit = new Visit();
    visit.setDescription("Checkup");
    owner.addVisit(1, visit);  // Uses the pet's ID

    // Verify
    assertEquals(1, pet.getVisits().size());
}
```

3. Common causes:
   - "Pet identifier must not be null" = pet needs an ID set
   - "Invalid Pet identifier" = pet with that ID doesn't exist, ADD it first with setId()
4. Always set entity IDs when testing ID-based lookups
""",

            ErrorCategory.MISSING_METHOD: """
FIX STRATEGY FOR MISSING METHOD:
1. Check the DISCOVERED API SIGNATURES for the correct method name
2. Method names are case-sensitive (getPet != getpet != GetPet)
3. Check if the method exists on the object you're calling it on
4. The method might be on a different class - check inheritance
""",

            ErrorCategory.WRONG_ARGUMENTS: """
FIX STRATEGY FOR WRONG ARGUMENTS:
1. Check DISCOVERED API SIGNATURES for the exact constructor/method signature
2. Common issues:
   - Passing arguments to no-arg constructor - use setters instead:
     CORRECT: Pet pet = new Pet(); pet.setName("x");
     WRONG:   new Pet("x")
   - Wrong number of arguments - match the signature exactly
   - Wrong argument types - check type compatibility
3. For domain objects, ALWAYS prefer: new Foo(); foo.setBar(value);
""",

            ErrorCategory.CLASS_NOT_FOUND: """
FIX STRATEGY FOR CLASS NOT FOUND:
1. Check if the class name is spelled correctly (case-sensitive)
2. Verify the import statement is present and correct
3. Check DISCOVERED API SIGNATURES for available classes
""",

            ErrorCategory.TYPE_MISMATCH: """
FIX STRATEGY FOR TYPE MISMATCH:
1. Check the expected vs actual types in the error
2. Common fixes:
   - Use correct wrapper type (Integer vs int)
   - Convert between types (String.valueOf(), Integer.parseInt())
3. Check method return types in DISCOVERED API SIGNATURES
""",

            ErrorCategory.COMPILATION_ERROR: """
FIX STRATEGY FOR COMPILATION ERROR:
1. Look for syntax errors (missing semicolons, braces, etc.)
2. Check for undefined variables
3. Verify method calls match signatures in DISCOVERED API SIGNATURES
""",

            ErrorCategory.UNKNOWN: """
FIX STRATEGY FOR UNKNOWN ERROR:
1. Read the error message carefully
2. Check the stack trace for the exact failure point
3. Compare test code against DISCOVERED API SIGNATURES
4. Ensure proper object setup before method calls
"""
        }

        return strategies.get(error_category, strategies[ErrorCategory.UNKNOWN])

    # =========================================================================
    # SEMANTIC VERIFICATION (Pre-apply validation)
    # =========================================================================

    def _verify_fix_semantics(
        self,
        test_name: str,
        fixed_code: str,
        error_msg: str
    ) -> tuple[bool, str]:
        """
        Verify the fix makes semantic sense BEFORE applying.

        This is a general-purpose verifier that catches common LLM mistakes:
        1. Setting variables to null when comments suggest non-null
        2. Using wrong assertion types for the test intent
        3. Passing null params when test doesn't test null handling

        Returns:
            (is_valid, reason_if_invalid)
        """
        issues = []

        # 1. Check for null assignments that contradict comments
        # Pattern: finds "variable = null;" assignments
        null_assignments = re.findall(r'(\w+)\s*=\s*null\s*;', fixed_code)
        for var in null_assignments:
            # Check if there's a comment suggesting this should be non-null
            # Look for patterns like "valid birthDate", "required type", "non-null value"
            if re.search(rf'(valid|required|non-?null|must be set|not null)[^;]*{var}', fixed_code, re.I):
                issues.append(f"'{var}' set to null but comments suggest it should be non-null")
            # Also check if variable name suggests it should have a value
            if var.lower() in ['birthdate', 'date', 'type', 'name'] and 'null' not in test_name.lower():
                # Check if the context suggests this should be set
                if re.search(rf'(add|set|create|valid)[^;]*{var}', fixed_code, re.I):
                    issues.append(f"'{var}' set to null but appears to need a real value")

        # 2. Check assertion type matches test intent (from test name)
        test_name_lower = test_name.lower()

        # First check if this is explicitly a "Throws" test - those ARE expected to test exceptions
        is_throws_test = 'throws' in test_name_lower

        # Tests with "Passes" or "Valid" or "HappyPath" should NOT use assertThrows
        # UNLESS they also contain "Throws" (which takes precedence)
        if not is_throws_test:
            if any(x in test_name_lower for x in ['passes', 'valid', 'happypath', 'accepts', 'success']):
                if 'assertThrows' in fixed_code:
                    issues.append("Test name suggests success case but uses assertThrows")
                # Should use assertFalse(errors.hasErrors()) for validation tests
                if 'hasErrors()' in fixed_code and 'assertTrue' in fixed_code:
                    # Check if assertTrue is used with hasErrors (wrong - should be assertFalse)
                    if re.search(r'assertTrue\s*\([^)]*hasErrors\s*\(\s*\)', fixed_code):
                        issues.append("Using assertTrue(hasErrors()) but test expects valid input (should be assertFalse)")

        # Tests with "Throws" should use assertThrows UNLESS error says "nothing was thrown"
        # In that case, the method doesn't throw, so we should use assertDoesNotThrow or just call it
        if is_throws_test:
            if 'nothing was thrown' in error_msg.lower():
                if 'assertThrows' in fixed_code:
                    issues.append("Error says 'nothing was thrown' but fix still uses assertThrows - use assertDoesNotThrow instead")
            elif 'unexpected exception type' in error_msg.lower():
                # The method throws a DIFFERENT exception than expected
                # Check which exception the error says was thrown
                actual_exception_match = re.search(r'but was:\s*<([^>]+)>', error_msg)
                if actual_exception_match:
                    actual_exception = actual_exception_match.group(1).split('.')[-1]  # Get just class name
                    # Check if the fix uses the correct exception type
                    if actual_exception not in fixed_code:
                        issues.append(f"Test throws {actual_exception} but fix doesn't use assertThrows({actual_exception}.class, ...)")
            elif 'assertThrows' not in fixed_code:
                issues.append("Test name contains 'Throws' but doesn't use assertThrows")

        # 3. Check for null params in methods when test doesn't test null handling
        if 'null' not in test_name_lower:
            # Find method calls with null parameters
            method_calls = re.findall(r'(\w+)\s*\.\s*(\w+)\s*\(\s*null\s*\)', fixed_code)
            for obj, method in method_calls:
                # Skip if it's a setter (setX(null) might be intentional)
                if not method.startswith('set'):
                    issues.append(f"Passing null to {obj}.{method}() but test doesn't test null handling")

        # 4. Check for incomplete date/time initialization
        if 'birthdate' in test_name_lower or 'date' in test_name_lower:
            if 'LocalDate' in fixed_code or 'Date' in fixed_code:
                # Ensure date is actually set to a value, not left as null
                if '.setBirthDate(null)' in fixed_code:
                    issues.append("setBirthDate(null) but test likely needs a real date value")

        if issues:
            return False, "; ".join(issues)
        return True, ""

    # =========================================================================
    # SELF-REFLECTION
    # =========================================================================

    def _record_fix_attempt(
        self,
        test_key: str,
        error_category: ErrorCategory,
        original_code: str,
        attempted_fix: str,
        result: str,
        error_after_fix: str | None
    ) -> None:
        """Record a fix attempt for future self-reflection."""
        if test_key not in self._fix_history:
            self._fix_history[test_key] = []

        iteration = len(self._fix_history[test_key]) + 1
        self._fix_history[test_key].append(FixAttempt(
            iteration=iteration,
            error_category=error_category,
            original_code=original_code,
            attempted_fix=attempted_fix,
            result=result,
            error_after_fix=error_after_fix
        ))

    def _build_reflection_section(self, test_key: str) -> str:
        """
        Build a structured reflection section from previous fix attempts.

        Includes:
        1. What was tried (summarized)
        2. Why it failed (specific error)
        3. Explicit anti-patterns to avoid
        4. Pattern-based warnings (e.g., null issues, repeated attempts)
        """
        if test_key not in self._fix_history or not self._fix_history[test_key]:
            return ""

        attempts = self._fix_history[test_key][-3:]  # Last 3 attempts only
        if not attempts:
            return ""

        reflection = "\n\n" + "="*50 + "\n"
        reflection += "PREVIOUS FIX ATTEMPTS - LEARN FROM THESE FAILURES:\n"
        reflection += "="*50 + "\n"

        for attempt in attempts:
            # Summarize what was tried
            fix_summary = attempt.attempted_fix[:100].replace('\n', ' ')
            error_summary = (attempt.error_after_fix or 'N/A')[:80]

            reflection += f"""
Attempt {attempt.iteration}:
  Tried: {fix_summary}...
  Result: {attempt.result}
  Error: {error_summary}
  DO NOT: Repeat this same approach!
"""

        # Add pattern-based warnings
        warnings = []

        # Check for null-related issues in previous attempts
        null_issues = any(
            'null' in (a.error_after_fix or '').lower() or
            'null' in (a.attempted_fix or '').lower()
            for a in attempts
        )
        if null_issues:
            warnings.append("Previous attempts had null-related issues. Ensure ALL required fields are set to valid non-null values (use LocalDate.now() for dates, new PetType() for types).")

        # Check for repeated similar fixes
        if len(attempts) >= 2:
            last_two = [a.attempted_fix[:50] for a in attempts[-2:]]
            if last_two[0] == last_two[1]:
                warnings.append("You're repeating the same fix. Try a COMPLETELY DIFFERENT approach.")

        # Check for assertion-related failures
        assertion_issues = any(
            'assert' in (a.error_after_fix or '').lower()
            for a in attempts
        )
        if assertion_issues:
            warnings.append("Assertion failures persist. Double-check: Is the assertion type correct for what you're testing? (assertFalse for 'no errors', assertTrue for 'has errors')")

        # Check for validation errors continuing
        validation_issues = any(
            'hasErrors' in (a.error_after_fix or '') or 'hasFieldErrors' in (a.error_after_fix or '')
            for a in attempts
        )
        if validation_issues:
            warnings.append("Validation is still failing. Make sure ALL required fields are properly set BEFORE calling validate().")

        if warnings:
            reflection += "\n⚠️ CRITICAL WARNINGS:\n"
            for i, warning in enumerate(warnings, 1):
                reflection += f"  {i}. {warning}\n"

        reflection += "\n" + "-"*50 + "\n"
        return reflection

    def _get_fix_prompt(
        self,
        failure: JavaTestFailure,
        original_test: str,
        error_category: ErrorCategory,
        test_key: str
    ) -> str:
        """
        Generate a strategy-specific prompt with API context and self-reflection.
        """
        # Get dynamic API context
        api_context = self._build_api_context_for_test(failure.test_class)

        # Get strategy-specific instructions
        strategy = self._get_strategy_instructions(error_category)

        # Get reflection from previous attempts
        reflection = self._build_reflection_section(test_key)

        return f"""You are fixing a failing JUnit 5 test.

DISCOVERED API SIGNATURES (use ONLY these - extracted from actual source code):
{api_context}
{reflection}
ERROR CATEGORY: {error_category.value}
ERROR: {failure.error_type} - {failure.error_message}

STACK TRACE:
{failure.stack_trace[:500]}

{strategy}

ORIGINAL TEST CODE:
```java
{original_test}
```

CRITICAL RULES:
1. Use null (NOT None), true/false (NOT True/False)
2. assertEquals(expected, actual) - expected FIRST
3. Check DISCOVERED API SIGNATURES before using any constructor or method
4. Only use classes that are already imported in the test file
5. Keep it simple - avoid complex date/time operations

Return ONLY the corrected @Test method (must be complete and compilable):
```java
@Test
"""

    # =========================================================================
    # CODE SANITIZATION
    # =========================================================================

    def _sanitize_java_code(self, code: str, available_imports: set[str]) -> str:
        """
        Sanitize LLM-generated Java code to fix common issues.

        This is a general-purpose sanitizer that:
        1. Fixes Python syntax leaking into Java
        2. Removes usage of classes not available in imports
        3. Balances braces
        """
        # Fix Python -> Java syntax
        code = re.sub(r'\bNone\b', 'null', code)
        code = re.sub(r'\bTrue\b', 'true', code)
        code = re.sub(r'\bFalse\b', 'false', code)

        # Handle unimported classes by using fully qualified names
        # This is better than replacing with null which would cause semantic issues
        class_fqn = {
            'LocalDate': 'java.time.LocalDate',
            'LocalDateTime': 'java.time.LocalDateTime',
            'Instant': 'java.time.Instant',
            'Date': 'java.util.Date',
            'Calendar': 'java.util.Calendar',
        }

        for cls, fqn in class_fqn.items():
            if cls in code and cls not in available_imports:
                # Replace class name with fully qualified name
                # But only for static method calls like LocalDate.now()
                code = re.sub(rf'\b{cls}\.', f'{fqn}.', code)
                # Also handle constructor calls like new LocalDate(...)
                code = re.sub(rf'new\s+{cls}\(', f'new {fqn}(', code)

        return code

    def _extract_imports_from_file(self, content: str) -> set[str]:
        """Extract imported class names from Java file content."""
        imports = set()
        import_pattern = re.compile(r'import\s+([\w.]+);')
        for match in import_pattern.finditer(content):
            full_import = match.group(1)
            # Get the class name (last part)
            class_name = full_import.split('.')[-1]
            if class_name != '*':
                imports.add(class_name)
        return imports

    # =========================================================================
    # LEGACY METHOD (kept for backward compatibility, now uses dynamic discovery)
    # =========================================================================

    def _get_source_context(self, test_class: str) -> str:
        """
        Get source code context - now uses dynamic API discovery.

        This is a compatibility wrapper that uses the new dynamic approach.
        """
        return self._build_api_context_for_test(test_class)

    def _verify_compilation(self) -> tuple[bool, str]:
        """Verify tests compile after fix."""
        # First apply Spring formatting
        subprocess.run(
            ["./mvnw", "spring-javaformat:apply", "-q"],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Then try to compile
        result = subprocess.run(
            ["./mvnw", "test-compile", "-q"],
            cwd=self.project_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0, result.stdout + result.stderr

    def _fix_compilation_errors(self, error_output: str) -> int:
        """Parse compilation errors and fix them using LLM."""
        # Parse error messages to find file and line info
        error_pattern = re.compile(
            r'/([^\s:]+\.java):\[(\d+),\d+\]\s+(.+?)(?=\n|$)',
            re.MULTILINE
        )

        fixes_applied = 0
        files_to_fix = {}

        for match in error_pattern.finditer(error_output):
            file_path = match.group(1)
            line_num = int(match.group(2))
            error_msg = match.group(3)

            # Find the full file path
            for java_file in self.project_path.rglob("*.java"):
                if java_file.name in file_path or str(java_file).endswith(file_path):
                    if java_file not in files_to_fix:
                        files_to_fix[java_file] = []
                    files_to_fix[java_file].append({
                        'line': line_num,
                        'error': error_msg
                    })
                    break

        # Fix each file
        for java_file, errors in files_to_fix.items():
            self._log(f"  Fixing {java_file.name} ({len(errors)} errors)")

            content = java_file.read_text()
            original_content = content

            # Get DYNAMIC API context (no hardcoding!)
            test_class_name = java_file.stem  # e.g., OwnerAddpetTest
            api_context = self._build_api_context_for_test(test_class_name)

            # Ask LLM to fix the entire file
            error_summary = "\n".join([f"Line {e['line']}: {e['error']}" for e in errors])

            prompt = f"""Fix the compilation errors in this Java test file.

DISCOVERED API SIGNATURES (use ONLY these - extracted from actual source code):
{api_context}

CURRENT TEST FILE WITH ERRORS:
```java
{content}
```

COMPILATION ERRORS:
{error_summary}

CRITICAL RULES:
1. Check DISCOVERED API SIGNATURES above for correct constructor and method signatures
2. If a class only has a no-arg constructor, do NOT pass arguments to it
3. Use setters after construction: Foo foo = new Foo(); foo.setBar(value);
4. Match method signatures exactly - check parameter types and counts
5. Use null (not None), true/false (not True/False)

Return the COMPLETE fixed Java file:
```java
"""

            try:
                fixed_content = self._call_llm_for_fix(prompt)

                if fixed_content and len(fixed_content) > 100:
                    # Clean up - remove markdown artifacts
                    fixed_content = fixed_content.strip()

                    # Handle various markdown formats
                    if "```java" in fixed_content:
                        parts = fixed_content.split("```java")
                        if len(parts) > 1:
                            fixed_content = parts[1].split("```")[0]
                    elif fixed_content.startswith("```"):
                        fixed_content = fixed_content[3:]
                        if fixed_content.startswith("java\n"):
                            fixed_content = fixed_content[5:]
                        fixed_content = fixed_content.split("```")[0]

                    fixed_content = fixed_content.strip()

                    # Verify the fix has the package declaration
                    if "package " in content and "package " not in fixed_content:
                        self._log(f"    Fix missing package declaration, skipping")
                        continue

                    java_file.write_text(fixed_content)

                    # Verify it compiles now
                    compiles, err = self._verify_compilation()
                    if compiles:
                        fixes_applied += 1
                        self._log(f"    ✓ Fixed")
                    else:
                        # Rollback
                        java_file.write_text(original_content)
                        self._log(f"    ✗ Fix didn't compile, rolled back")
                        # Show what error remains
                        if "error" in err.lower():
                            self._log(f"    Remaining error: {err[-300:]}")
                else:
                    self._log(f"    LLM returned empty or too short response")

            except Exception as e:
                self._log(f"    Error fixing: {e}")
                import traceback
                self._log(traceback.format_exc())

        return fixes_applied

    def _get_source_context_from_file(self, test_file: Path) -> str:
        """Get source code context based on test file name."""
        # Extract class name from test file name
        # e.g., OwnerAddpetTest.java -> Owner
        file_name = test_file.stem  # OwnerAddpetTest
        for pattern in ["Addpet", "Getpet", "Addvisit", "Addspecialty", "Test"]:
            if pattern in file_name:
                source_class = file_name.replace(pattern, "").replace("Test", "")
                break
        else:
            source_class = file_name.replace("Test", "")

        return self._get_source_context(f"org.test.{source_class}")

    def _fix_failure(self, failure: JavaTestFailure) -> bool:
        """
        Attempt to fix a single test failure using LLM with:
        - AST-based test method extraction (replaces fragile regex)
        - Error categorization for strategy-specific prompts
        - Dynamic API context from discovered classes
        - Self-reflection from previous fix attempts
        - Retry logic for unbalanced braces
        """
        test_file = failure.test_file
        test_key = f"{failure.test_class}.{failure.test_name}"

        if not test_file.exists():
            self._log(f"    Could not find test file: {test_file}")
            return False

        # Read current test file (save for rollback)
        original_content = test_file.read_text()

        # Use AST parser to find test method (replaces fragile regex)
        test_methods = self._parse_test_file(test_file)
        method_context = test_methods.get(failure.test_name)

        if method_context and method_context.source_code:
            # AST found the method - use its source code
            original_test = method_context.source_code
            # Add @Test annotation if not in source (AST might not capture it)
            if "@Test" not in original_test:
                # Find the annotation by looking at lines before the method
                lines = original_content.split('\n')
                method_start = method_context.location.start_line - 1
                # Look back up to 3 lines for @Test annotation
                annotation_lines = []
                for i in range(max(0, method_start - 3), method_start):
                    line = lines[i].strip()
                    if line.startswith("@"):
                        annotation_lines.append(lines[i])
                if annotation_lines:
                    original_test = '\n'.join(annotation_lines) + '\n' + original_test
        else:
            # Fallback to regex if AST fails
            self._log(f"    AST didn't find method, falling back to regex")
            test_pattern = re.compile(
                rf'(@Test[^\n]*\n\s*(?:@DisplayName\([^)]+\)\s*\n\s*)?(?:void\s+)?{re.escape(failure.test_name)}\s*\([^)]*\)\s*\{{.*?\n\s*\}})',
                re.DOTALL
            )

            match = test_pattern.search(original_content)
            if not match:
                # Try simpler pattern
                test_pattern = re.compile(
                    rf'(@Test.*?void\s+{re.escape(failure.test_name)}\s*\([^)]*\)\s*\{{[^}}]*\}})',
                    re.DOTALL
                )
                match = test_pattern.search(original_content)

            if not match:
                self._log(f"    Could not find test method: {failure.test_name}")
                return False

            original_test = match.group(1)

        # Step 1: Categorize the error for strategy-specific handling
        error_category = self._categorize_error(failure)
        self._log(f"    Category: {error_category.value}")

        # Step 2: Generate strategy-specific prompt with API context and reflection
        prompt = self._get_fix_prompt(failure, original_test, error_category, test_key)

        try:
            # Use retry-enabled LLM call to handle unbalanced braces
            fixed_code = self._call_llm_with_retry(prompt, failure.test_name)

            if not fixed_code:
                self._record_fix_attempt(
                    test_key, error_category, original_test,
                    "LLM returned empty", "no_response", None
                )
                return False

            # Clean up fixed code
            fixed_code = fixed_code.strip()

            # Sanitize the code - fix common LLM issues
            available_imports = self._extract_imports_from_file(original_content)
            fixed_code = self._sanitize_java_code(fixed_code, available_imports)

            # SEMANTIC VERIFICATION: Check for common LLM mistakes before applying
            is_valid, semantic_issue = self._verify_fix_semantics(
                failure.test_name,
                fixed_code,
                failure.error_message
            )

            if not is_valid:
                self._log(f"    Semantic issue detected: {semantic_issue}")

                # Build specific fix instructions based on the issue
                fix_instructions = []
                if 'null' in semantic_issue.lower() and 'birthdate' in semantic_issue.lower():
                    fix_instructions.append("Replace pet.setBirthDate(null) with pet.setBirthDate(java.time.LocalDate.now())")
                if 'null' in semantic_issue.lower() and 'value' in semantic_issue.lower():
                    fix_instructions.append("Initialize the variable to a real value, not null")
                if 'assertThrows' in semantic_issue.lower() and 'nothing was thrown' in semantic_issue.lower():
                    fix_instructions.append("Replace assertThrows with assertDoesNotThrow(() -> validator.validate(null, null))")
                if 'success case' in semantic_issue.lower():
                    fix_instructions.append("Remove assertThrows and use assertFalse(errors.hasErrors()) for success validation")
                if 'NullPointerException' in semantic_issue:
                    fix_instructions.append("Change assertThrows(IllegalArgumentException.class, ...) to assertThrows(NullPointerException.class, ...)")
                if 'unexpected exception type' in failure.error_message.lower():
                    actual_match = re.search(r'but was:\s*<([^>]+)>', failure.error_message)
                    if actual_match:
                        actual = actual_match.group(1).split('.')[-1]
                        fix_instructions.append(f"The method throws {actual}, so use assertThrows({actual}.class, ...)")

                specific_fixes = "\n".join(f"- {f}" for f in fix_instructions) if fix_instructions else "- Fix the issue described above"

                # Retry with specific feedback about the semantic issue
                retry_prompt = f"""Your fix has a semantic issue that will cause the test to fail:

ISSUE: {semantic_issue}

SPECIFIC FIX REQUIRED:
{specific_fixes}

Your previous fix:
```java
{fixed_code}
```

Return ONLY the corrected @Test method with the issue fixed:
```java
@Test"""

                retry_code = self._call_llm_for_fix(retry_prompt)
                if retry_code:
                    retry_code = retry_code.strip()
                    retry_code = self._sanitize_java_code(retry_code, available_imports)

                    # Verify the retry fixes the semantic issue
                    is_valid_retry, _ = self._verify_fix_semantics(
                        failure.test_name,
                        retry_code,
                        failure.error_message
                    )

                    if is_valid_retry:
                        self._log(f"    Semantic issue fixed on retry")
                        fixed_code = retry_code
                    else:
                        self._log(f"    Retry still has semantic issues, proceeding anyway")
                        # Use the retry code anyway - it might be better

            # Ensure it starts with @Test
            if not fixed_code.startswith("@Test"):
                fixed_code = "@Test\n\t" + fixed_code

            # Preserve original indentation
            original_indent = ""
            for char in original_test:
                if char in ' \t':
                    original_indent += char
                else:
                    break

            # Apply indentation
            fixed_lines = fixed_code.split('\n')
            reindented_lines = []
            for line in fixed_lines:
                if line.strip():
                    reindented_lines.append(original_indent + line.lstrip())
                else:
                    reindented_lines.append('')

            fixed_code = '\n'.join(reindented_lines)

            # Replace the original test with the fixed one
            new_content = original_content.replace(original_test, fixed_code)

            # Final validation - check for balanced braces (after retry already attempted)
            if new_content.count('{') != new_content.count('}'):
                self._log(f"    Unbalanced braces in fix after retry, skipping")
                self._record_fix_attempt(
                    test_key, error_category, original_test,
                    fixed_code, "compile_error", "Unbalanced braces after retry"
                )
                return False

            # Write the fix
            test_file.write_text(new_content)

            # Invalidate cache for this file since we modified it
            if test_file in self._test_method_cache:
                del self._test_method_cache[test_file]

            # Verify compilation
            compiles, error_msg = self._verify_compilation()
            if not compiles:
                self._log(f"    Fix doesn't compile, rolling back")
                # Show more of the error
                if error_msg:
                    self._log(f"    Compile error: {error_msg[:300]}")
                else:
                    # Try to get stdout too
                    result = subprocess.run(
                        ["./mvnw", "test-compile"],
                        cwd=self.project_path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    error_msg = (result.stdout + result.stderr)[-300:]
                    self._log(f"    Compile output: {error_msg}")

                # Record failed attempt for self-reflection
                self._record_fix_attempt(
                    test_key, error_category, original_test,
                    fixed_code, "compile_error", error_msg
                )

                # Rollback
                test_file.write_text(original_content)
                # Re-invalidate cache after rollback
                if test_file in self._test_method_cache:
                    del self._test_method_cache[test_file]
                return False

            # Success! Record it
            self._record_fix_attempt(
                test_key, error_category, original_test,
                fixed_code, "compiled", None
            )
            return True

        except Exception as e:
            self._log(f"    Error during fix: {e}")
            import traceback
            self._log(f"    {traceback.format_exc()}")

            # Record failed attempt
            self._record_fix_attempt(
                test_key, error_category, original_test,
                "Exception occurred", "error", str(e)
            )

            # Rollback on any error
            test_file.write_text(original_content)
            # Re-invalidate cache after rollback
            if test_file in self._test_method_cache:
                del self._test_method_cache[test_file]
            return False

    def _call_llm_for_fix(self, prompt: str) -> Optional[str]:
        """Call LLM to get a fix for a failing test."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            self._log(f"    LLM API error: {response.status_code}")
            return None

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Extract code from response
        if "```java" in content:
            code = content.split("```java")[1].split("```")[0]
        elif "```" in content:
            code = content.split("```")[1].split("```")[0]
        else:
            code = content

        return code.strip()

    def _call_llm_with_retry(
        self,
        prompt: str,
        method_name: str,
        max_retries: int = 1
    ) -> Optional[str]:
        """
        Call LLM with retry for unbalanced braces.

        If the LLM generates incomplete code with unbalanced braces,
        retries with an explicit prompt emphasizing complete code.
        """
        fixed_code = self._call_llm_for_fix(prompt)

        if not fixed_code:
            return None

        # Check brace balance
        open_braces = fixed_code.count('{')
        close_braces = fixed_code.count('}')

        if open_braces != close_braces and max_retries > 0:
            self._log(f"    Unbalanced braces ({open_braces} open, {close_braces} close), retrying...")

            # Retry with explicit prompt
            retry_prompt = f"""Your previous fix had unbalanced braces ({open_braces} '{{' vs {close_braces} '}}').

Please provide a COMPLETE test method with ALL braces properly closed.

Requirements:
1. Start with @Test annotation
2. Method signature: void {method_name}()
3. Have EXACTLY matching {{ and }} braces
4. Be complete and compilable Java code

Previous incomplete code:
```java
{fixed_code}
```

Return ONLY the COMPLETE corrected @Test method with all braces closed:
```java
@Test"""

            retry_code = self._call_llm_for_fix(retry_prompt)
            if retry_code:
                # Check balance again
                retry_open = retry_code.count('{')
                retry_close = retry_code.count('}')
                if retry_open == retry_close:
                    return retry_code
                else:
                    self._log(f"    Retry still unbalanced ({retry_open} vs {retry_close})")
                    # Return original if retry also fails
                    return fixed_code

        return fixed_code

    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)


def run_java_react_loop(
    project_path: Path,
    api_key: str,
    test_classes: list[str] = None,
    max_iterations: int = 3,
    verbose: bool = True
) -> bool:
    """
    Convenience function to run the Java ReAct loop.

    Args:
        project_path: Path to Maven project root
        api_key: OpenRouter API key
        test_classes: List of test class names to run (None = all)
        max_iterations: Maximum number of fix iterations
        verbose: Print progress

    Returns:
        True if all tests pass, False otherwise
    """
    loop = JavaReActLoop(
        project_path=project_path,
        api_key=api_key,
        test_classes=test_classes,
        max_iterations=max_iterations,
        verbose=verbose
    )

    results = loop.run()

    if results:
        return results[-1].success
    return False
