"""
SpecAgent - Test generation agent using DSL + Compiler + Feedback Loop.

This agent:
1. Extracts dependencies from source class
2. Asks LLM to generate TestSpec JSON (not Java code)
3. Compiles TestSpec to Java deterministically
4. Runs Java compiler and tests as loss function
5. Feeds errors back to LLM for improvement
6. Saves successful specs for future learning

Key insight: By constraining LLM output to structured JSON, we eliminate
syntax hallucinations. The compiler handles all Java syntax generation.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import requests

from .models import DslTestSpec, DslTestCaseSpec, DslMockSpec, MockSetup, Assertion, MethodCall
from .models import get_testspec_prompt_schema
from .compiler import SpecCompiler
from .maven_utils import run_compile, run_tests
from .java_compiler import FastJavaCompiler, FastTestRunner

if TYPE_CHECKING:
    from .models import ClassContext
    from .context.project_context import ProjectContext


@dataclass
class SpecAgentResult:
    """Result from SpecAgent run."""
    success: bool
    spec: DslTestSpec | None = None
    java_code: str = ""
    tests_passed: int = 0
    tests_total: int = 0
    iterations: int = 0
    error: str | None = None


@dataclass
class SpecAgentConfig:
    """Configuration for SpecAgent."""
    project_path: Path
    api_key: str
    model: str = "x-ai/grok-code-fast-1"
    max_iterations: int = 3
    verbose: bool = True


class SpecAgent:
    """
    Agent that generates tests using DSL + Compiler + Feedback Loop.

    The agent uses the compiler as a loss function:
    - If Java doesn't compile → feed error back to LLM
    - If tests fail → feed failure info back to LLM
    - If all tests pass → save spec to learned library
    """

    def __init__(self, config: SpecAgentConfig, project_context: "ProjectContext | None" = None):
        self.config = config
        self.compiler = SpecCompiler()
        self.last_error: str | None = None
        self.project_context = project_context
        self._current_test_file: Path | None = None

        # Fast Java compiler for iteration speed
        self.fast_compiler = FastJavaCompiler(config.project_path, verbose=config.verbose)
        # Ensure project is compiled once at startup
        self.fast_compiler.ensure_project_compiled()

        # Fast test runner using JUnit ConsoleLauncher (skips Maven overhead)
        classpath = self.fast_compiler._get_classpath()
        self.fast_test_runner = FastTestRunner(config.project_path, classpath, verbose=config.verbose) if classpath else None

    def run(self, target_class: "ClassContext") -> SpecAgentResult:
        """
        Generate tests for a class using DSL approach.

        Args:
            target_class: The class to generate tests for

        Returns:
            SpecAgentResult with success status and generated code
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"SpecAgent: {target_class.name}")
            print(f"{'='*60}")

        # 1. Extract dependencies from source
        deps = self._extract_dependencies(target_class)
        if self.config.verbose:
            print(f"  Dependencies: {[f'{d.type} {d.name}' for d in deps]}")

        # 2. Extract package name
        package_name = self._extract_package(target_class)

        # 3. Get methods to test
        methods_info = self._get_methods_info(target_class)

        # 4. Read source code for context
        source_code = ""
        if target_class.location and target_class.location.file_path:
            try:
                source_code = target_class.location.file_path.read_text()
            except:
                pass

        # 5. Extract entity APIs used by this class
        entity_apis = self._extract_entity_apis(source_code)
        if self.config.verbose and entity_apis:
            print(f"  Entity APIs: {len(entity_apis)} entities found")

        self.last_error = None

        for iteration in range(1, self.config.max_iterations + 1):
            if self.config.verbose:
                print(f"\n[Iteration {iteration}/{self.config.max_iterations}]")

            # 6. Generate TestSpec via LLM
            if self.config.verbose:
                print("  Generating TestSpec...")

            spec = self._generate_spec(
                class_name=target_class.name,
                package_name=package_name,
                dependencies=deps,
                methods_info=methods_info,
                source_code=source_code,
                entity_apis=entity_apis,
                feedback=self.last_error
            )

            if spec is None:
                if self.config.verbose:
                    print(f"  Failed to generate valid spec")
                continue

            if self.config.verbose:
                print(f"  Generated {len(spec.test_cases)} test cases")

            # 6. Compile to Java (deterministic)
            java_code = self.compiler.compile(spec)

            # 7. Write to file
            test_file = self._get_test_file_path(target_class)
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(java_code)
            self._current_test_file = test_file  # Store for fast compilation

            if self.config.verbose:
                print(f"  Written to: {test_file}")

            # 8. Compile Java (loss function #1) - now uses fast javac
            compile_result = self._java_compile()
            if not compile_result["success"]:
                self.last_error = f"Compilation error:\n{compile_result['error']}"
                if self.config.verbose:
                    print(f"  Compile: FAILED")
                    print(f"    {compile_result['error'][:200]}...")
                continue

            if self.config.verbose:
                print(f"  Compile: OK")

            # 9. Run tests (loss function #2)
            test_result = self._run_tests(target_class.name)

            if self.config.verbose:
                print(f"  Tests: {test_result['passed']}/{test_result['total']}")

            if test_result["passed"] == test_result["total"] and test_result["total"] > 0:
                # Success!
                return SpecAgentResult(
                    success=True,
                    spec=spec,
                    java_code=java_code,
                    tests_passed=test_result["passed"],
                    tests_total=test_result["total"],
                    iterations=iteration
                )
            else:
                # Tests failed - feed back to LLM
                self.last_error = f"Test failures:\n{test_result['failures']}"
                if self.config.verbose:
                    print(f"  Failures: {test_result['failures'][:200]}...")

        # Exhausted iterations
        return SpecAgentResult(
            success=False,
            java_code=java_code if 'java_code' in dir() else "",
            tests_passed=test_result.get("passed", 0) if 'test_result' in dir() else 0,
            tests_total=test_result.get("total", 0) if 'test_result' in dir() else 0,
            iterations=self.config.max_iterations,
            error=self.last_error
        )

    def _extract_dependencies(self, cls: "ClassContext") -> list[DslMockSpec]:
        """Extract dependencies from constructor params and @Autowired fields."""
        deps = []

        # From constructor parameters
        if cls.constructors:
            for constructor in cls.constructors:
                for param in constructor.parameters:
                    if param.type_hint:
                        deps.append(DslMockSpec(type=param.type_hint, name=param.name))

        # From @Autowired fields
        if cls.location and cls.location.file_path:
            try:
                source_code = cls.location.file_path.read_text()
                autowired_pattern = r'@Autowired\s+(?:private\s+)?(\w+)\s+(\w+)\s*;'
                for match in re.finditer(autowired_pattern, source_code):
                    field_type, field_name = match.groups()
                    # Check not already added
                    if not any(d.name == field_name for d in deps):
                        deps.append(DslMockSpec(type=field_type, name=field_name))
            except:
                pass

        return deps

    def _extract_package(self, cls: "ClassContext") -> str:
        """Extract package name from class file path."""
        if cls.location and cls.location.file_path:
            path_str = str(cls.location.file_path)
            if "src/main/java/" in path_str:
                pkg_path = path_str.split("src/main/java/")[1]
                pkg_path = pkg_path.rsplit("/", 1)[0]  # Remove filename
                return pkg_path.replace("/", ".")
        return "generated.tests"

    def _get_methods_info(self, cls: "ClassContext") -> str:
        """Get methods info for prompt."""
        methods = []
        for method in cls.methods:
            params = ", ".join(
                f"{p.type_hint or 'Object'} {p.name}"
                for p in method.parameters
            )
            return_type = method.return_type or "void"
            methods.append(f"- {return_type} {method.name}({params})")
        return "\n".join(methods)

    def _get_test_file_path(self, cls: "ClassContext") -> Path:
        """Get the test file path for a class."""
        if cls.location and cls.location.file_path:
            source_path = str(cls.location.file_path)
            # Convert src/main/java to src/test/java
            test_path = source_path.replace("src/main/java", "src/test/java")
            # Change ClassName.java to ClassNameTest.java
            test_path = test_path.replace(".java", "Test.java")
            return Path(test_path)
        return self.config.project_path / "src/test/java" / f"{cls.name}Test.java"

    def _extract_package_from_file(self, file_path: Path) -> str | None:
        """Extract package name from a Java file."""
        if not file_path or not file_path.exists():
            return None
        try:
            content = file_path.read_text()
            match = re.search(r'package\s+([\w.]+)\s*;', content)
            return match.group(1) if match else None
        except Exception:
            return None

    def _generate_spec(
        self,
        class_name: str,
        package_name: str,
        dependencies: list[DslMockSpec],
        methods_info: str,
        source_code: str,
        entity_apis: str,
        feedback: str | None
    ) -> DslTestSpec | None:
        """Ask LLM to generate TestSpec JSON."""

        deps_str = "\n".join(f"- {d.type} {d.name}" for d in dependencies)

        prompt = f'''You are generating a TestSpec (NOT Java code) for {class_name}.

SOURCE CLASS:
```java
{source_code[:3000]}
```

DEPENDENCIES (will be mocked with @Mock):
{deps_str}

METHODS TO TEST:
{methods_info}
'''

        # Add entity APIs if available
        if entity_apis:
            prompt += f'''
ENTITY APIS (use ONLY these methods - do NOT invent method names):
{entity_apis}

'''

        prompt += f'''OUTPUT FORMAT - Generate JSON with test_cases array:
{get_testspec_prompt_schema()}

RULES:
1. Generate 2 test cases per method (happy path + edge case)
2. Use "controller" as the object name in action
3. For void service methods, use verify assertions
4. For return values, use assertEquals with the "result" variable (from action.returns_to)
5. Keep arrange statements as valid Java code snippets
6. CRITICAL: Only use entity setter/getter methods listed in ENTITY APIS above. Do NOT invent methods like setName() - use the actual methods like setPname().
7. CRITICAL: Every variable used in verify() or assertions MUST be declared in arrange. If a test verifies a mock like dataBinder, you MUST include it in arrange: "WebDataBinder dataBinder = mock(WebDataBinder.class);"
8. CRITICAL: assertEquals(expected, actual) - the 'actual' MUST be "result" (literally the string "result", which comes from action.returns_to). NEVER use arrange variables in assertEquals actual. Example: assertEquals("redirect:/owners", result) NOT assertEquals("redirect:/owners", bindingResult).
9. CRITICAL: For assertThrows edge cases, the assertions array should contain ONLY the assertThrows, and the action should be a dummy that doesn't throw. OR use arrange=[] and let assertThrows be the only thing.
10. CRITICAL: For edge cases testing null input, use arrange to set up a null variable if needed, or just pass null directly in action args.
11. CRITICAL: For Pet/Owner relationships, add Pet to Owner FIRST, then set IDs (addPet only works for new pets with null id):
    Pet pet = new Pet();
    owner.addPet(pet);  // Add first when pet.isNew() is true
    pet.setId(1);       // Then set ID
12. CRITICAL: assertThrows syntax: {{"type": "throws", "exception": "IllegalArgumentException", "actual": "() -> controller.method(args)"}}
13. CRITICAL: When testing repository.save(entity), if the method returns a redirect with entity.getId(), set the ID BEFORE calling:
    Owner owner = new Owner();
    owner.setId(1);  // Set ID before save() so redirect:/owners/1 works
14. CRITICAL: For pagination, use page=1 (not page=0). Spring uses 1-based pages. Use different variable names: "int pageNum = 1" and "Page<Owner> ownersPage = ...".
15. CRITICAL: Read EXACT return strings from source. If source says return "redirect:/owners/{{ownerId}}", expect exactly that string.
16. CRITICAL: For void methods (return type void), do NOT set action.returns_to. Only non-void methods should have returns_to.

'''
        if feedback:
            prompt += f'''
PREVIOUS ERROR - Fix this issue:
{feedback[:500]}

'''

        prompt += "Output ONLY valid JSON, no explanation, no markdown code blocks."

        # Call LLM
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON using robust extraction with fallbacks
            data = self._extract_json(content)
            if data is None:
                if self.config.verbose:
                    print(f"  JSON extraction failed. Raw content: {content[:200]}...")
                return None

            # Build TestSpec
            test_cases = []
            for tc_data in data.get("test_cases", []):
                # Parse mocks
                mocks = []
                for m in tc_data.get("mocks", []):
                    mocks.append(MockSetup(
                        mock_name=m.get("mock_name", ""),
                        method=m.get("method", ""),
                        args=m.get("args", []),
                        returns=m.get("returns"),
                        throws=m.get("throws")
                    ))

                # Parse action
                action_data = tc_data.get("action", {})
                # Ensure args is a list of strings, not None
                action_args = action_data.get("args") or []
                action_args = [str(a) if a is not None else "" for a in action_args]
                action = MethodCall(
                    object=action_data.get("object", "controller"),
                    method=action_data.get("method", ""),
                    args=action_args,
                    returns_to=action_data.get("returns_to")
                )

                # Parse assertions
                assertions = []
                for a in tc_data.get("assertions", []):
                    # Ensure args is list of strings
                    assertion_args = a.get("args")
                    if assertion_args:
                        assertion_args = [str(arg) for arg in assertion_args]

                    # Handle verify assertions - need valid mock_name
                    mock_name = a.get("mock_name")
                    if a.get("type") == "verify":
                        # Fix common LLM errors
                        if mock_name in (None, "None", "null", ""):
                            # Try to infer from method or skip this assertion
                            continue  # Skip invalid verify assertions

                    assertions.append(Assertion(
                        type=a.get("type", "not_null"),
                        expected=a.get("expected"),
                        actual=a.get("actual"),
                        exception=a.get("exception"),
                        mock_name=mock_name,
                        method=a.get("method"),
                        args=assertion_args
                    ))

                test_cases.append(DslTestCaseSpec(
                    name=tc_data.get("name", "test"),
                    description=tc_data.get("description", ""),
                    mocks=mocks,
                    arrange=tc_data.get("arrange", []),
                    action=action,
                    assertions=assertions
                ))

            # Build additional imports from source
            additional_imports = self._extract_imports(source_code, package_name)

            return DslTestSpec(
                class_under_test=class_name,
                package_name=package_name,
                dependencies=dependencies,
                test_cases=test_cases,
                additional_imports=additional_imports
            )

        except json.JSONDecodeError as e:
            if self.config.verbose:
                print(f"  JSON parse error at pos {e.pos}: {e.msg}")
            return None
        except requests.RequestException as e:
            if self.config.verbose:
                print(f"  LLM API error: {e}")
            return None
        except Exception as e:
            if self.config.verbose:
                print(f"  Unexpected error: {type(e).__name__}: {e}")
            return None

    def _extract_json(self, content: str) -> dict | None:
        """Extract JSON from LLM response with multiple fallback strategies."""
        strategies = [
            # Strategy 1: ```json ... ```
            lambda c: c.split("```json")[1].split("```")[0] if "```json" in c else None,
            # Strategy 2: ``` ... ``` (any code block)
            lambda c: c.split("```")[1].split("```")[0] if c.count("```") >= 2 else None,
            # Strategy 3: Find { ... } directly (outermost braces)
            lambda c: c[c.find("{"):c.rfind("}")+1] if "{" in c and "}" in c else None,
            # Strategy 4: Raw content as-is
            lambda c: c.strip()
        ]

        for i, strategy in enumerate(strategies):
            try:
                extracted = strategy(content)
                if extracted:
                    # Try to repair before parsing
                    repaired = self._repair_json(extracted)
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return data
            except (json.JSONDecodeError, IndexError, ValueError):
                continue
        return None

    def _repair_json(self, text: str) -> str:
        """Fix common JSON errors from LLM responses."""
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        # Remove comments (// style)
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        # Fix single quotes to double quotes for keys
        # Be careful not to break strings - only do simple cases
        text = re.sub(r"'(\w+)':", r'"\1":', text)
        return text.strip()

    def _extract_imports(self, source_code: str, package_name: str) -> list[str]:
        """Extract imports from source code needed for tests."""
        imports = []

        for line in source_code.split("\n"):
            line = line.strip()
            if line.startswith("import "):
                # Skip Spring framework imports (test has its own)
                if line.startswith("import org.springframework"):
                    continue
                # Skip Jakarta annotations
                if line.startswith("import jakarta."):
                    continue
                # Skip common java imports that are rarely needed
                if line.startswith("import java.util."):
                    continue
                # Include java.text (ParseException), java.time, etc
                imports.append(line.rstrip(";"))

        return imports

    def _extract_entity_apis(self, source_code: str) -> str:
        """Extract entity and repository APIs from imports.

        Looks for entity/repository imports and reads their source files to extract:
        1. Available setters and getters (prevents hallucinating method names)
        2. Relationships (Owner has List<Pet>, Pet has List<Visit>)
        3. Repository method signatures (findByLastNameStartingWith, etc.)
        """
        entity_apis = []
        repo_apis = []

        # Find entity and repository imports
        entity_imports = []
        repo_imports = []
        for line in source_code.split("\n"):
            line = line.strip()
            if line.startswith("import "):
                fqn = line.replace("import ", "").rstrip(";")
                parts = fqn.split(".")
                name = parts[-1] if parts else ""

                # Repository interfaces
                if "Repository" in name:
                    repo_imports.append((name, fqn))
                # Entity classes
                elif any(pkg in line for pkg in [".entities.", ".owner.", ".vet.", ".model."]):
                    entity_imports.append((name, fqn))

        # Also check constructor params for Repository types (same package, no import)
        constructor_match = re.search(r'public\s+\w+\s*\(([^)]+)\)', source_code)
        if constructor_match:
            params = constructor_match.group(1)
            for param in params.split(','):
                param = param.strip()
                if 'Repository' in param:
                    # Extract type name (e.g., "OwnerRepository owners" -> "OwnerRepository")
                    parts = param.split()
                    if parts:
                        repo_type = parts[0]
                        # Find in same package
                        pkg_match = re.search(r'package\s+([\w.]+);', source_code)
                        if pkg_match:
                            pkg = pkg_match.group(1)
                            repo_imports.append((repo_type, f"{pkg}.{repo_type}"))

        # For each entity, extract methods and relationships
        for entity_name, full_path in entity_imports:
            file_path = self.config.project_path / "src/main/java" / full_path.replace(".", "/")
            file_path = Path(str(file_path) + ".java")

            if file_path.exists():
                try:
                    entity_source = file_path.read_text()
                    methods = []
                    relationships = []

                    # Extract setter methods
                    setter_pattern = r'public\s+void\s+(set\w+)\s*\(([^)]+)\)'
                    for match in re.finditer(setter_pattern, entity_source):
                        method_name = match.group(1)
                        params = match.group(2)
                        methods.append(f"  void {method_name}({params})")

                    # Extract getter methods
                    getter_pattern = r'public\s+([\w<>]+)\s+(get\w+)\s*\(\s*\)'
                    for match in re.finditer(getter_pattern, entity_source):
                        return_type = match.group(1)
                        method_name = match.group(2)
                        methods.append(f"  {return_type} {method_name}()")

                        # Detect relationships: List<X>, Set<X>
                        if "List<" in return_type or "Set<" in return_type:
                            related_type = re.search(r'<(\w+)>', return_type)
                            if related_type:
                                relationships.append(f"  HAS-MANY {related_type.group(1)} (via {method_name})")

                    # Extract add* methods (relationship helpers)
                    add_pattern = r'public\s+void\s+(add\w+)\s*\(([^)]+)\)'
                    for match in re.finditer(add_pattern, entity_source):
                        method_name = match.group(1)
                        params = match.group(2)
                        methods.append(f"  void {method_name}({params})")

                    # Extract other important methods (get by id/name)
                    get_by_pattern = r'public\s+(\w+)\s+(get\w+)\s*\((\w+\s+\w+)\)'
                    for match in re.finditer(get_by_pattern, entity_source):
                        return_type = match.group(1)
                        method_name = match.group(2)
                        params = match.group(3)
                        methods.append(f"  {return_type} {method_name}({params})")

                    # Build entity API string
                    api_str = f"{entity_name}:"
                    if relationships:
                        api_str += "\n  # Relationships:"
                        api_str += "\n" + "\n".join(relationships)
                    if methods:
                        api_str += "\n  # Methods:"
                        api_str += "\n" + "\n".join(methods)

                    if methods or relationships:
                        entity_apis.append(api_str)
                except:
                    pass

        # Extract repository methods
        for repo_name, full_path in repo_imports:
            file_path = self.config.project_path / "src/main/java" / full_path.replace(".", "/")
            file_path = Path(str(file_path) + ".java")

            if file_path.exists():
                try:
                    repo_source = file_path.read_text()
                    methods = []

                    # Extract method signatures from interface
                    method_pattern = r'(?:Page<\w+>|\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)'
                    for match in re.finditer(method_pattern, repo_source):
                        method_name = match.group(1)
                        params = match.group(2).strip()
                        # Get return type from before method name
                        line_start = repo_source.rfind('\n', 0, match.start()) + 1
                        line = repo_source[line_start:match.end()]
                        return_match = re.search(r'([\w<>,\s]+?)\s+' + method_name, line)
                        return_type = return_match.group(1).strip() if return_match else "Object"
                        methods.append(f"  {return_type} {method_name}({params})")

                    if methods:
                        repo_apis.append(f"{repo_name}:\n" + "\n".join(methods))
                except:
                    pass

        # Combine entity and repository APIs
        all_apis = entity_apis + repo_apis

        # Add mock setup hints
        if all_apis:
            hints = """
MOCK SETUP HINTS:
- For entities with relationships (HAS-MANY), create child objects and add them:
  Owner owner = new Owner();
  Pet pet = new Pet();
  pet.setId(1);  // IMPORTANT: Set ID before adding
  owner.addPet(pet);  // Or: owner.getPets().add(pet);
- When mocking repository.findById(id), return Optional.of(entity) with relationships set up
- For getPet(id) to work, the Pet must have matching ID set via setId()
- CRITICAL: Use exact repository method names shown above, do NOT invent similar names"""
            all_apis.append(hints)

        return "\n\n".join(all_apis)

    def _java_compile(self) -> dict:
        """
        Compile Java tests using fast javac (1-3s vs Maven's 20-40s).

        Uses FastJavaCompiler for single-file compilation when a specific
        test file is being worked on, falls back to Maven otherwise.
        """
        # Use fast javac if we have a specific test file
        if self._current_test_file and self._current_test_file.exists():
            success, error = self.fast_compiler.compile_test(self._current_test_file)
            if success:
                return {"success": True, "error": None}
            else:
                return {"success": False, "error": error}

        # Fallback to Maven for full project compile
        success, stdout, stderr = run_compile(self.config.project_path, timeout=120)
        if success:
            return {"success": True, "error": None}
        else:
            return {"success": False, "error": stdout + stderr}

    def _run_tests(self, class_name: str) -> dict:
        """
        Run tests for a specific class using JUnit ConsoleLauncher (7s vs Maven's 40-50s).

        Falls back to Maven if JUnit launcher isn't available.
        """
        # Get fully qualified class name from the test file
        package_name = self._extract_package_from_file(self._current_test_file) if self._current_test_file else None
        fq_class_name = f"{package_name}.{class_name}Test" if package_name else f"{class_name}Test"

        # Try fast JUnit runner first
        if self.fast_test_runner:
            success, output, passed, total = self.fast_test_runner.run_test(fq_class_name)

            if total > 0:  # JUnit ran successfully
                return {
                    "passed": passed,
                    "total": total,
                    "failures": output if passed < total else ""
                }

        # Fallback to Maven
        from .maven_utils import build_test_cmd
        import subprocess

        cmd = build_test_cmd(
            self.config.project_path,
            test_class=f"{class_name}Test",
            quiet=False,
            fast=True
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.project_path,
                capture_output=True,
                text=True,
                timeout=180
            )
            output = result.stdout + result.stderr
        except Exception as e:
            output = str(e)

        # Parse Maven/Surefire output
        matches = re.findall(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", output)
        match = matches[-1] if matches else None
        if match:
            total = int(match[0])
            failures = int(match[1])
            errors = int(match[2])
            passed = total - failures - errors
            return {
                "passed": passed,
                "total": total,
                "failures": output if (failures + errors) > 0 else ""
            }

        # Check for BUILD SUCCESS as fallback
        if "BUILD SUCCESS" in output:
            return {"passed": 1, "total": 1, "failures": ""}

        # If no match, assume failure
        return {"passed": 0, "total": 0, "failures": output}
