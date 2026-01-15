#!/usr/bin/env python3
"""
Unified Java Test Agent.

Orchestrates the existing pipeline components:
1. ASTParser - Java source parsing
2. SpecGenerator - LLM-based test spec generation
3. TemplateRenderer - Test code rendering
4. JavaReActLoop - Test fixing

All components share a unified ProjectContext built once at startup.
"""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from dataclasses import dataclass

from .context.ast_parser import ASTParser
from .context.project_context import ProjectContext
from .context.test_examples import TestExamplesFinder
from .generator.spec_generator import OpenRouterSpecGenerator
from .generator.direct_generator import DirectCodeGenerator
from .renderer.template_renderer import TemplateRenderer
from .react_loop_java import JavaReActLoop
from .models import Language
from .validator import SpecValidator
from .code_postprocessor import JavaCodePostProcessor
from .maven_utils import run_compile
from .few_shot import FewShotExampleSelector
from .few_shot.auto_learner import FewShotAutoLearner
from .spec_agent import SpecAgent, SpecAgentConfig


@dataclass
class JavaAgentConfig:
    project_path: Path
    api_key: str
    model: str = "x-ai/grok-code-fast-1"
    max_iterations: int = 5
    verbose: bool = True
    max_methods: int = 0  # 0 = no limit, >0 = limit methods per class for faster evals
    generator_type: str = "spec"  # "spec", "direct", or "dsl" - architecture comparison
    self_verify: bool = True  # For direct generator: enable self-verification loop


@dataclass
class JavaAgentResult:
    class_name: str
    tests_generated: int
    tests_passed: int
    tests_total: int
    iterations: int
    generation_time: float
    fix_time: float
    success: bool
    skipped: bool = False  # True if class had no methods to test


class JavaTestAgent:
    """
    Unified agent for Java test generation and fixing.

    Uses existing pipeline components:
    - ASTParser for parsing Java source
    - SpecGenerator for generating test specs via LLM
    - TemplateRenderer for rendering test code
    - JavaReActLoop for fixing failing tests

    All components share a unified ProjectContext built once.
    """

    def __init__(self, config: JavaAgentConfig):
        self.config = config
        self.project_path = config.project_path
        self.api_key = config.api_key
        self.model = config.model

        # Initialize pipeline components
        self.ast_parser = ASTParser(language=Language.JAVA)
        self.spec_generator = OpenRouterSpecGenerator(api_key=config.api_key, model=config.model)
        self.validator = SpecValidator()
        self.template_renderer = TemplateRenderer()
        self.postprocessor = JavaCodePostProcessor()

        # Auto-learning: captures successful tests for future use
        self.auto_learner = FewShotAutoLearner()
        self.example_selector = FewShotExampleSelector()

        # Shared project context (built lazily on first run)
        self._project_context: ProjectContext | None = None

    def _build_project_context(self) -> ProjectContext:
        """Build unified project context by parsing all source files.

        Uses disk cache for faster subsequent runs. Cache is invalidated
        when source files change (based on file paths and modification times).
        """
        if self._project_context is not None:
            return self._project_context

        src_dir = self.project_path / "src/main/java"
        cache_dir = self.project_path / ".test-agent-cache"
        cache_file = cache_dir / "project_context.pkl"
        hash_file = cache_dir / "source_hash.txt"

        # Compute hash of source files (paths + mtimes)
        current_hash = self._compute_source_hash(src_dir)

        # Try loading from cache
        if cache_file.exists() and hash_file.exists():
            cached_hash = hash_file.read_text().strip()
            if cached_hash == current_hash:
                try:
                    with open(cache_file, "rb") as f:
                        self._project_context = pickle.load(f)
                    if self.config.verbose:
                        print(f"  Loaded project context from cache")
                    return self._project_context
                except Exception:
                    pass  # Cache corrupted, rebuild

        if self.config.verbose:
            print(f"  Building project context...")

        all_classes = []
        if src_dir.exists():
            for java_file in src_dir.rglob("*.java"):
                try:
                    _, classes = self.ast_parser.parse_file(java_file)
                    all_classes.extend(classes)
                except Exception:
                    pass  # Skip unparseable files

        self._project_context = ProjectContext.from_ast_classes(
            self.project_path, all_classes
        )

        # Save to cache
        try:
            cache_dir.mkdir(exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self._project_context, f)
            hash_file.write_text(current_hash)
            if self.config.verbose:
                print(f"  Cached project context ({len(all_classes)} classes)")
        except Exception:
            pass  # Cache write failed, continue without caching

        if self.config.verbose:
            print(f"  Found {len(all_classes)} classes in project")

        return self._project_context

    def _compute_source_hash(self, src_dir: Path) -> str:
        """Compute hash of source files for cache invalidation."""
        if not src_dir.exists():
            return "empty"

        file_info = []
        for java_file in sorted(src_dir.rglob("*.java")):
            try:
                mtime = java_file.stat().st_mtime
                file_info.append(f"{java_file}:{mtime}")
            except Exception:
                pass

        content = "\n".join(file_info)
        return hashlib.md5(content.encode()).hexdigest()

    def run(self, class_name: str, source_file: Path = None) -> JavaAgentResult:
        """
        Run full pipeline for a Java class.

        1. Build/reuse project context (shared across all runs)
        2. Parse source file with ASTParser
        3. Generate test specs with SpecGenerator (uses context)
        4. Render test code with TemplateRenderer
        5. Fix with JavaReActLoop (uses context)
        """
        start_total = time.time()

        # Find source file if not provided
        if source_file is None:
            source_file = self._find_source_file(class_name)
            if source_file is None:
                raise FileNotFoundError(f"Could not find source file for {class_name}")

        if self.config.verbose:
            print("=" * 60)
            print(f"Java Test Agent: {class_name}")
            print(f"Model: {self.model}")
            print("=" * 60)

        # Step 0: Build project context (once per agent instance)
        project_context = self._build_project_context()

        # Step 1: Parse source with ASTParser
        if self.config.verbose:
            print(f"\n[1/4] Parsing {source_file.name} with ASTParser...")

        functions, classes = self.ast_parser.parse_file(source_file)

        # Find the target class
        target_class = None
        for cls in classes:
            if cls.name == class_name:
                target_class = cls
                break

        if target_class is None:
            raise ValueError(f"Class {class_name} not found in {source_file}")

        # Apply method limit for faster evals
        if self.config.max_methods > 0 and len(target_class.methods) > self.config.max_methods:
            target_class.methods = target_class.methods[:self.config.max_methods]

        if self.config.verbose:
            print(f"  Found class: {target_class.name}")
            print(f"  Methods: {len(target_class.methods)}")
            print(f"  Constructors: {len(target_class.constructors)}")

        # Skip empty classes (no methods to test)
        if len(target_class.methods) == 0:
            if self.config.verbose:
                print(f"\n[SKIP] Class {class_name} has no methods to test")
            return JavaAgentResult(
                class_name=class_name,
                tests_generated=0,
                tests_passed=0,
                tests_total=0,
                iterations=0,
                generation_time=0,
                fix_time=0,
                success=True,  # Not a failure - just nothing to test
                skipped=True,
            )

        # Step 2 & 3: Generate test code (branch based on generator type)
        start_gen = time.time()

        if self.config.generator_type == "dsl":
            # DSL + Compiler + Feedback Loop approach
            # LLM generates structured TestSpec JSON, deterministic compiler produces Java
            if self.config.verbose:
                print(f"\n[2/4] Generating tests with SpecAgent (DSL + Compiler + Feedback Loop)...")

            spec_agent_config = SpecAgentConfig(
                project_path=self.project_path,
                api_key=self.api_key,
                model=self.model,
                max_iterations=self.config.max_iterations,
                verbose=self.config.verbose
            )
            spec_agent = SpecAgent(spec_agent_config, project_context=project_context)
            spec_result = spec_agent.run(target_class)

            gen_time = time.time() - start_gen

            if self.config.verbose:
                print(f"  Generation time: {gen_time:.1f}s")
                print(f"  SpecAgent iterations: {spec_result.iterations}")
                print(f"  Tests: {spec_result.tests_passed}/{spec_result.tests_total}")

            if spec_result.success:
                # SpecAgent already ran tests and they passed
                return JavaAgentResult(
                    class_name=class_name,
                    tests_generated=spec_result.tests_total,
                    tests_passed=spec_result.tests_passed,
                    tests_total=spec_result.tests_total,
                    iterations=spec_result.iterations,
                    generation_time=gen_time,
                    fix_time=0,  # SpecAgent includes fix iterations in its own loop
                    success=True
                )
            elif spec_result.java_code:
                # SpecAgent generated code but tests didn't all pass
                # Continue to ReAct loop for additional fixing
                test_code = spec_result.java_code
                if self.config.verbose:
                    print(f"\n[3/4] SpecAgent produced code but not all tests pass, continuing to ReAct loop...")
            else:
                # SpecAgent failed to generate valid code
                if self.config.verbose:
                    print(f"\n[FAIL] SpecAgent could not generate valid test code")
                    print(f"  Error: {spec_result.error}")
                return JavaAgentResult(
                    class_name=class_name,
                    tests_generated=0,
                    tests_passed=0,
                    tests_total=0,
                    iterations=spec_result.iterations,
                    generation_time=gen_time,
                    fix_time=0,
                    success=False
                )

        elif self.config.generator_type == "direct":
            # Direct Code + Self-Verify approach
            if self.config.verbose:
                print(f"\n[2/4] Generating tests with DirectCodeGenerator (self_verify={self.config.self_verify})...")

            # Initialize test examples finder for few-shot learning
            test_examples_finder = TestExamplesFinder(self.config.project_path)
            if test_examples_finder.has_existing_tests():
                if self.config.verbose:
                    print(f"  Found existing tests for few-shot examples")

            direct_gen = DirectCodeGenerator(
                api_key=self.api_key,
                model=self.model,
                provider="openrouter",
                self_verify=self.config.self_verify
            )
            test_code = direct_gen.generate_for_class(
                target_class,
                project_context,
                test_examples_finder=test_examples_finder
            )
            gen_time = time.time() - start_gen

            if self.config.verbose:
                print(f"  Generation time: {gen_time:.1f}s")
                print(f"\n[3/4] Skipping spec validation (direct code generation)...")

        else:
            # Spec Generation approach (default)
            if self.config.verbose:
                print(f"\n[2/4] Generating test specs with SpecGenerator...")

            specs = self.spec_generator.generate_for_class(target_class, project_context)
            gen_time = time.time() - start_gen

            if self.config.verbose:
                print(f"  Generated {len(specs)} test specs")
                print(f"  Generation time: {gen_time:.1f}s")

            # Step 3: Validate and render test code
            if self.config.verbose:
                print(f"\n[3/4] Validating and rendering test code...")

            test_code_parts = []
            for spec in specs:
                try:
                    # Validate and sanitize spec before rendering
                    validation_result = self.validator.validate(spec)
                    if not validation_result.is_valid:
                        if self.config.verbose:
                            print(f"  Warning: Invalid spec for {spec.target_name}: {validation_result.issues}")
                        continue

                    # Log any warnings from sanitization
                    if validation_result.warnings and self.config.verbose:
                        for warning in validation_result.warnings:
                            print(f"  Sanitized: {warning}")

                    # Render the sanitized spec
                    rendered = self.template_renderer.render(validation_result.sanitized_spec)
                    test_code_parts.append(rendered)
                except ValueError as e:
                    if self.config.verbose:
                        print(f"  Warning: Could not render spec: {e}")
                    continue

            # Combine into single test file
            test_code = self._combine_test_code(class_name, test_code_parts, source_file)

        # Post-process to fix common LLM patterns
        postprocess_result = self.postprocessor.process(test_code)
        if postprocess_result.was_modified:
            test_code = postprocess_result.fixed_code
            if self.config.verbose:
                print(f"  Post-processed {len(postprocess_result.changes_made)} fixes:")
                for change in postprocess_result.changes_made[:5]:  # Show first 5
                    print(f"    - {change}")

        # Write test file
        test_file = self._get_test_file_path(source_file, class_name)
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(test_code)

        if self.config.verbose:
            print(f"  Written to: {test_file}")

        # Step 3.5: Quick compile gate (informational)
        compiles, compile_error = self._quick_compile_check()
        if self.config.verbose:
            if compiles:
                print(f"  Compile check: PASSED")
            else:
                print(f"  Compile check: FAILED (ReAct loop will attempt fixes)")
                # Show first few lines of error
                error_lines = compile_error.strip().split('\n')[:5]
                for line in error_lines:
                    if line.strip():
                        print(f"    {line[:100]}")

        # Step 4: Run ReAct loop (with project context)
        if self.config.verbose:
            print(f"\n[4/4] Running ReAct loop...")

        start_fix = time.time()
        loop = JavaReActLoop(
            project_path=self.project_path,
            api_key=self.api_key,
            test_classes=[f"{class_name}Test"],
            max_iterations=self.config.max_iterations,
            verbose=self.config.verbose,
            model=self.model,
            project_context=project_context  # Share context with ReAct loop
        )
        results = loop.run()
        fix_time = time.time() - start_fix

        # Extract results - report iterations even if compilation failed
        iterations = len(results) if results else 0

        if results and results[-1].total_tests > 0:
            final = results[-1]
            result = JavaAgentResult(
                class_name=class_name,
                tests_generated=final.total_tests,
                tests_passed=final.passed_tests,
                tests_total=final.total_tests,
                iterations=iterations,
                generation_time=gen_time,
                fix_time=fix_time,
                success=final.passed_tests == final.total_tests
            )

            # Auto-learning: capture successful test for future use
            if result.success:
                source_code = source_file.read_text()
                # Read the final test code (may have been modified by ReAct loop)
                final_test_code = test_file.read_text()
                classification = self.example_selector.get_classification(target_class, source_code)

                captured = self.auto_learner.capture_if_worthy(
                    target_class=target_class,
                    source_code=source_code,
                    test_code=final_test_code,
                    result=result,
                    classification=classification,
                    project_name=self.project_path.name
                )
                if captured and self.config.verbose:
                    print(f"  [AUTO-LEARN] Captured {class_name} as learned example ({self.auto_learner.count()} total)")

            return result
        else:
            # Still report iterations even when tests=0 (compilation failure)
            return JavaAgentResult(
                class_name=class_name,
                tests_generated=0,
                tests_passed=0,
                tests_total=0,
                iterations=iterations,
                generation_time=gen_time,
                fix_time=fix_time,
                success=False
            )

    def run_multiple(self, class_names: list[str]) -> list[JavaAgentResult]:
        """Run pipeline for multiple classes."""
        results = []
        for class_name in class_names:
            try:
                result = self.run(class_name)
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"ERROR processing {class_name}: {e}")
                results.append(JavaAgentResult(
                    class_name=class_name,
                    tests_generated=0, tests_passed=0, tests_total=0,
                    iterations=0, generation_time=0, fix_time=0, success=False
                ))
        return results

    def _find_source_file(self, class_name: str) -> Path | None:
        """Find Java source file for a class."""
        src_dir = self.project_path / "src/main/java"
        for java_file in src_dir.rglob(f"{class_name}.java"):
            return java_file
        return None

    def _get_test_file_path(self, source_file: Path, class_name: str) -> Path:
        """Get test file path mirroring source structure."""
        rel_path = source_file.relative_to(self.project_path / "src/main/java")
        test_path = self.project_path / "src/test/java" / rel_path.parent / f"{class_name}Test.java"
        return test_path

    def _combine_test_code(self, class_name: str, test_parts: list[str], source_file: Path) -> str:
        """Combine multiple rendered test specs into a single test class."""
        # Extract package from source file path
        rel_path = source_file.relative_to(self.project_path / "src/main/java")
        package_parts = list(rel_path.parent.parts)
        package_name = ".".join(package_parts)

        # Detect if this is a Spring component (controller, service, etc.)
        source_content = source_file.read_text()
        is_spring_component = any(ann in source_content for ann in [
            '@Controller', '@RestController', '@Service', '@Repository', '@Component'
        ]) or class_name.endswith(('Controller', 'Service', 'Repository'))

        # Collect all imports and test methods
        imports = set()
        test_methods = []

        for part in test_parts:
            lines = part.split('\n')
            in_imports = False
            in_test = False
            current_test = []

            for line in lines:
                if line.startswith('import '):
                    imports.add(line.rstrip(';') + ';')
                elif '@Test' in line:
                    # Only collect @Test methods, skip @BeforeEach (we generate our own)
                    in_test = True
                    current_test = [line]
                elif in_test:
                    current_test.append(line)
                    if line.strip() == '}' and len(current_test) > 2:
                        test_methods.append('\n'.join(current_test))
                        in_test = False

        # Build combined test file
        code = f"package {package_name};\n\n"

        # Standard imports
        code += "import org.junit.jupiter.api.Test;\n"
        code += "import org.junit.jupiter.api.BeforeEach;\n"
        code += "import org.junit.jupiter.api.DisplayName;\n"
        code += "import java.time.LocalDate;\n"
        code += "import static org.junit.jupiter.api.Assertions.*;\n"

        if is_spring_component:
            # Add Spring/Mockito imports for controllers
            code += "import org.junit.jupiter.api.extension.ExtendWith;\n"
            code += "import org.mockito.InjectMocks;\n"
            code += "import org.mockito.Mock;\n"
            code += "import org.mockito.junit.jupiter.MockitoExtension;\n"
            code += "import static org.mockito.Mockito.*;\n"
            code += "import static org.mockito.ArgumentMatchers.*;\n"
            # Add Spring Data domain imports (commonly used by controllers)
            code += "import org.springframework.data.domain.Page;\n"
            code += "import org.springframework.data.domain.PageRequest;\n"
            code += "import org.springframework.data.domain.Pageable;\n"

        # Additional imports (filter out invalid ones)
        for imp in sorted(imports):
            # Skip Python-style imports and duplicates
            if 'from ' in imp or 'import pytest' in imp or 'import unittest' in imp:
                continue
            # Skip if already in code
            if imp in code:
                continue
            # Only add valid Java imports
            if imp.startswith('import ') and ('.' in imp or 'static ' in imp):
                code += imp + "\n"

        if is_spring_component:
            code += f"\n@ExtendWith(MockitoExtension.class)\n"
        code += f"class {class_name}Test {{\n\n"

        if is_spring_component:
            # For Spring components, use @Mock/@InjectMocks pattern
            # Try to detect injected dependencies from constructor
            import re
            constructor_match = re.search(r'public\s+' + class_name + r'\s*\(([^)]+)\)', source_content)
            if constructor_match:
                params = constructor_match.group(1)
                # Parse parameters: "TypeName paramName, TypeName2 paramName2"
                for param in params.split(','):
                    param = param.strip()
                    if param:
                        parts = param.split()
                        if len(parts) >= 2:
                            param_type = parts[-2]
                            param_name = parts[-1]
                            code += f"    @Mock\n"
                            code += f"    private {param_type} {param_name};\n\n"

            code += f"    @InjectMocks\n"
            code += f"    private {class_name} controller;\n\n"
        else:
            code += f"    private {class_name} instance;\n\n"

        code += f"    @BeforeEach\n"
        code += f"    void setUp() {{\n"
        if not is_spring_component:
            code += f"        instance = new {class_name}();\n"
        else:
            code += f"        // Mockito initializes @Mock and @InjectMocks automatically\n"
        code += f"    }}\n\n"

        # Add test methods
        for method in test_methods:
            code += "    " + method.replace('\n', '\n    ') + "\n\n"

        code += "}\n"

        return code

    def _quick_compile_check(self) -> tuple[bool, str]:
        """
        Quick compilation check using mvn test-compile (with speed optimizations).

        Returns (success, error_output).
        Uses mvnd if available for ~30-40% faster builds.
        """
        success, stdout, stderr = run_compile(
            self.project_path,
            timeout=60,
            fast=True
        )
        if success:
            return True, ""
        else:
            return False, stdout + stderr


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.java_agent <project_path> <class_name> [class_name2 ...]")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    class_names = sys.argv[2:]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    config = JavaAgentConfig(
        project_path=project_path,
        api_key=api_key
    )

    agent = JavaTestAgent(config)
    results = agent.run_multiple(class_names)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {r.class_name}: {r.tests_passed}/{r.tests_total} ({status}) - {r.iterations} iter, {r.generation_time + r.fix_time:.1f}s")


if __name__ == "__main__":
    main()
