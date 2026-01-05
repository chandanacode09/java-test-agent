#!/usr/bin/env python3
"""
Unified Java Test Agent.

Orchestrates the existing pipeline components:
1. ASTParser - Java source parsing
2. SpecGenerator - LLM-based test spec generation
3. TemplateRenderer - Test code rendering
4. JavaReActLoop - Test fixing
"""
import os
import time
from pathlib import Path
from dataclasses import dataclass

from .context.ast_parser import ASTParser
from .generator.spec_generator import OpenRouterSpecGenerator
from .renderer.template_renderer import TemplateRenderer
from .react_loop_java import JavaReActLoop
from .models import Language


@dataclass
class JavaAgentConfig:
    project_path: Path
    api_key: str
    model: str = "x-ai/grok-code-fast-1"
    max_iterations: int = 5
    verbose: bool = True


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


class JavaTestAgent:
    """
    Unified agent for Java test generation and fixing.

    Uses existing pipeline components:
    - ASTParser for parsing Java source
    - SpecGenerator for generating test specs via LLM
    - TemplateRenderer for rendering test code
    - JavaReActLoop for fixing failing tests
    """

    def __init__(self, config: JavaAgentConfig):
        self.config = config
        self.project_path = config.project_path
        self.api_key = config.api_key
        self.model = config.model

        # Initialize pipeline components
        self.ast_parser = ASTParser(language=Language.JAVA)
        self.spec_generator = OpenRouterSpecGenerator(api_key=config.api_key, model=config.model)
        self.template_renderer = TemplateRenderer()

    def run(self, class_name: str, source_file: Path = None) -> JavaAgentResult:
        """
        Run full pipeline for a Java class.

        1. Parse source file with ASTParser
        2. Generate test specs with SpecGenerator
        3. Render test code with TemplateRenderer
        4. Fix with JavaReActLoop
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

        if self.config.verbose:
            print(f"  Found class: {target_class.name}")
            print(f"  Methods: {len(target_class.methods)}")
            print(f"  Constructors: {len(target_class.constructors)}")

        # Step 2: Generate test specs with SpecGenerator
        if self.config.verbose:
            print(f"\n[2/4] Generating test specs with SpecGenerator...")

        start_gen = time.time()
        specs = self.spec_generator.generate_for_class(target_class)
        gen_time = time.time() - start_gen

        if self.config.verbose:
            print(f"  Generated {len(specs)} test specs")
            print(f"  Generation time: {gen_time:.1f}s")

        # Step 3: Render test code with TemplateRenderer
        if self.config.verbose:
            print(f"\n[3/4] Rendering test code with TemplateRenderer...")

        test_code_parts = []
        for spec in specs:
            try:
                rendered = self.template_renderer.render(spec)
                test_code_parts.append(rendered)
            except ValueError as e:
                if self.config.verbose:
                    print(f"  Warning: Could not render spec: {e}")
                continue

        # Combine into single test file
        test_code = self._combine_test_code(class_name, test_code_parts, source_file)

        # Write test file
        test_file = self._get_test_file_path(source_file, class_name)
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(test_code)

        if self.config.verbose:
            print(f"  Written to: {test_file}")

        # Step 4: Run ReAct loop
        if self.config.verbose:
            print(f"\n[4/4] Running ReAct loop...")

        start_fix = time.time()
        loop = JavaReActLoop(
            project_path=self.project_path,
            api_key=self.api_key,
            test_classes=[f"{class_name}Test"],
            max_iterations=self.config.max_iterations,
            verbose=self.config.verbose,
            model=self.model
        )
        results = loop.run()
        fix_time = time.time() - start_fix

        # Extract results
        if results and results[-1].total_tests > 0:
            final = results[-1]
            return JavaAgentResult(
                class_name=class_name,
                tests_generated=final.total_tests,
                tests_passed=final.passed_tests,
                tests_total=final.total_tests,
                iterations=len(results),
                generation_time=gen_time,
                fix_time=fix_time,
                success=final.passed_tests == final.total_tests
            )
        else:
            return JavaAgentResult(
                class_name=class_name,
                tests_generated=0,
                tests_passed=0,
                tests_total=0,
                iterations=0,
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
                elif '@Test' in line or '@BeforeEach' in line:
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

        # Additional imports
        for imp in sorted(imports):
            if imp not in code:
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
