"""
Template Renderer - Converts TestSpec to test code using Jinja2 templates.

This is the deterministic part of the pipeline - same spec = same output.
"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import TestSpec, TestType, Language, ReturnSemantics


class TemplateRenderer:
    """
    Render test specifications into actual test code.

    Uses Jinja2 templates for deterministic, consistent output.
    Now with smart template selection based on semantic analysis.
    """

    # Map test types to template names (default fallback)
    TEMPLATE_MAP = {
        TestType.UNIT_PURE: "unit_pure.py.j2",
        TestType.UNIT_CLASS: "unit_class.py.j2",
        TestType.UNIT_MOCKED: "unit_mocked.py.j2",
        TestType.EDGE_CASE: "edge_case.py.j2",
        TestType.INTEGRATION_API: "integration_api.py.j2",
        TestType.INTEGRATION_DB: "integration_db.py.j2",
        TestType.CUSTOM: None,  # Fallback to LLM generation
    }

    # Semantic-based template selection (higher priority)
    SEMANTIC_TEMPLATES = {
        'flask_context': 'flask_context.py.j2',
        'generator': 'generator_function.py.j2',
        'side_effect': 'side_effect.py.j2',
        'class_method': 'unit_method.py.j2',
        'spring_controller': 'spring_controller.py.j2',  # For Spring MVC controllers
    }

    def __init__(self, templates_dir: Path | str | None = None):
        """
        Initialize the template renderer.

        Args:
            templates_dir: Path to templates directory.
                          Defaults to package templates directory.
        """
        if templates_dir is None:
            # Use package templates
            templates_dir = Path(__file__).parent.parent.parent / "templates"

        self.templates_dir = Path(templates_dir)
        self._env = None

    @property
    def env(self) -> Environment:
        """Lazy initialization of Jinja2 environment."""
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader(self.templates_dir),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            # Add custom filters
            self._env.filters['safe'] = lambda x: x  # Pass through for test values
        return self._env

    def render(self, spec: TestSpec) -> str:
        """
        Render a TestSpec to test code.

        Args:
            spec: The test specification to render

        Returns:
            Generated test code as a string
        """
        # Check if template exists for this test type
        if spec.requires_custom_generation or spec.test_type == TestType.CUSTOM:
            raise ValueError(
                f"TestSpec requires custom generation. Use SpecGenerator.generate_custom() instead."
            )

        template_name = self._get_template_name(spec)
        if template_name is None:
            raise ValueError(f"No template available for test type: {spec.test_type}")

        template = self._load_template(spec.language, template_name)

        # Prepare template context
        context = self._prepare_context(spec)

        # Render
        code = template.render(**context)

        # Post-process for Java to fix common Python→Java syntax leakage
        if spec.language == Language.JAVA:
            code = self._sanitize_java_output(code)

        return code

    def render_to_file(self, spec: TestSpec, output_path: Path) -> Path:
        """
        Render a TestSpec and save to a file.

        Args:
            spec: The test specification to render
            output_path: Path to save the test file

        Returns:
            Path to the created file
        """
        code = self.render(spec)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)

        return output_path

    def render_multiple(self, specs: list[TestSpec]) -> dict[str, str]:
        """
        Render multiple TestSpecs.

        Args:
            specs: List of test specifications

        Returns:
            Dict mapping target names to generated code
        """
        results = {}
        for spec in specs:
            try:
                code = self.render(spec)
                results[spec.target_name] = code
            except ValueError as e:
                # Mark custom generation needed
                results[spec.target_name] = f"# Custom generation required: {e}"
        return results

    def can_render(self, spec: TestSpec) -> bool:
        """Check if a spec can be rendered with templates."""
        if spec.requires_custom_generation:
            return False
        template_name = self._get_template_name(spec)
        if template_name is None:
            return False
        return self._template_exists(spec.language, template_name)

    def _is_spring_component(self, spec: TestSpec) -> bool:
        """Check if spec is for a Spring component that requires @Mock/@InjectMocks pattern."""
        # Check framework hints
        if spec.framework_hints:
            spring_hints = ['spring_mvc', 'spring', 'controller', 'service', 'repository']
            for hint in spec.framework_hints:
                if hint.lower() in spring_hints:
                    return True

        # Check target class name
        if spec.target_class:
            if spec.target_class.endswith(('Controller', 'Service', 'Repository')):
                return True

        return False

    def _get_template_name(self, spec: TestSpec) -> str | None:
        """Get the template name using smart semantic selection.

        Priority order:
        0. Spring controller → spring_controller.py.j2 (highest priority for Java controllers)
        1. Class method → unit_method.py.j2
        2. Flask context required → flask_context.py.j2
        3. Generator function → generator_function.py.j2
        4. Side-effect function → side_effect.py.j2
        5. Default test type mapping
        """
        # Priority 0: Spring MVC Controller (check framework hints)
        if spec.language == Language.JAVA and self._is_spring_component(spec):
            return self.SEMANTIC_TEMPLATES['spring_controller']

        # Priority 1: Class method (has target_class)
        if spec.target_class:
            return self.SEMANTIC_TEMPLATES['class_method']

        # Priority 2: Flask context
        if spec.requires_context and 'flask_app' in spec.requires_context:
            return self.SEMANTIC_TEMPLATES['flask_context']

        # Priority 3: Generator function
        if spec.is_generator or spec.return_semantics == ReturnSemantics.GENERATOR:
            return self.SEMANTIC_TEMPLATES['generator']

        # Priority 4: Iterator return type
        if spec.return_semantics == ReturnSemantics.ITERATOR:
            return self.SEMANTIC_TEMPLATES['generator']

        # Priority 5: Side-effect function
        if spec.mutates_args or spec.return_semantics == ReturnSemantics.NONE_SIDEEFFECT:
            return self.SEMANTIC_TEMPLATES['side_effect']

        # Default: Use test type mapping
        return self.TEMPLATE_MAP.get(spec.test_type)

    def _load_template(self, language: Language, template_name: str):
        """Load a template for the given language."""
        # Map language to template subdirectory
        lang_dir = {
            Language.PYTHON: "python",
            Language.TYPESCRIPT: "typescript",
            Language.JAVASCRIPT: "typescript",  # Share with TypeScript
            Language.JAVA: "java",
        }.get(language, "python")

        # For Java, map Python template names to Java equivalents
        if language == Language.JAVA:
            template_name = template_name.replace('.py.j2', '.java.j2')

        template_path = f"{lang_dir}/{template_name}"

        try:
            return self.env.get_template(template_path)
        except Exception as e:
            raise ValueError(f"Template not found: {template_path}") from e

    def _template_exists(self, language: Language, template_name: str) -> bool:
        """Check if a template exists."""
        lang_dir = {
            Language.PYTHON: "python",
            Language.TYPESCRIPT: "typescript",
            Language.JAVASCRIPT: "typescript",
            Language.JAVA: "java",
        }.get(language, "python")

        # For Java, map Python template names to Java equivalents
        if language == Language.JAVA:
            template_name = template_name.replace('.py.j2', '.java.j2')

        template_path = self.templates_dir / lang_dir / template_name
        return template_path.exists()

    def _prepare_context(self, spec: TestSpec) -> dict:
        """Prepare the template context from a TestSpec."""
        # Normalize file path to relative module path
        target_file = self._normalize_import_path(spec.target_file, spec.language)

        # For Java, extract package name from the normalized path
        package_name = None
        if spec.language == Language.JAVA:
            parts = target_file.rsplit('.', 1)
            if len(parts) > 1:
                package_name = parts[0]  # Everything before the class name

        # Infer if function has return type from test cases
        has_return_type = any(
            tc.expected.returns is not None
            for tc in spec.test_cases
        )

        return {
            "test_type": spec.test_type.value,
            "target_file": target_file,
            "target_name": spec.target_name,
            "target_class": spec.target_class,
            "language": spec.language.value,
            "test_cases": spec.test_cases,
            "fixtures_needed": spec.fixtures_needed,
            "imports_needed": spec.imports_needed,
            "parametrize": spec.parametrize,
            "complexity_score": spec.complexity_score,
            # Java-specific
            "package_name": package_name,
            "return_type": has_return_type,  # Boolean for template to use
            # NEW: Semantic fields for template selection
            "is_generator": spec.is_generator,
            "return_semantics": spec.return_semantics.value if spec.return_semantics else "value",
            "requires_context": spec.requires_context,
            "mutates_args": spec.mutates_args,
            "framework_hints": spec.framework_hints,
        }

    def _sanitize_java_output(self, code: str) -> str:
        """Fix common Python syntax that leaks into Java output."""
        import re

        # Replace Python None with Java null
        code = re.sub(r'\bNone\b', 'null', code)

        # Replace string "null" with proper null keyword in method calls
        # Catches: method("null") -> method(null)
        code = re.sub(r'\("null"\)', '(null)', code)
        code = re.sub(r', "null"\)', ', null)', code)
        code = re.sub(r'\("null",', '(null,', code)

        # Replace Python True/False with Java true/false
        code = re.sub(r'\bTrue\b', 'true', code)
        code = re.sub(r'\bFalse\b', 'false', code)

        # Replace Python string quotes for booleans
        code = re.sub(r'"true"', 'true', code)
        code = re.sub(r'"false"', 'false', code)

        # Replace Mock(spec=X) with mock(X.class)
        code = re.sub(r'Mock\(spec=(\w+)\)', r'mock(\1.class)', code)

        # Replace Python dicts in assertEquals with assertNotNull (can't compare dicts in Java)
        # Match both single and double quote styles: {'key': 'value'} or {"key": "value"}
        code = re.sub(r"assertEquals\(\{[^}]+\},\s*result\);", "assertNotNull(result);", code)

        # Handle multi-line dicts
        lines = code.split('\n')
        new_lines = []
        skip_until_semicolon = False
        for line in lines:
            if skip_until_semicolon:
                if ';' in line:
                    skip_until_semicolon = False
                continue
            if "assertEquals({" in line or "assertEquals( {" in line:
                # Replace with assertNotNull
                if ';' in line:
                    new_lines.append(line.split('assertEquals')[0] + "assertNotNull(result);")
                else:
                    new_lines.append(line.split('assertEquals')[0] + "assertNotNull(result);")
                    skip_until_semicolon = True
            else:
                new_lines.append(line)
        code = '\n'.join(new_lines)

        return code

    def _normalize_import_path(self, file_path: str, language: Language = Language.PYTHON) -> str:
        """Convert a file path to a valid import path for the target language."""
        path = Path(file_path)

        # Handle Java package paths
        if language == Language.JAVA:
            path_str = str(path)
            # Look for standard Java source roots
            for marker in ['/src/main/java/', '/src/test/java/', '/java/']:
                if marker in path_str:
                    idx = path_str.rfind(marker)
                    relative = path_str[idx + len(marker):]
                    # Convert path to package: com/example/Foo.java -> com.example.Foo
                    package_path = relative.replace('.java', '').replace('/', '.')
                    return package_path
            # Fallback: just use filename without extension
            return path.stem

        # If it's an absolute path, try to make it relative to package roots
        if path.is_absolute():
            path_str = str(path)

            # For Flask specifically - look for /flask/ and use 'flask.' prefix
            if '/flask/' in path_str:
                # Get from last /flask/ onwards
                idx = path_str.rfind('/flask/')
                relative = path_str[idx + 1:]  # Includes 'flask/'
                path = Path(relative)
            # For examples/ in our repo
            elif '/examples/' in path_str:
                idx = path_str.rfind('/examples/')
                relative = path_str[idx + 1:]
                path = Path(relative)
            # Common patterns: /src/, /lib/
            elif '/src/' in path_str:
                idx = path_str.rfind('/src/')
                relative = path_str[idx + 5:]  # Skip /src/
                path = Path(relative)
            elif '/lib/' in path_str:
                idx = path_str.rfind('/lib/')
                relative = path_str[idx + 5:]  # Skip /lib/
                path = Path(relative)
            else:
                # Just use the filename
                path = Path(path.name)

        # Remove .py extension and convert to module path
        module_path = str(path).replace('.py', '').replace('/', '.')

        # Clean up leading dots
        module_path = module_path.lstrip('.')

        return module_path

    def get_output_filename(self, spec: TestSpec) -> str:
        """Generate the output filename for a test spec."""
        base_name = Path(spec.target_file).stem

        # Java uses different naming convention (CamelCase, no test_ prefix, .java extension)
        if spec.language == Language.JAVA:
            if spec.target_class:
                # e.g., OwnerServiceFindByIdTest.java
                method_name = ''.join(word.title() for word in spec.target_name.split('_'))
                return f"{spec.target_class}{method_name}Test.java"
            else:
                # e.g., CalculateAgeTest.java
                class_name = ''.join(word.title() for word in base_name.split('_'))
                return f"{class_name}Test.java"

        # Python naming convention
        if spec.target_class:
            return f"test_{base_name}_{spec.target_class.lower()}_{spec.target_name}.py"
        else:
            return f"test_{base_name}_{spec.target_name}.py"

    def list_available_templates(self) -> dict[str, list[str]]:
        """List all available templates by language."""
        result = {}

        for lang_dir in self.templates_dir.iterdir():
            if lang_dir.is_dir():
                templates = [
                    f.name for f in lang_dir.glob("*.j2")
                ]
                result[lang_dir.name] = templates

        return result


class TemplateSelector:
    """
    Select the best template for a given TestSpec.

    Considers complexity, test patterns, and available templates.
    """

    def __init__(self, renderer: TemplateRenderer):
        self.renderer = renderer

    def select(self, spec: TestSpec) -> str | None:
        """
        Select the best template for a spec.

        Returns:
            Template name or None if custom generation is needed
        """
        # High complexity = custom generation
        if spec.complexity_score > 8:
            return None

        # Check if spec has mocks
        has_mocks = any(tc.mocks for tc in spec.test_cases)

        # Check if all cases are error handling
        all_errors = all(
            tc.expected.raises is not None
            for tc in spec.test_cases
        )

        # Select template based on characteristics
        if spec.test_type == TestType.CUSTOM:
            return None

        if has_mocks:
            return "unit_mocked.py.j2"

        if all_errors:
            return "edge_case.py.j2"

        if spec.target_class:
            return "unit_class.py.j2"

        return "unit_pure.py.j2"

    def should_use_parametrize(self, spec: TestSpec) -> bool:
        """Determine if parametrize should be used."""
        if len(spec.test_cases) < 2:
            return False

        # Check if all test cases have same input structure
        if not spec.test_cases:
            return False

        first_inputs = set(spec.test_cases[0].inputs.keys())
        same_structure = all(
            set(tc.inputs.keys()) == first_inputs
            for tc in spec.test_cases
        )

        # Check if all are happy path with simple returns
        all_simple = all(
            tc.expected.raises is None and not tc.mocks
            for tc in spec.test_cases
        )

        return same_structure and all_simple and len(spec.test_cases) >= 3
