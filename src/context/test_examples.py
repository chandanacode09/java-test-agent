"""
Test Examples Finder - Find similar existing tests for few-shot learning.

This provides relevant test examples to improve LLM code generation by
showing real patterns from the project.
"""

from pathlib import Path
from typing import Optional
import re


class TestExamplesFinder:
    """Find existing test examples for few-shot learning."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.test_dir = project_path / "src" / "test" / "java"
        self._test_files: dict[str, Path] = {}  # class_name -> test_file_path
        self._test_contents: dict[str, str] = {}  # class_name -> test_file_content
        self._built = False

    def build(self) -> None:
        """Index all test files in the project."""
        if self._built:
            return

        if not self.test_dir.exists():
            return

        for test_file in self.test_dir.rglob("*Test.java"):
            # Extract class name from filename
            class_name = test_file.stem.replace("Test", "")
            self._test_files[class_name] = test_file

            # Read and store content
            try:
                content = test_file.read_text()
                self._test_contents[class_name] = content
            except Exception:
                pass

        # Also index Tests suffix
        for test_file in self.test_dir.rglob("*Tests.java"):
            class_name = test_file.stem.replace("Tests", "")
            if class_name not in self._test_files:
                self._test_files[class_name] = test_file
                try:
                    content = test_file.read_text()
                    self._test_contents[class_name] = content
                except Exception:
                    pass

        self._built = True

    def find_similar_tests(self, target_class: str, max_examples: int = 2) -> list[str]:
        """
        Find test files for classes similar to the target.

        Uses simple heuristics:
        1. Same package (inferred from path)
        2. Similar class name suffix (e.g., Controller, Service, Repository)
        3. Similar inheritance (Entity, etc.)

        Returns list of test file contents.
        """
        self.build()

        if not self._test_files:
            return []

        # Extract suffix pattern (e.g., "Controller", "Service", "Entity")
        suffix_match = re.search(r'([A-Z][a-z]+)$', target_class)
        target_suffix = suffix_match.group(1) if suffix_match else ""

        examples = []
        seen = set()

        # First priority: exact match (in case we already have tests for this class)
        if target_class in self._test_contents:
            # Skip - we don't want to show the target's own tests
            pass

        # Second priority: same suffix type (e.g., other Controller tests)
        if target_suffix:
            for class_name, content in self._test_contents.items():
                if class_name == target_class:
                    continue
                if class_name.endswith(target_suffix) and class_name not in seen:
                    examples.append(self._format_example(class_name, content))
                    seen.add(class_name)
                    if len(examples) >= max_examples:
                        break

        # Third priority: any other tests (if we still need more)
        if len(examples) < max_examples:
            for class_name, content in self._test_contents.items():
                if class_name == target_class or class_name in seen:
                    continue
                examples.append(self._format_example(class_name, content))
                seen.add(class_name)
                if len(examples) >= max_examples:
                    break

        return examples

    def _format_example(self, class_name: str, content: str, max_lines: int = 100) -> str:
        """Format a test file as an example, truncating if needed."""
        lines = content.split("\n")
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append("// ... (truncated)")

        return f"// Example test for {class_name}:\n" + "\n".join(lines)

    def get_test_patterns(self, target_class: str) -> Optional[str]:
        """
        Extract common testing patterns from existing tests.

        Returns a summary of patterns like:
        - Import statements used
        - Test annotations used
        - Common mock setup patterns
        """
        self.build()

        if not self._test_contents:
            return None

        patterns = []

        # Collect common patterns from all tests
        common_imports = set()
        common_annotations = set()
        mock_patterns = set()

        for content in self._test_contents.values():
            # Find imports
            imports = re.findall(r'^import\s+([\w.]+);', content, re.MULTILINE)
            for imp in imports:
                if 'org.junit' in imp or 'org.mockito' in imp or 'org.springframework.test' in imp:
                    common_imports.add(imp)

            # Find annotations
            annotations = re.findall(r'^@(\w+)', content, re.MULTILINE)
            for ann in annotations:
                if ann in ('Test', 'BeforeEach', 'AfterEach', 'Mock', 'InjectMocks',
                          'ExtendWith', 'MockitoExtension', 'WebMvcTest', 'DataJpaTest'):
                    common_annotations.add(f"@{ann}")

            # Find mock setup patterns
            mocks = re.findall(r'when\(([^)]+)\)\.thenReturn\([^)]+\)', content)
            for mock in mocks[:3]:  # Limit to avoid too many
                mock_patterns.add(f"when({mock}).thenReturn(...)")

        if common_imports:
            patterns.append("Common test imports:\n" + "\n".join(sorted(common_imports)[:10]))

        if common_annotations:
            patterns.append("Common annotations: " + ", ".join(sorted(common_annotations)))

        if mock_patterns:
            patterns.append("Mock patterns:\n" + "\n".join(sorted(mock_patterns)[:5]))

        return "\n\n".join(patterns) if patterns else None

    def has_existing_tests(self) -> bool:
        """Check if the project has any existing tests we can learn from."""
        self.build()
        return len(self._test_files) > 0
