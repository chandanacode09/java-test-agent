"""
Auto-learner for few-shot examples.

Captures successful test generations and stores them for future use.
The agent learns from its own successes on your codebase.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..java_agent import JavaAgentResult
    from ..models import ClassContext
    from .component_classifier import ClassificationResult

# Storage location
LEARNED_FILE = Path(__file__).parent / "learned.json"


@dataclass
class LearnedExample:
    """A learned example from a successful test run."""
    hash: str
    class_name: str
    component_type: str
    keywords: list[str]
    test_code: str
    source_code: str
    tests_passed: int
    iterations: int
    timestamp: str
    project: str


def extract_keywords(source_code: str) -> list[str]:
    """
    Extract keywords from Java source code for matching.

    Uses simple pattern matching - fast and deterministic.
    """
    keywords = []

    patterns = {
        # Annotations
        "@Entity": "entity",
        "@MappedSuperclass": "entity",
        "@RestController": "rest_controller",
        "@Controller": "controller",
        "@Service": "service",
        "@Repository": "repository",
        "@Component": "component",
        "@WebMvcTest": "webmvc",
        "@DataJpaTest": "data_jpa",
        "@Valid": "validation",

        # JPA relationships
        "@ManyToMany": "many_to_many",
        "@OneToMany": "one_to_many",
        "@ManyToOne": "many_to_one",
        "@OneToOne": "one_to_one",

        # Patterns
        "Pageable": "pagination",
        "Page<": "pagination",
        "orElseThrow": "exception_handling",
        "Optional<": "optional",
        "ResponseEntity": "response_entity",

        # Interfaces
        "implements Validator": "validator",
        "implements Formatter": "formatter",

        # Collections
        "Set<": "collection_set",
        "List<": "collection_list",
    }

    for pattern, keyword in patterns.items():
        if pattern in source_code:
            keywords.append(keyword)

    return keywords


def compute_hash(test_code: str) -> str:
    """Compute MD5 hash of test code for deduplication."""
    # Normalize: remove whitespace variations
    normalized = " ".join(test_code.split())
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


class FewShotAutoLearner:
    """
    Captures successful test generations for future use.

    Quality gates ensure only good examples are saved:
    - All tests must pass
    - At least 2 tests generated
    - Needed 3 or fewer fix iterations
    """

    def __init__(self):
        self.learned_file = LEARNED_FILE
        self._examples: list[dict] | None = None

    def _load_examples(self) -> list[dict]:
        """Load existing examples from disk."""
        if self._examples is not None:
            return self._examples

        if self.learned_file.exists():
            try:
                data = json.loads(self.learned_file.read_text())
                self._examples = data.get("examples", [])
            except (json.JSONDecodeError, IOError):
                self._examples = []
        else:
            self._examples = []

        return self._examples

    def _save_examples(self, examples: list[dict]) -> None:
        """Save examples to disk."""
        self._examples = examples
        self.learned_file.write_text(json.dumps(
            {"examples": examples},
            indent=2
        ))

    def should_save(self, result: "JavaAgentResult") -> bool:
        """Check if result passes quality gates."""
        return (
            result.success and
            result.tests_passed >= 2 and
            result.iterations <= 3
        )

    def capture_if_worthy(
        self,
        target_class: "ClassContext",
        source_code: str,
        test_code: str,
        result: "JavaAgentResult",
        classification: "ClassificationResult",
        project_name: str = ""
    ) -> bool:
        """
        Capture a successful test as a learned example.

        Returns True if captured, False if skipped.
        """
        if not self.should_save(result):
            return False

        # Compute hash for deduplication
        code_hash = compute_hash(test_code)

        # Check for duplicates
        examples = self._load_examples()
        if any(e["hash"] == code_hash for e in examples):
            return False  # Already have this one

        # Check if we have a better example for this class
        existing = [e for e in examples if e["class_name"] == target_class.name]
        if existing:
            best = max(existing, key=lambda e: e["tests_passed"])
            if best["tests_passed"] >= result.tests_passed:
                return False  # Already have a better or equal example
            # Remove inferior examples for this class
            examples = [e for e in examples if e["class_name"] != target_class.name]

        # Extract keywords for matching
        keywords = extract_keywords(source_code)

        # Create learned example
        example = LearnedExample(
            hash=code_hash,
            class_name=target_class.name,
            component_type=classification.primary_type.value,
            keywords=keywords,
            test_code=test_code,
            source_code=source_code[:2000],  # Truncate for storage
            tests_passed=result.tests_passed,
            iterations=result.iterations,
            timestamp=datetime.now().isoformat(),
            project=project_name
        )

        examples.append(asdict(example))
        self._save_examples(examples)

        return True

    def get_examples_by_type(self, component_type: str) -> list[dict]:
        """Get learned examples matching a component type."""
        examples = self._load_examples()
        return [e for e in examples if e["component_type"] == component_type]

    def get_examples_by_keywords(self, keywords: list[str], limit: int = 3) -> list[dict]:
        """
        Get learned examples with most keyword overlap.

        Returns examples sorted by:
        1. Keyword overlap count (descending)
        2. Tests passed (descending)
        """
        examples = self._load_examples()

        def score(example: dict) -> tuple[int, int]:
            overlap = len(set(example.get("keywords", [])) & set(keywords))
            tests = example.get("tests_passed", 0)
            return (overlap, tests)

        # Filter to examples with at least 1 keyword match
        matched = [e for e in examples if score(e)[0] > 0]
        matched.sort(key=score, reverse=True)

        return matched[:limit]

    def get_best_example(
        self,
        component_type: str,
        keywords: list[str]
    ) -> dict | None:
        """
        Get the single best learned example for a class.

        Strategy:
        1. Same component_type + keyword overlap
        2. Just keyword overlap (cross-type learning)
        3. Just component_type match
        """
        examples = self._load_examples()

        # Score each example
        def score(example: dict) -> tuple[int, int, int]:
            type_match = 1 if example.get("component_type") == component_type else 0
            keyword_overlap = len(set(example.get("keywords", [])) & set(keywords))
            tests = example.get("tests_passed", 0)
            return (type_match, keyword_overlap, tests)

        if not examples:
            return None

        best = max(examples, key=score)

        # Only return if there's at least some relevance
        type_match, keyword_overlap, _ = score(best)
        if type_match == 0 and keyword_overlap == 0:
            return None

        return best

    def count(self) -> int:
        """Return number of learned examples."""
        return len(self._load_examples())
