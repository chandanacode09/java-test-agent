"""
Few-shot example selector.

Selects the most appropriate examples for a given Java class.
Priority: curated > learned

Curated examples are hand-crafted gold standards.
Learned examples are captured from successful test runs on your codebase.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .component_classifier import ComponentClassifier, ComponentType, ClassificationResult
from .auto_learner import FewShotAutoLearner, extract_keywords
from .examples import (
    ENTITY_BASIC_EXAMPLE, ENTITY_BASIC_META,
    ENTITY_COLLECTIONS_EXAMPLE, ENTITY_COLLECTIONS_META,
    CONTROLLER_PAGINATION_EXAMPLE, CONTROLLER_PAGINATION_META,
    CONTROLLER_EXCEPTION_EXAMPLE, CONTROLLER_EXCEPTION_META,
    VALIDATOR_EXAMPLE, VALIDATOR_META,
    FORMATTER_EXAMPLE, FORMATTER_META,
    SERVICE_REPOSITORY_EXAMPLE, SERVICE_REPOSITORY_META,
)

if TYPE_CHECKING:
    from ..models import ClassContext


@dataclass
class SelectedExample:
    """A selected few-shot example with metadata."""
    code: str
    component_type: ComponentType
    relevance_score: float
    reason: str


# Mapping from ComponentType to (example_code, metadata)
EXAMPLE_MAP = {
    ComponentType.ENTITY_BASIC: (ENTITY_BASIC_EXAMPLE, ENTITY_BASIC_META),
    ComponentType.ENTITY_COLLECTIONS: (ENTITY_COLLECTIONS_EXAMPLE, ENTITY_COLLECTIONS_META),
    ComponentType.CONTROLLER_PAGINATION: (CONTROLLER_PAGINATION_EXAMPLE, CONTROLLER_PAGINATION_META),
    ComponentType.CONTROLLER_EXCEPTION: (CONTROLLER_EXCEPTION_EXAMPLE, CONTROLLER_EXCEPTION_META),
    ComponentType.CONTROLLER_BASIC: (CONTROLLER_PAGINATION_EXAMPLE, CONTROLLER_PAGINATION_META),  # Use pagination as default
    ComponentType.VALIDATOR: (VALIDATOR_EXAMPLE, VALIDATOR_META),
    ComponentType.FORMATTER: (FORMATTER_EXAMPLE, FORMATTER_META),
    ComponentType.SERVICE: (SERVICE_REPOSITORY_EXAMPLE, SERVICE_REPOSITORY_META),
    # POJO uses entity_basic as fallback
    ComponentType.POJO: (ENTITY_BASIC_EXAMPLE, ENTITY_BASIC_META),
}


class FewShotExampleSelector:
    """
    Select the most relevant examples for a given class.

    Priority:
    1. Curated examples (hand-crafted, highest quality)
    2. Learned examples (from successful runs on your codebase)
    3. entity_basic fallback (always available)
    """

    def __init__(self):
        self.classifier = ComponentClassifier()
        self.auto_learner = FewShotAutoLearner()

    def select_examples(
        self,
        cls: "ClassContext",
        source_code: str = "",
        max_examples: int = 2,
        include_secondary: bool = True
    ) -> list[SelectedExample]:
        """
        Select the best curated examples for a class.

        Strategy:
        1. Classify the component type
        2. Return primary example
        3. Optionally include secondary type examples

        Args:
            cls: The class to generate tests for
            source_code: Source code of the class (for analysis)
            max_examples: Maximum examples to return
            include_secondary: Whether to include secondary pattern examples

        Returns:
            List of SelectedExample objects
        """
        classification = self.classifier.classify(cls, source_code)
        examples = []

        # Primary example
        if classification.primary_type in EXAMPLE_MAP:
            code, meta = EXAMPLE_MAP[classification.primary_type]
            examples.append(SelectedExample(
                code=code,
                component_type=classification.primary_type,
                relevance_score=classification.confidence,
                reason=f"Primary match: {', '.join(classification.signals[:3])}"  # Limit to 3 signals
            ))

        # Secondary examples if requested and we have room
        if include_secondary and len(examples) < max_examples:
            for secondary_type in classification.secondary_types:
                if secondary_type in EXAMPLE_MAP and len(examples) < max_examples:
                    # Don't add duplicate examples
                    if any(e.component_type == secondary_type for e in examples):
                        continue
                    code, meta = EXAMPLE_MAP[secondary_type]
                    examples.append(SelectedExample(
                        code=code,
                        component_type=secondary_type,
                        relevance_score=0.5,  # Lower confidence for secondary
                        reason=f"Secondary pattern: {secondary_type.value}"
                    ))

        # For controllers, always consider both pagination and exception patterns
        if (classification.primary_type in (ComponentType.CONTROLLER_PAGINATION, ComponentType.CONTROLLER_EXCEPTION, ComponentType.CONTROLLER_BASIC)
            and len(examples) < max_examples):
            # Add the other controller pattern if not already present
            other_type = (ComponentType.CONTROLLER_EXCEPTION
                         if classification.primary_type == ComponentType.CONTROLLER_PAGINATION
                         else ComponentType.CONTROLLER_PAGINATION)
            if not any(e.component_type == other_type for e in examples):
                code, meta = EXAMPLE_MAP[other_type]
                examples.append(SelectedExample(
                    code=code,
                    component_type=other_type,
                    relevance_score=0.4,
                    reason=f"Supplementary controller pattern: {other_type.value}"
                ))

        # Try learned examples if we have room and they might help
        if len(examples) < max_examples:
            learned = self._get_learned_example(
                classification.primary_type.value,
                source_code
            )
            if learned:
                examples.append(learned)

        return examples

    def _get_learned_example(
        self,
        component_type: str,
        source_code: str
    ) -> SelectedExample | None:
        """
        Get a learned example from successful past runs.

        Returns the best matching learned example based on:
        1. Component type match
        2. Keyword overlap
        3. Test count (quality indicator)
        """
        keywords = extract_keywords(source_code)
        learned = self.auto_learner.get_best_example(component_type, keywords)

        if not learned:
            return None

        # Convert learned dict to SelectedExample
        # Map string component_type back to enum
        try:
            comp_type = ComponentType(learned["component_type"])
        except ValueError:
            comp_type = ComponentType.UNKNOWN

        keyword_overlap = len(set(learned.get("keywords", [])) & set(keywords))

        return SelectedExample(
            code=learned["test_code"],
            component_type=comp_type,
            relevance_score=0.6 + (0.1 * min(keyword_overlap, 4)),  # 0.6-1.0 based on overlap
            reason=f"Learned from {learned['class_name']} ({learned['tests_passed']} tests, {keyword_overlap} keyword matches)"
        )

    def format_for_prompt(self, examples: list[SelectedExample]) -> str:
        """
        Format selected examples for injection into the LLM prompt.

        Args:
            examples: List of SelectedExample objects

        Returns:
            Formatted string ready for prompt injection
        """
        if not examples:
            return ""

        sections = [
            "## Reference Examples - Follow These Patterns",
            "",
            "The following are gold-standard test examples. CAREFULLY study them for:",
            "- Correct mocking patterns (when/thenReturn, any())",
            "- Proper assertion usage (assertThrows for exceptions, not null checks)",
            "- Setting ALL required fields in test objects",
            "- Using appropriate Errors implementations for validators",
            "",
            "IMPORTANT: Adapt these patterns to the actual class being tested.",
            ""
        ]

        for i, example in enumerate(examples, 1):
            sections.append(f"### Example {i}: {example.component_type.value}")
            sections.append(f"Relevance: {example.reason}")
            sections.append(f"```java\n{example.code.strip()}\n```")
            sections.append("")

        return "\n".join(sections)

    def get_classification(self, cls: "ClassContext", source_code: str = "") -> ClassificationResult:
        """Get the classification result for a class (for debugging/logging)."""
        return self.classifier.classify(cls, source_code)
