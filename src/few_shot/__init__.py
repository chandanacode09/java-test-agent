"""
Few-shot example library for Java test generation.

Provides curated, gold-standard test examples for different component types
(controllers, entities, validators, formatters) that can be used across
any Spring Boot/JPA project.

Also supports auto-learning from successful test runs on your codebase.
"""

from .component_classifier import ComponentClassifier, ComponentType, ClassificationResult
from .example_selector import FewShotExampleSelector, SelectedExample
from .auto_learner import FewShotAutoLearner, extract_keywords, LearnedExample

__all__ = [
    "ComponentClassifier",
    "ComponentType",
    "ClassificationResult",
    "FewShotExampleSelector",
    "SelectedExample",
    "FewShotAutoLearner",
    "extract_keywords",
    "LearnedExample",
]
