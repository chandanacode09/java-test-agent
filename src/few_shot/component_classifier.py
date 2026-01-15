"""
Component classifier for Java classes.

Analyzes a Java class to determine its type (entity, controller, validator, etc.)
for selecting appropriate few-shot examples.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ClassContext


class ComponentType(Enum):
    """Types of Spring/Java components we can generate tests for."""
    ENTITY_BASIC = "entity_basic"                    # JPA entity with simple getters/setters
    ENTITY_COLLECTIONS = "entity_collections"        # Entity with collection management
    CONTROLLER_PAGINATION = "controller_pagination"  # Controller using Pageable/Page
    CONTROLLER_EXCEPTION = "controller_exception"    # Controller with orElseThrow
    CONTROLLER_BASIC = "controller_basic"            # Basic controller
    VALIDATOR = "validator"                          # Spring Validator implementation
    FORMATTER = "formatter"                          # Spring Formatter implementation
    SERVICE = "service"                              # Service with repository dependency
    REPOSITORY = "repository"                        # Spring Data Repository (skip)
    POJO = "pojo"                                    # Plain Java object
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of classifying a Java class."""
    primary_type: ComponentType
    secondary_types: list[ComponentType] = field(default_factory=list)
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)


class ComponentClassifier:
    """Classify Java classes by their component type for example selection."""

    def classify(self, cls: "ClassContext", source_code: str = "") -> ClassificationResult:
        """
        Classify a class based on:
        1. Annotations (@Entity, @Controller, @Service, @Component)
        2. Implemented interfaces (Validator, Formatter)
        3. Class name suffix (Controller, Service, Repository)
        4. Method patterns (findAll, orElseThrow, addX/removeX)
        5. Constructor dependencies (Repository injection)
        """
        signals = []
        scores: dict[ComponentType, float] = {t: 0.0 for t in ComponentType}

        # Get source code if not provided
        if not source_code and cls.location and cls.location.file_path:
            try:
                source_code = cls.location.file_path.read_text()
            except:
                source_code = ""

        # Check annotations
        if self._has_annotation(source_code, "Entity", "MappedSuperclass"):
            signals.append("@Entity or @MappedSuperclass annotation")
            if self._has_collection_methods(cls):
                scores[ComponentType.ENTITY_COLLECTIONS] += 1.0
                signals.append("Has add/remove collection methods")
            else:
                scores[ComponentType.ENTITY_BASIC] += 1.0

        if self._has_annotation(source_code, "Controller", "RestController"):
            signals.append("@Controller annotation")
            if self._uses_pagination(source_code):
                scores[ComponentType.CONTROLLER_PAGINATION] += 0.8
                signals.append("Uses Pageable/Page")
            if self._uses_or_else_throw(source_code):
                scores[ComponentType.CONTROLLER_EXCEPTION] += 0.8
                signals.append("Uses orElseThrow pattern")
            # Default controller score if no specific pattern
            if scores[ComponentType.CONTROLLER_PAGINATION] == 0 and scores[ComponentType.CONTROLLER_EXCEPTION] == 0:
                scores[ComponentType.CONTROLLER_BASIC] += 0.5

        if self._has_annotation(source_code, "Service"):
            scores[ComponentType.SERVICE] += 1.0
            signals.append("@Service annotation")

        # @Component with Repository injection is a service pattern
        if self._has_annotation(source_code, "Component") and "Repository" in source_code:
            scores[ComponentType.SERVICE] += 0.8
            signals.append("@Component with Repository injection (service pattern)")

        if self._has_annotation(source_code, "Repository"):
            scores[ComponentType.REPOSITORY] += 1.0
            signals.append("@Repository annotation")

        # Check interfaces via base classes
        base_classes = getattr(cls, 'base_classes', []) or []

        if "Validator" in base_classes or self._implements_interface(source_code, "Validator"):
            scores[ComponentType.VALIDATOR] += 1.0
            signals.append("Implements Validator")

        if "Formatter" in base_classes or self._implements_interface(source_code, "Formatter"):
            scores[ComponentType.FORMATTER] += 1.0
            signals.append("Implements Formatter")

        # Check class name patterns
        class_name = cls.name
        if class_name.endswith("Controller"):
            if scores[ComponentType.CONTROLLER_PAGINATION] == 0 and scores[ComponentType.CONTROLLER_EXCEPTION] == 0:
                scores[ComponentType.CONTROLLER_BASIC] += 0.3
                signals.append("Class name ends with Controller")
        elif class_name.endswith("Service") or class_name.endswith("Services"):
            scores[ComponentType.SERVICE] += 0.3
            signals.append("Class name ends with Service/Services")
        elif class_name.endswith("Repository"):
            scores[ComponentType.REPOSITORY] += 0.3
            signals.append("Class name ends with Repository")
        elif class_name.endswith("Validator"):
            scores[ComponentType.VALIDATOR] += 0.3
            signals.append("Class name ends with Validator")
        elif class_name.endswith("Formatter"):
            scores[ComponentType.FORMATTER] += 0.3
            signals.append("Class name ends with Formatter")

        # Determine primary type
        primary_type = max(scores, key=lambda t: scores[t])
        if scores[primary_type] == 0:
            # Default to POJO if nothing matches
            primary_type = ComponentType.POJO
            signals.append("No specific component signals, treating as POJO")

        # Find secondary types (score > 0.3 and not primary)
        secondary = [t for t, s in scores.items() if s > 0.3 and t != primary_type]

        return ClassificationResult(
            primary_type=primary_type,
            secondary_types=secondary,
            confidence=scores[primary_type] if primary_type != ComponentType.POJO else 0.5,
            signals=signals
        )

    def _has_annotation(self, source: str, *annotations: str) -> bool:
        """Check if source code has any of the given annotations."""
        return any(f"@{ann}" in source for ann in annotations)

    def _implements_interface(self, source: str, interface: str) -> bool:
        """Check if class implements an interface."""
        return f"implements {interface}" in source or f"implements {interface}<" in source

    def _has_collection_methods(self, cls: "ClassContext") -> bool:
        """Check if class has collection management methods (addX, removeX)."""
        methods = getattr(cls, 'methods', []) or []
        method_names = [m.name for m in methods]
        return any(
            n.startswith("add") or n.startswith("remove") or n.startswith("get")
            for n in method_names
            if n not in ("getId", "getName", "getClass")  # Exclude common getters
        )

    def _uses_pagination(self, source: str) -> bool:
        """Check if code uses Spring pagination."""
        return "Pageable" in source or "Page<" in source or "PageRequest" in source

    def _uses_or_else_throw(self, source: str) -> bool:
        """Check if code uses Optional.orElseThrow pattern."""
        return "orElseThrow" in source
