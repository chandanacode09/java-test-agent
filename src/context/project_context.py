"""
ProjectContext - Unified context for Java project analysis.

This module provides a shared context that all pipeline components use,
eliminating duplicate parsing and ensuring consistent API information
across SpecGenerator, TemplateRenderer, and ReAct Loop.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..models import ClassContext, FunctionContext
from .dependency_stubs import get_stub_context_for_classes, get_methods_for_class


@dataclass
class MethodSignature:
    """Compact representation of a method signature for prompts."""
    name: str
    parameters: list[tuple[str, str]]  # [(type, name), ...]
    return_type: str
    is_static: bool = False


@dataclass
class ClassInfo:
    """Information about a Java class for context building."""
    name: str
    fully_qualified_name: str
    package: str
    base_classes: list[str]
    constructors: list[MethodSignature]
    methods: list[MethodSignature]
    fields: list[tuple[str, str]]  # [(type, name), ...]
    is_interface: bool = False
    is_abstract: bool = False
    annotations: list[str] = field(default_factory=list)

    def get_all_methods(self, project_context: 'ProjectContext') -> list[MethodSignature]:
        """Get all methods including inherited ones."""
        all_methods = list(self.methods)

        # Add methods from parent classes
        for base in self.base_classes:
            parent = project_context.get_class(base)
            if parent:
                # Add parent methods that aren't overridden
                existing_names = {m.name for m in all_methods}
                for method in parent.methods:
                    if method.name not in existing_names:
                        all_methods.append(method)
                # Recursively get grandparent methods
                for grandparent_method in parent.get_all_methods(project_context):
                    if grandparent_method.name not in existing_names:
                        all_methods.append(grandparent_method)
                        existing_names.add(grandparent_method.name)

        return all_methods


class ProjectContext:
    """
    Unified context for a Java project.

    Built once by AST Parser, shared by all pipeline components.
    Provides:
    - All classes with their methods, constructors, inheritance
    - Quick lookup by class name
    - API context string generation for LLM prompts
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._classes: dict[str, ClassInfo] = {}
        self._fqn_to_name: dict[str, str] = {}  # fully.qualified.Name -> Name
        self._built = False

    def add_class(self, class_info: ClassInfo) -> None:
        """Add a class to the context."""
        self._classes[class_info.name] = class_info
        self._fqn_to_name[class_info.fully_qualified_name] = class_info.name

    def get_class(self, name: str) -> Optional[ClassInfo]:
        """Get class by simple name or fully qualified name."""
        if name in self._classes:
            return self._classes[name]
        # Try as FQN
        simple_name = self._fqn_to_name.get(name)
        if simple_name:
            return self._classes.get(simple_name)
        return None

    def get_all_classes(self) -> list[ClassInfo]:
        """Get all classes in the project."""
        return list(self._classes.values())

    @property
    def is_built(self) -> bool:
        """Check if context has been built."""
        return self._built

    def mark_built(self) -> None:
        """Mark context as fully built."""
        self._built = True

    def build_api_context_for_class(self, target_class: str) -> str:
        """
        Build API context string for a target class.

        Includes:
        - Target class constructors and methods
        - Parent class methods (inheritance chain)
        - Related classes (from method parameters/returns)

        This is what gets included in LLM prompts.
        """
        if not self._classes:
            return "// No API information available"

        target = self.get_class(target_class)
        if not target:
            return f"// Class {target_class} not found in project"

        # Collect relevant classes
        relevant_classes: set[str] = {target_class}

        # Add parent classes (inheritance chain)
        to_process = list(target.base_classes)
        while to_process:
            parent_name = to_process.pop()
            parent = self.get_class(parent_name)
            if parent and parent.name not in relevant_classes:
                relevant_classes.add(parent.name)
                to_process.extend(parent.base_classes)

        # Add classes from method signatures
        for method in target.methods:
            # Check return type
            if method.return_type:
                type_name = self._extract_type_name(method.return_type)
                if type_name in self._classes:
                    relevant_classes.add(type_name)
            # Check parameters
            for param_type, _ in method.parameters:
                type_name = self._extract_type_name(param_type)
                if type_name in self._classes:
                    relevant_classes.add(type_name)

        # Add classes from constructor parameters (CRITICAL for dependency injection)
        for ctor in target.constructors:
            for param_type, _ in ctor.parameters:
                type_name = self._extract_type_name(param_type)
                if type_name in self._classes:
                    relevant_classes.add(type_name)

        # Build context string
        lines = []
        lines.append("// DISCOVERED API SIGNATURES (use ONLY these)")
        lines.append("// These are extracted from actual source code")
        lines.append("// IMPORTANT: Use the exact import paths shown below!\n")

        # Add import section first
        lines.append("// === REQUIRED IMPORTS ===")
        for class_name in sorted(relevant_classes):
            cls = self._classes.get(class_name)
            if cls and cls.fully_qualified_name:
                lines.append(f"import {cls.fully_qualified_name};")
        lines.append("")

        for class_name in sorted(relevant_classes):
            cls = self._classes.get(class_name)
            if not cls:
                continue

            # Show class with its full package path
            if cls.fully_qualified_name and cls.fully_qualified_name != class_name:
                lines.append(f"// === {class_name} ({cls.fully_qualified_name}) ===")
            else:
                lines.append(f"// === {class_name} ===")

            # Show inheritance
            if cls.base_classes:
                lines.append(f"//     extends {', '.join(cls.base_classes)}")

            # Add methods from dependency stubs for base classes
            if cls.base_classes:
                stub_context = get_stub_context_for_classes(cls.base_classes)
                if stub_context:
                    lines.append("// Inherited from framework/library classes:")
                    lines.append(stub_context)

            # Constructors
            for ctor in cls.constructors:
                params = ", ".join(f"{t} {n}" for t, n in ctor.parameters)
                lines.append(f"Constructor: {class_name}({params})")

            # If no constructors shown, indicate no-arg constructor
            if not cls.constructors:
                lines.append(f"Constructor: {class_name}()  // default no-arg")

            # Methods (including inherited for target class)
            if class_name == target_class:
                all_methods = cls.get_all_methods(self)
            else:
                all_methods = cls.methods

            for method in all_methods:
                params = ", ".join(f"{t} {n}" for t, n in method.parameters)
                ret = method.return_type or "void"
                static = "static " if method.is_static else ""
                lines.append(f"Method: {static}{ret} {method.name}({params})")

            lines.append("")  # Empty line between classes

        return "\n".join(lines)

    def build_full_api_context(self) -> str:
        """Build API context for all classes (for ReAct loop fixes)."""
        if not self._classes:
            return "// No API information available"

        lines = []
        lines.append("// ALL DISCOVERED API SIGNATURES")
        lines.append("// Use ONLY methods that appear here")
        lines.append("// IMPORTANT: Use the exact import paths shown!\n")

        # Add import section
        lines.append("// === REQUIRED IMPORTS ===")
        for class_name in sorted(self._classes.keys()):
            cls = self._classes[class_name]
            if cls.fully_qualified_name:
                lines.append(f"import {cls.fully_qualified_name};")
        lines.append("")

        for class_name in sorted(self._classes.keys()):
            cls = self._classes[class_name]
            # Show class with its full package path
            if cls.fully_qualified_name and cls.fully_qualified_name != class_name:
                lines.append(f"// === {class_name} ({cls.fully_qualified_name}) ===")
            else:
                lines.append(f"// === {class_name} ===")

            if cls.base_classes:
                lines.append(f"//     extends {', '.join(cls.base_classes)}")

            for ctor in cls.constructors:
                params = ", ".join(f"{t} {n}" for t, n in ctor.parameters)
                lines.append(f"Constructor: {class_name}({params})")

            if not cls.constructors:
                lines.append(f"Constructor: {class_name}()")

            for method in cls.methods:
                params = ", ".join(f"{t} {n}" for t, n in method.parameters)
                ret = method.return_type or "void"
                lines.append(f"Method: {ret} {method.name}({params})")

            lines.append("")

        return "\n".join(lines)

    def _extract_type_name(self, type_str: str) -> str:
        """Extract simple type name from a type string (handles generics)."""
        # Remove generics: List<Pet> -> List
        if '<' in type_str:
            type_str = type_str.split('<')[0]
        # Remove array brackets: Pet[] -> Pet
        type_str = type_str.replace('[]', '')
        # Get simple name from FQN
        if '.' in type_str:
            type_str = type_str.split('.')[-1]
        return type_str.strip()

    def get_inheritance_chain(self, class_name: str) -> list[str]:
        """Get the full inheritance chain for a class."""
        chain = []
        current = self.get_class(class_name)

        while current:
            chain.append(current.name)
            if current.base_classes:
                # Follow first parent (single inheritance for simplicity)
                parent_name = current.base_classes[0]
                current = self.get_class(parent_name)
            else:
                break

        return chain

    @classmethod
    def from_ast_classes(cls, project_path: Path, classes: list[ClassContext]) -> 'ProjectContext':
        """
        Build ProjectContext from AST-parsed ClassContext objects.

        This is the bridge from AST Parser output to ProjectContext.
        """
        context = cls(project_path)

        for ast_class in classes:
            # Convert constructors
            constructors = []
            for ctor in ast_class.constructors:
                params = [(p.type_hint or "Object", p.name) for p in ctor.parameters]
                constructors.append(MethodSignature(
                    name=ast_class.name,
                    parameters=params,
                    return_type="",
                    is_static=False
                ))

            # Convert methods
            methods = []
            for method in ast_class.methods:
                params = [(p.type_hint or "Object", p.name) for p in method.parameters]
                methods.append(MethodSignature(
                    name=method.name,
                    parameters=params,
                    return_type=method.return_type or "void",
                    is_static=False  # Could detect from source
                ))

            # Build fully qualified name from file path
            fqn = ast_class.name
            if hasattr(ast_class, 'location') and ast_class.location.file_path:
                # Extract package from path like src/main/java/com/example/Foo.java
                path_str = str(ast_class.location.file_path)
                if 'src/main/java/' in path_str:
                    package_path = path_str.split('src/main/java/')[1]
                    package_path = package_path.rsplit('/', 1)[0]  # Remove filename
                    package = package_path.replace('/', '.')
                    fqn = f"{package}.{ast_class.name}"

            class_info = ClassInfo(
                name=ast_class.name,
                fully_qualified_name=fqn,
                package=fqn.rsplit('.', 1)[0] if '.' in fqn else "",
                base_classes=ast_class.base_classes or [],
                constructors=constructors,
                methods=methods,
                fields=[],  # Could extract if needed
                annotations=[]
            )

            context.add_class(class_info)

        context.mark_built()
        return context
