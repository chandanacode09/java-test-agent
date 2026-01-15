"""
Tree-sitter based Java parser for robust AST extraction.

Replaces regex-based parsing with proper AST parsing that handles:
- Complex generics: Map<String, List<Item>>
- Annotations with nested parens: @RequestParam(defaultValue = "()")
- Constructors with any parameter types
- Inner classes, lambdas, etc.
"""

from pathlib import Path
from typing import Optional
import tree_sitter_java as ts_java
from tree_sitter import Language, Parser

from ..models import (
    ClassContext, FunctionContext, CodeLocation, Parameter
)


class TreeSitterJavaParser:
    """Parse Java files using tree-sitter for robust AST extraction."""

    def __init__(self):
        self.language = Language(ts_java.language())
        self.parser = Parser(self.language)

    def parse_file(self, file_path: Path) -> tuple[list[FunctionContext], list[ClassContext]]:
        """
        Parse a Java file and extract classes and functions.

        Returns:
            Tuple of (functions, classes) where functions are top-level
            and classes contain their methods.
        """
        try:
            content = file_path.read_bytes()
            tree = self.parser.parse(content)
        except Exception as e:
            return [], []

        text = content.decode('utf-8', errors='replace')
        functions = []
        classes = []

        # Walk the tree to find class declarations
        for node in self._walk_tree(tree.root_node):
            if node.type == 'class_declaration':
                class_ctx = self._extract_class(node, file_path, text)
                if class_ctx:
                    classes.append(class_ctx)
            elif node.type == 'interface_declaration':
                class_ctx = self._extract_interface(node, file_path, text)
                if class_ctx:
                    classes.append(class_ctx)

        return functions, classes

    def _walk_tree(self, node):
        """Yield all nodes in the tree."""
        yield node
        for child in node.children:
            yield from self._walk_tree(child)

    def _extract_class(self, node, file_path: Path, text: str) -> Optional[ClassContext]:
        """Extract class information from a class_declaration node."""
        class_name = None
        base_classes = []
        methods = []
        constructors = []
        fields = []

        for child in node.children:
            if child.type == 'identifier':
                class_name = self._get_text(child, text)
            elif child.type == 'superclass':
                # Extract extends clause (handle generics)
                for sc_child in child.children:
                    if sc_child.type in ('type_identifier', 'generic_type'):
                        base_classes.append(self._get_text(sc_child, text))
            elif child.type == 'super_interfaces':
                # Extract implements clause - find type_list and get each type
                for si_child in child.children:
                    if si_child.type == 'type_list':
                        for type_node in si_child.children:
                            if type_node.type in ('type_identifier', 'generic_type'):
                                base_classes.append(self._get_text(type_node, text))
            elif child.type == 'class_body':
                # Extract methods, constructors, fields
                for body_child in child.children:
                    if body_child.type == 'method_declaration':
                        method = self._extract_method(body_child, class_name, file_path, text)
                        if method:
                            methods.append(method)
                    elif body_child.type == 'constructor_declaration':
                        ctor = self._extract_constructor(body_child, class_name, file_path, text)
                        if ctor:
                            constructors.append(ctor)
                    elif body_child.type == 'field_declaration':
                        field_info = self._extract_field(body_child, text)
                        if field_info:
                            fields.append(field_info)

        if not class_name:
            return None

        # If no constructors, add default no-arg constructor
        if not constructors:
            constructors.append(FunctionContext(
                name=class_name,
                location=CodeLocation(file_path=file_path, start_line=node.start_point[0] + 1, end_line=node.start_point[0] + 1),
                parameters=[],
                return_type=None,
                docstring=None,
                is_async=False,
                is_method=False,
                class_name=class_name,
                decorators=[],
                source_code=f"public {class_name}()",
                calls=[],
                imports=[]
            ))

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        return ClassContext(
            name=class_name,
            location=CodeLocation(file_path=file_path, start_line=start_line, end_line=end_line),
            methods=methods,
            constructors=constructors,
            base_classes=base_classes,
            docstring=None,
            decorators=[],
            attributes=fields  # fields stored as attributes
        )

    def _extract_interface(self, node, file_path: Path, text: str) -> Optional[ClassContext]:
        """Extract interface information."""
        interface_name = None
        base_classes = []
        methods = []

        for child in node.children:
            if child.type == 'identifier':
                interface_name = self._get_text(child, text)
            elif child.type == 'extends_interfaces':
                # Find type_list and extract each type (including generics)
                for ei_child in child.children:
                    if ei_child.type == 'type_list':
                        for type_node in ei_child.children:
                            if type_node.type in ('type_identifier', 'generic_type'):
                                base_classes.append(self._get_text(type_node, text))
            elif child.type == 'interface_body':
                for body_child in child.children:
                    if body_child.type == 'method_declaration':
                        method = self._extract_method(body_child, interface_name, file_path, text)
                        if method:
                            methods.append(method)

        if not interface_name:
            return None

        return ClassContext(
            name=interface_name,
            location=CodeLocation(file_path=file_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1),
            methods=methods,
            constructors=[],
            base_classes=base_classes,
            docstring=None,
            decorators=[],
            attributes=[]
        )

    def _extract_method(self, node, class_name: str, file_path: Path, text: str) -> Optional[FunctionContext]:
        """Extract method information from a method_declaration node."""
        method_name = None
        return_type = None
        parameters = []
        is_static = False
        visibility = 'package'

        # Check modifiers
        for child in node.children:
            if child.type == 'modifiers':
                mod_text = self._get_text(child, text)
                if 'static' in mod_text:
                    is_static = True
                if 'public' in mod_text:
                    visibility = 'public'
                elif 'private' in mod_text:
                    visibility = 'private'
                elif 'protected' in mod_text:
                    visibility = 'protected'

        # Skip private methods
        if visibility == 'private':
            return None

        for child in node.children:
            if child.type == 'identifier':
                method_name = self._get_text(child, text)
            elif child.type in ('type_identifier', 'void_type', 'generic_type', 'array_type', 'integral_type', 'floating_point_type', 'boolean_type'):
                return_type = self._get_text(child, text)
            elif child.type == 'formal_parameters':
                parameters = self._extract_parameters(child, text)

        if not method_name:
            return None

        return FunctionContext(
            name=method_name,
            location=CodeLocation(file_path=file_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1),
            parameters=parameters,
            return_type=return_type if return_type != 'void' else None,
            docstring=None,
            is_async=False,
            is_method=True,
            class_name=class_name,
            decorators=[],
            source_code=self._get_text(node, text),
            calls=[],
            imports=[]
        )

    def _extract_constructor(self, node, class_name: str, file_path: Path, text: str) -> Optional[FunctionContext]:
        """Extract constructor information."""
        parameters = []
        visibility = 'package'

        for child in node.children:
            if child.type == 'modifiers':
                mod_text = self._get_text(child, text)
                if 'public' in mod_text:
                    visibility = 'public'
                elif 'private' in mod_text:
                    visibility = 'private'
                elif 'protected' in mod_text:
                    visibility = 'protected'
            elif child.type == 'formal_parameters':
                parameters = self._extract_parameters(child, text)

        # Skip private constructors
        if visibility == 'private':
            return None

        return FunctionContext(
            name=class_name,
            location=CodeLocation(file_path=file_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1),
            parameters=parameters,
            return_type=None,
            docstring=None,
            is_async=False,
            is_method=False,
            class_name=class_name,
            decorators=[],
            source_code=self._get_text(node, text),
            calls=[],
            imports=[]
        )

    def _extract_parameters(self, node, text: str) -> list[Parameter]:
        """Extract parameters from formal_parameters node."""
        parameters = []

        for child in node.children:
            if child.type == 'formal_parameter':
                param_type = None
                param_name = None

                for pc in child.children:
                    if pc.type in ('type_identifier', 'generic_type', 'array_type', 'integral_type', 'floating_point_type', 'boolean_type'):
                        param_type = self._get_text(pc, text)
                    elif pc.type == 'identifier':
                        param_name = self._get_text(pc, text)

                if param_name:
                    parameters.append(Parameter(
                        name=param_name,
                        type_hint=param_type,
                        default_value=None,
                        is_optional=False
                    ))
            elif child.type == 'spread_parameter':
                # Handle varargs like String... args
                param_type = None
                param_name = None

                for pc in child.children:
                    if pc.type in ('type_identifier', 'generic_type', 'array_type'):
                        param_type = self._get_text(pc, text) + "..."
                    elif pc.type == 'identifier':
                        param_name = self._get_text(pc, text)
                    elif pc.type == 'variable_declarator':
                        for vd in pc.children:
                            if vd.type == 'identifier':
                                param_name = self._get_text(vd, text)

                if param_name:
                    parameters.append(Parameter(
                        name=param_name,
                        type_hint=param_type,
                        default_value=None,
                        is_optional=False
                    ))

        return parameters

    def _extract_field(self, node, text: str) -> Optional[tuple[str, str]]:
        """Extract field name and type."""
        field_type = None
        field_name = None

        for child in node.children:
            if child.type in ('type_identifier', 'generic_type', 'array_type', 'integral_type'):
                field_type = self._get_text(child, text)
            elif child.type == 'variable_declarator':
                for vc in child.children:
                    if vc.type == 'identifier':
                        field_name = self._get_text(vc, text)
                        break

        if field_name and field_type:
            return (field_type, field_name)
        return None

    def _get_text(self, node, text: str) -> str:
        """Get the text content of a node."""
        return text[node.start_byte:node.end_byte]
