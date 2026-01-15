"""
AST Parser using tree-sitter for extracting code structure.

This provides deterministic, reliable extraction of:
- Functions and their signatures
- Classes and their methods
- Dependencies and imports
"""

from pathlib import Path
from dataclasses import dataclass
import re

from ..models import (
    FunctionContext, ClassContext, Parameter, CodeLocation, Language, ReturnSemantics
)


# Flask context indicators
FLASK_CONTEXT_INDICATORS = {
    'current_app', 'g', 'request', 'session', '_request_ctx_stack',
    '_app_ctx_stack', 'flask.current_app', 'flask.g', 'flask.request'
}

# Framework detection patterns
FRAMEWORK_PATTERNS = {
    'flask': {'flask', 'Flask', 'current_app', 'Blueprint', 'make_response', 'url_for'},
    'django': {'django', 'HttpResponse', 'HttpRequest', 'View', 'Model'},
    'sqlalchemy': {'sqlalchemy', 'Session', 'create_engine', 'Column', 'relationship'},
    'requests': {'requests.get', 'requests.post', 'requests.Session'},
    'werkzeug': {'werkzeug', 'Request', 'Response', 'MultiDict'},
}


# Try to import tree-sitter, fall back to regex-based parsing
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language as TSLanguage, Parser
    TREE_SITTER_PYTHON_AVAILABLE = True
except ImportError:
    TREE_SITTER_PYTHON_AVAILABLE = False

# Try to import tree-sitter-java for robust Java parsing
try:
    from .tree_sitter_parser import TreeSitterJavaParser
    TREE_SITTER_JAVA_AVAILABLE = True
except ImportError:
    TREE_SITTER_JAVA_AVAILABLE = False


class ASTParser:
    """
    Parse source code to extract function and class information.

    Uses tree-sitter when available, falls back to regex-based parsing.
    """

    def __init__(self, language: Language = Language.PYTHON):
        self.language = language
        self._parser = None
        self._java_parser = None

        if TREE_SITTER_PYTHON_AVAILABLE and language == Language.PYTHON:
            self._init_tree_sitter()

        if TREE_SITTER_JAVA_AVAILABLE and language == Language.JAVA:
            self._java_parser = TreeSitterJavaParser()

    def _init_tree_sitter(self):
        """Initialize tree-sitter parser."""
        try:
            self._parser = Parser(TSLanguage(tspython.language()))
        except Exception:
            self._parser = None

    def parse_file(self, file_path: Path) -> tuple[list[FunctionContext], list[ClassContext]]:
        """
        Parse a file and extract all functions and classes.

        Returns:
            Tuple of (functions, classes)
        """
        if self.language == Language.JAVA:
            # Use tree-sitter for robust Java parsing, fallback to regex
            if self._java_parser:
                return self._java_parser.parse_file(file_path)
            else:
                content = file_path.read_text()
                return self._parse_java_regex(content, file_path)

        content = file_path.read_text()
        if self._parser and self.language == Language.PYTHON:
            return self._parse_python_tree_sitter(content, file_path)
        else:
            return self._parse_python_regex(content, file_path)

    def parse_content(
        self,
        content: str,
        file_path: Path
    ) -> tuple[list[FunctionContext], list[ClassContext]]:
        """Parse content string directly."""
        if self.language == Language.JAVA:
            return self._parse_java_regex(content, file_path)
        elif self._parser and self.language == Language.PYTHON:
            return self._parse_python_tree_sitter(content, file_path)
        else:
            return self._parse_python_regex(content, file_path)

    def extract_function_at_line(
        self,
        file_path: Path,
        line_number: int
    ) -> FunctionContext | None:
        """Extract the function that contains or starts at the given line."""
        functions, classes = self.parse_file(file_path)

        # Check standalone functions
        for func in functions:
            if func.location.start_line <= line_number <= func.location.end_line:
                return func

        # Check class methods
        for cls in classes:
            for method in cls.methods:
                if method.location.start_line <= line_number <= method.location.end_line:
                    return method

        return None

    def extract_functions_in_range(
        self,
        file_path: Path,
        start_line: int,
        end_line: int
    ) -> list[FunctionContext]:
        """Extract all functions that overlap with the given line range."""
        functions, classes = self.parse_file(file_path)
        result = []

        # Check standalone functions
        for func in functions:
            if self._ranges_overlap(
                func.location.start_line, func.location.end_line,
                start_line, end_line
            ):
                result.append(func)

        # Check class methods
        for cls in classes:
            for method in cls.methods:
                if self._ranges_overlap(
                    method.location.start_line, method.location.end_line,
                    start_line, end_line
                ):
                    result.append(method)

        return result

    def _ranges_overlap(self, s1: int, e1: int, s2: int, e2: int) -> bool:
        """Check if two line ranges overlap."""
        return s1 <= e2 and s2 <= e1

    def _parse_python_tree_sitter(
        self,
        content: str,
        file_path: Path
    ) -> tuple[list[FunctionContext], list[ClassContext]]:
        """Parse Python using tree-sitter for accurate AST."""
        tree = self._parser.parse(bytes(content, 'utf-8'))
        root = tree.root_node

        functions = []
        classes = []
        lines = content.split('\n')

        def get_node_text(node) -> str:
            return content[node.start_byte:node.end_byte]

        def extract_docstring(body_node) -> str | None:
            """Extract docstring from function/class body."""
            if body_node and body_node.child_count > 0:
                first_child = body_node.children[0]
                if first_child.type == 'expression_statement':
                    expr = first_child.children[0] if first_child.children else None
                    if expr and expr.type == 'string':
                        doc = get_node_text(expr)
                        # Clean up the docstring
                        return doc.strip('"""').strip("'''").strip()
            return None

        def extract_parameters(params_node) -> list[Parameter]:
            """Extract parameters from function definition."""
            params = []
            if not params_node:
                return params

            for child in params_node.children:
                if child.type in ('identifier', 'typed_parameter', 'default_parameter',
                                   'typed_default_parameter'):
                    param = self._extract_parameter_from_node(child, content)
                    if param and param.name not in ('self', 'cls'):
                        params.append(param)

            return params

        def extract_return_type(node) -> str | None:
            """Extract return type annotation."""
            for child in node.children:
                if child.type == 'type':
                    return get_node_text(child)
            return None

        def extract_decorators(node) -> list[str]:
            """Extract decorator names."""
            decorators = []
            # Look for decorator nodes before the function/class
            prev = node.prev_sibling
            while prev and prev.type == 'decorator':
                dec_text = get_node_text(prev)
                decorators.insert(0, dec_text)
                prev = prev.prev_sibling
            return decorators

        def process_function(node, class_name: str | None = None) -> FunctionContext:
            """Process a function definition node."""
            name = None
            params_node = None
            body_node = None

            for child in node.children:
                if child.type == 'identifier':
                    name = get_node_text(child)
                elif child.type == 'parameters':
                    params_node = child
                elif child.type == 'block':
                    body_node = child

            # Get source code for the function
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            source_lines = lines[start_line:end_line + 1]

            # Check if async
            is_async = any(c.type == 'async' for c in node.children) or \
                       node.type == 'async_function_definition'

            source_code = '\n'.join(source_lines)
            calls = self._extract_function_calls(body_node, content) if body_node else []

            func = FunctionContext(
                name=name or "unknown",
                location=CodeLocation(
                    file_path=file_path,
                    start_line=start_line + 1,  # 1-indexed
                    end_line=end_line + 1,
                    start_col=node.start_point[1],
                    end_col=node.end_point[1]
                ),
                parameters=extract_parameters(params_node),
                return_type=extract_return_type(node),
                docstring=extract_docstring(body_node),
                is_async=is_async,
                is_method=class_name is not None,
                class_name=class_name,
                decorators=extract_decorators(node),
                source_code=source_code,
                calls=calls,
                imports=[]
            )

            # Enrich with semantic information
            return self._enrich_function_semantics(func)

        def process_class(node) -> ClassContext:
            """Process a class definition node."""
            name = None
            bases = []
            body_node = None
            methods = []
            attributes = []

            for child in node.children:
                if child.type == 'identifier':
                    name = get_node_text(child)
                elif child.type == 'argument_list':
                    # Base classes
                    for arg in child.children:
                        if arg.type == 'identifier':
                            bases.append(get_node_text(arg))
                elif child.type == 'block':
                    body_node = child

            # Extract methods from class body
            if body_node:
                for child in body_node.children:
                    if child.type in ('function_definition', 'async_function_definition'):
                        method = process_function(child, class_name=name)
                        methods.append(method)
                    elif child.type == 'expression_statement':
                        # Could be an attribute assignment
                        pass

            return ClassContext(
                name=name or "unknown",
                location=CodeLocation(
                    file_path=file_path,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_col=node.start_point[1],
                    end_col=node.end_point[1]
                ),
                methods=methods,
                base_classes=bases,
                docstring=extract_docstring(body_node),
                decorators=extract_decorators(node),
                attributes=attributes
            )

        # Walk the AST
        def walk(node):
            if node.type == 'function_definition':
                functions.append(process_function(node))
            elif node.type == 'async_function_definition':
                functions.append(process_function(node))
            elif node.type == 'class_definition':
                classes.append(process_class(node))
            else:
                for child in node.children:
                    walk(child)

        walk(root)
        return functions, classes

    def _extract_parameter_from_node(self, node, content: str) -> Parameter | None:
        """Extract parameter info from a tree-sitter node."""
        def get_text(n):
            return content[n.start_byte:n.end_byte]

        if node.type == 'identifier':
            return Parameter(name=get_text(node))

        elif node.type == 'typed_parameter':
            name = None
            type_hint = None
            for child in node.children:
                if child.type == 'identifier':
                    name = get_text(child)
                elif child.type == 'type':
                    type_hint = get_text(child)
            if name:
                return Parameter(name=name, type_hint=type_hint)

        elif node.type == 'default_parameter':
            name = None
            default = None
            for child in node.children:
                if child.type == 'identifier':
                    name = get_text(child)
                elif child.type not in ('=',):
                    default = get_text(child)
            if name:
                return Parameter(name=name, default_value=default, is_optional=True)

        elif node.type == 'typed_default_parameter':
            name = None
            type_hint = None
            default = None
            for child in node.children:
                if child.type == 'identifier':
                    name = get_text(child)
                elif child.type == 'type':
                    type_hint = get_text(child)
            if name:
                return Parameter(
                    name=name,
                    type_hint=type_hint,
                    default_value=default,
                    is_optional=True
                )

        return None

    def _extract_function_calls(self, body_node, content: str) -> list[str]:
        """Extract function calls from a function body."""
        calls = []

        def get_text(n):
            return content[n.start_byte:n.end_byte]

        def walk(node):
            if node.type == 'call':
                # Get the function being called
                if node.children:
                    func = node.children[0]
                    call_name = get_text(func)
                    if call_name not in calls:
                        calls.append(call_name)
            for child in node.children:
                walk(child)

        if body_node:
            walk(body_node)

        return calls

    # =========================================================================
    # Semantic Analysis Methods
    # =========================================================================

    def _detect_is_generator(self, source_code: str) -> bool:
        """Detect if function uses yield (is a generator)."""
        # Match 'yield' or 'yield from' but not inside strings or comments
        yield_pattern = re.compile(r'(?<!["\'])\byield\b(?:\s+from)?(?!["\'])')
        return bool(yield_pattern.search(source_code))

    def _analyze_return_semantics(
        self,
        return_type: str | None,
        source_code: str,
        calls: list[str]
    ) -> ReturnSemantics:
        """Determine the semantic meaning of the return type."""
        # Check for generator first (highest priority)
        if self._detect_is_generator(source_code):
            return ReturnSemantics.GENERATOR

        if not return_type:
            # Check if function has side effects (modifies args, calls external)
            if self._detect_mutates_args(source_code):
                return ReturnSemantics.NONE_SIDEEFFECT
            return ReturnSemantics.VALUE

        return_type_lower = return_type.lower()

        # Check for iterator/generator types
        iterator_patterns = ['iterator', 'generator', 'iterable', 'iter[']
        if any(p in return_type_lower for p in iterator_patterns):
            return ReturnSemantics.ITERATOR

        # Check for context manager
        if 'contextmanager' in return_type_lower or '@contextmanager' in source_code:
            return ReturnSemantics.CONTEXT_MANAGER

        # Check for None return with side effects
        if return_type_lower == 'none':
            return ReturnSemantics.NONE_SIDEEFFECT

        return ReturnSemantics.VALUE

    def _detect_requires_context(self, source_code: str, calls: list[str]) -> list[str]:
        """Detect what context/fixtures the function requires."""
        contexts = []

        # Check for Flask context requirements
        all_refs = set(calls) | set(re.findall(r'\b\w+\b', source_code))
        if FLASK_CONTEXT_INDICATORS & all_refs:
            contexts.append('flask_app')

        # Check for request context
        if 'request' in all_refs and 'flask_app' in contexts:
            contexts.append('request_context')

        # Check for database session
        db_indicators = {'session', 'db', 'database', 'Session', 'engine'}
        if db_indicators & all_refs:
            contexts.append('db_session')

        return contexts

    def _detect_mutates_args(self, source_code: str) -> bool:
        """Detect if function mutates its arguments."""
        # Look for patterns like: arg.attr = value, arg[key] = value
        mutation_patterns = [
            r'\b\w+\.\w+\s*=',           # obj.attr =
            r'\b\w+\[.+\]\s*=',          # obj[key] =
            r'\b\w+\.__\w+__\s*=',       # obj.__class__ =
            r'\.append\(',                # list.append()
            r'\.extend\(',                # list.extend()
            r'\.update\(',                # dict.update()
            r'\.pop\(',                   # dict/list.pop()
            r'\.clear\(',                 # collection.clear()
        ]

        for pattern in mutation_patterns:
            if re.search(pattern, source_code):
                return True
        return False

    def _detect_frameworks(self, source_code: str, calls: list[str]) -> list[str]:
        """Detect which frameworks the function uses."""
        frameworks = []
        all_refs = set(calls) | set(re.findall(r'\b\w+\b', source_code))

        for framework, indicators in FRAMEWORK_PATTERNS.items():
            if indicators & all_refs:
                frameworks.append(framework)

        return frameworks

    def _enrich_function_semantics(self, func: FunctionContext) -> FunctionContext:
        """Add semantic information to a FunctionContext."""
        func.is_generator = self._detect_is_generator(func.source_code)
        func.return_semantics = self._analyze_return_semantics(
            func.return_type, func.source_code, func.calls
        )
        func.requires_context = self._detect_requires_context(func.source_code, func.calls)
        func.mutates_args = self._detect_mutates_args(func.source_code)
        func.framework_hints = self._detect_frameworks(func.source_code, func.calls)

        return func

    def _parse_python_regex(
        self,
        content: str,
        file_path: Path
    ) -> tuple[list[FunctionContext], list[ClassContext]]:
        """
        Fallback regex-based parsing when tree-sitter is not available.

        Less accurate but works without external dependencies.
        """
        functions = []
        classes = []
        lines = content.split('\n')

        # Patterns - use [ \t]* instead of \s* to avoid matching newlines
        func_pattern = re.compile(
            r'^([ \t]*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?\s*:',
            re.MULTILINE
        )
        class_pattern = re.compile(
            r'^([ \t]*)class\s+(\w+)(?:\s*\(([^)]*)\))?\s*:',
            re.MULTILINE
        )

        # Find all functions
        for match in func_pattern.finditer(content):
            indent = len(match.group(1))
            is_async = match.group(2) is not None
            name = match.group(3)
            params_str = match.group(4)
            return_type = match.group(5)

            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1, indent)

            # Parse parameters
            params = self._parse_params_regex(params_str)

            # Get source code
            source_lines = lines[start_line - 1:end_line]

            source_code = '\n'.join(source_lines)
            func = FunctionContext(
                name=name,
                location=CodeLocation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                ),
                parameters=params,
                return_type=return_type.strip() if return_type else None,
                docstring=self._extract_docstring_regex(lines, start_line),
                is_async=is_async,
                is_method=False,  # Will be corrected if inside a class
                class_name=None,
                decorators=[],
                source_code=source_code
            )

            # Enrich with semantic information
            func = self._enrich_function_semantics(func)
            functions.append(func)

        # First, collect all class info with their line ranges
        class_info = []
        for match in class_pattern.finditer(content):
            indent = len(match.group(1))
            name = match.group(2)
            bases_str = match.group(3)

            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1, indent)

            bases = []
            if bases_str:
                bases = [b.strip() for b in bases_str.split(',')]

            class_info.append({
                'name': name,
                'start_line': start_line,
                'end_line': end_line,
                'bases': bases,
                'indent': indent
            })

        # Now assign functions to classes based on line ranges
        methods_to_remove = set()
        for cls_data in class_info:
            class_methods = []
            for i, func in enumerate(functions):
                # Check if function is inside this class (proper indentation check)
                if cls_data['start_line'] < func.location.start_line <= cls_data['end_line']:
                    # Create a copy with updated class info
                    func.is_method = True
                    func.class_name = cls_data['name']
                    class_methods.append(func)
                    methods_to_remove.add(i)

            classes.append(ClassContext(
                name=cls_data['name'],
                location=CodeLocation(
                    file_path=file_path,
                    start_line=cls_data['start_line'],
                    end_line=cls_data['end_line']
                ),
                methods=class_methods,
                base_classes=cls_data['bases'],
                docstring=self._extract_docstring_regex(lines, cls_data['start_line']),
                decorators=[],
                attributes=[]
            ))

        # Remove methods from top-level functions
        functions = [f for i, f in enumerate(functions) if i not in methods_to_remove]

        return functions, classes

    def _find_block_end(self, lines: list[str], start_idx: int, base_indent: int) -> int:
        """Find the end of a Python block based on indentation."""
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i
        return len(lines)

    def _parse_params_regex(self, params_str: str) -> list[Parameter]:
        """Parse parameters from a parameter string."""
        params = []
        if not params_str.strip():
            return params

        # Simple parsing - doesn't handle all edge cases
        for param in params_str.split(','):
            param = param.strip()
            if not param or param in ('self', 'cls'):
                continue

            # Check for type hint
            if ':' in param:
                parts = param.split(':')
                name = parts[0].strip()
                rest = ':'.join(parts[1:])
                # Check for default
                if '=' in rest:
                    type_hint, default = rest.split('=', 1)
                    params.append(Parameter(
                        name=name,
                        type_hint=type_hint.strip(),
                        default_value=default.strip(),
                        is_optional=True
                    ))
                else:
                    params.append(Parameter(name=name, type_hint=rest.strip()))
            elif '=' in param:
                name, default = param.split('=', 1)
                params.append(Parameter(
                    name=name.strip(),
                    default_value=default.strip(),
                    is_optional=True
                ))
            else:
                params.append(Parameter(name=param))

        return params

    def _extract_docstring_regex(self, lines: list[str], start_line: int) -> str | None:
        """Extract docstring from lines after function/class definition."""
        # Look at the line(s) after the definition
        for i in range(start_line, min(start_line + 5, len(lines))):
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Find the end of docstring
                quote = line[:3]
                if line.count(quote) >= 2:
                    # Single line docstring
                    return line[3:-3].strip()
                else:
                    # Multi-line docstring
                    doc_lines = [line[3:]]
                    for j in range(i + 1, len(lines)):
                        doc_line = lines[j]
                        if quote in doc_line:
                            doc_lines.append(doc_line[:doc_line.index(quote)])
                            break
                        doc_lines.append(doc_line)
                    return '\n'.join(doc_lines).strip()
        return None

    # =========================================================================
    # Java Parsing (regex-based)
    # =========================================================================

    def _parse_java_regex(
        self,
        content: str,
        file_path: Path
    ) -> tuple[list[FunctionContext], list[ClassContext]]:
        """
        Parse Java source code using regex patterns.

        Extracts classes and their methods from Java files.
        """
        functions = []
        classes = []
        lines = content.split('\n')

        # Java class/interface pattern - matches class, abstract class, and interface declarations
        # Handles generic type parameters like <T, ID extends Comparable<ID>>
        class_pattern = re.compile(
            r'^[ \t]*(public|private|protected)?\s*'  # Optional visibility
            r'(abstract|final)?\s*'                    # Optional abstract/final
            r'(class|interface)\s+'                    # class or interface keyword
            r'(\w+)'                                   # Class/interface name
            r'(?:<[^>]+(?:<[^>]+>)?[^>]*>)?'          # Optional generic params <T, K extends X<Y>>
            r'(?:\s+extends\s+([\w<>,\s]+?))?'        # Optional extends (with generics)
            r'(?:\s+implements\s+([\w<>,\s]+?))?\s*\{',  # Optional implements (with generics)
            re.MULTILINE
        )

        # Java method pattern - matches method declarations
        # Handles regular methods, static methods, default interface methods
        method_pattern = re.compile(
            r'^[ \t]*(public|private|protected|default)\s+'   # Required visibility or default
            r'(static\s+)?'                                   # Optional static
            r'(final\s+)?'                                    # Optional final
            r'(?:<[\w\s,<>]+>\s+)?'                           # Optional generic type params
            r'(?:@\w+(?:\s*\([^)]*\))?\s+)?'                  # Optional annotation before return type (@ResponseBody)
            r'([\w<>\[\],]+)\s+'                              # Return type (no spaces - single word/generic)
            r'(\w+)\s*\('                                     # Method name
            r'((?:[^()]*|\([^)]*\))*)\)',                     # Parameters (handles nested parens in annotations)
            re.MULTILINE
        )

        # Additional pattern for interface method signatures (no visibility, no body)
        interface_method_pattern = re.compile(
            r'^[ \t]*(?:<[\w\s,<>]+>\s+)?'                    # Optional generic type params
            r'([\w<>\[\],]+)\s+'                              # Return type
            r'(\w+)\s*\('                                     # Method name
            r'((?:[^()]*|\([^)]*\))*)\)\s*;',                 # Parameters ending with ; (handles nested parens)
            re.MULTILINE
        )

        # Constructor pattern - constructors don't have return types
        # Matches: public ClassName(params) or just ClassName(params)
        # Note: Use simple [^)]* for params to avoid catastrophic backtracking
        constructor_pattern = re.compile(
            r'^[ \t]*(public|private|protected)?\s*'   # Optional visibility
            r'(\w+)\s*\('                              # Constructor name (same as class name)
            r'([^)]*)\)\s*'                            # Parameters (simple - no nested parens)
            r'(?:throws\s+[\w,\s]+\s*)?\{',            # Optional throws clause, then opening brace
            re.MULTILINE
        )

        # Find all classes and interfaces
        for match in class_pattern.finditer(content):
            class_type = match.group(3)  # 'class' or 'interface'
            class_name = match.group(4)
            extends = match.group(5)
            implements = match.group(6)

            start_line = content[:match.start()].count('\n') + 1
            class_start_pos = match.end()

            # Find the matching closing brace for this class
            end_line = self._find_java_block_end(content, class_start_pos, start_line)

            base_classes = []
            if extends:
                # Clean up extends - remove generic params for base class list
                extends_clean = re.sub(r'<[^>]+>', '', extends).strip()
                base_classes.append(extends_clean)
            if implements:
                # Clean up implements - remove generic params
                for impl in implements.split(','):
                    impl_clean = re.sub(r'<[^>]+>', '', impl).strip()
                    if impl_clean:
                        base_classes.append(impl_clean)

            # Extract class body
            class_body_start = match.end()
            class_body_end = self._find_closing_brace_pos(content, class_start_pos)
            class_body = content[class_body_start:class_body_end]

            # Find methods and constructors within this class
            methods = []
            constructors = []
            is_interface = (class_type == 'interface')

            # For interfaces, also parse method signatures (no body)
            if is_interface:
                for sig_match in interface_method_pattern.finditer(class_body):
                    return_type = sig_match.group(1).strip()
                    method_name = sig_match.group(2)
                    params_str = sig_match.group(3)

                    # Skip if method name is a Java keyword
                    if method_name in ('if', 'for', 'while', 'switch', 'try', 'catch', 'return'):
                        continue

                    method_start_line = start_line + class_body[:sig_match.start()].count('\n')

                    params = self._parse_java_params(params_str)
                    docstring = self._extract_javadoc(class_body, sig_match.start())

                    func_context = FunctionContext(
                        name=method_name,
                        location=CodeLocation(
                            file_path=file_path,
                            start_line=method_start_line,
                            end_line=method_start_line
                        ),
                        parameters=params,
                        return_type=return_type if return_type != 'void' else None,
                        docstring=docstring,
                        is_async=False,
                        is_method=True,
                        class_name=class_name,
                        decorators=[],
                        source_code=sig_match.group(0),
                        calls=[],
                        imports=[]
                    )
                    methods.append(func_context)
                    functions.append(func_context)

            # Find constructors using constructor pattern (they don't have return types)
            for ctor_match in constructor_pattern.finditer(class_body):
                visibility = ctor_match.group(1) or 'package'
                ctor_name = ctor_match.group(2)
                params_str = ctor_match.group(3)

                # Only match if constructor name equals class name
                if ctor_name != class_name:
                    continue

                # Skip private constructors
                if visibility == 'private':
                    continue

                ctor_start_line = start_line + class_body[:ctor_match.start()].count('\n')
                ctor_end_line = self._find_java_method_end(
                    class_body, ctor_match.end() - 1, ctor_start_line  # -1 because pattern includes {
                )

                # Parse parameters
                params = self._parse_java_params(params_str)

                # Get source code for the constructor
                ctor_body_start = ctor_match.start()
                ctor_body_end = self._find_closing_brace_pos(
                    class_body, ctor_match.end() - 1  # -1 because pattern includes {
                )
                if ctor_body_end > ctor_body_start:
                    source_code = class_body[ctor_body_start:ctor_body_end + 1]
                else:
                    source_code = ctor_match.group(0)

                # Extract Javadoc if present
                docstring = self._extract_javadoc(class_body, ctor_match.start())

                ctor_context = FunctionContext(
                    name=ctor_name,
                    location=CodeLocation(
                        file_path=file_path,
                        start_line=ctor_start_line,
                        end_line=ctor_end_line
                    ),
                    parameters=params,
                    return_type=None,  # Constructors have no return type
                    docstring=docstring,
                    is_async=False,
                    is_method=False,  # Constructors are not methods
                    class_name=class_name,
                    decorators=[],
                    source_code=source_code,
                    calls=[],
                    imports=[]
                )
                constructors.append(ctor_context)

            for method_match in method_pattern.finditer(class_body):
                visibility = method_match.group(1) or 'package'
                is_static = method_match.group(2) is not None
                return_type = method_match.group(4).strip()
                method_name = method_match.group(5)
                params_str = method_match.group(6)

                # Skip private methods - they can't be tested directly
                if visibility == 'private':
                    continue

                # Skip if method name is a Java keyword (false positive)
                if method_name in ('if', 'for', 'while', 'switch', 'try', 'catch', 'return'):
                    continue

                # Skip abstract methods, interface methods (no body)
                method_end_pos = method_match.end()
                remaining = class_body[method_end_pos:method_end_pos + 50].strip()
                if remaining.startswith(';'):
                    continue  # Abstract method - already handled for interfaces

                method_start_line = start_line + class_body[:method_match.start()].count('\n')
                method_end_line = self._find_java_method_end(
                    class_body, method_match.end(), method_start_line
                )

                # Parse parameters
                params = self._parse_java_params(params_str)

                # Get source code for the method
                method_body_start = method_match.start()
                method_body_end = self._find_closing_brace_pos(
                    class_body, method_match.end()
                )
                if method_body_end > method_body_start:
                    source_code = class_body[method_body_start:method_body_end + 1]
                else:
                    source_code = method_match.group(0)

                # Extract Javadoc if present
                docstring = self._extract_javadoc(class_body, method_match.start())

                func_context = FunctionContext(
                    name=method_name,
                    location=CodeLocation(
                        file_path=file_path,
                        start_line=method_start_line,
                        end_line=method_end_line
                    ),
                    parameters=params,
                    return_type=return_type if return_type != 'void' else None,
                    docstring=docstring,
                    is_async=False,
                    is_method=True,
                    class_name=class_name,
                    decorators=[],  # Java uses annotations, could add later
                    source_code=source_code,
                    calls=[],
                    imports=[]
                )

                methods.append(func_context)
                # Also add methods to top-level functions list for easier access
                functions.append(func_context)

            # If no explicit constructors found, add default no-arg constructor
            if not constructors:
                constructors.append(FunctionContext(
                    name=class_name,
                    location=CodeLocation(file_path=file_path, start_line=start_line, end_line=start_line),
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

            classes.append(ClassContext(
                name=class_name,
                location=CodeLocation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                ),
                methods=methods,
                base_classes=base_classes,
                docstring=self._extract_javadoc(content, match.start()),
                decorators=[],
                attributes=[],
                constructors=constructors
            ))

        return functions, classes

    def _find_java_block_end(self, content: str, start_pos: int, start_line: int) -> int:
        """Find the end line of a Java block (class or method)."""
        end_pos = self._find_closing_brace_pos(content, start_pos)
        return start_line + content[start_pos:end_pos].count('\n')

    def _find_closing_brace_pos(self, content: str, start_pos: int) -> int:
        """Find the position of the closing brace matching the block start."""
        brace_count = 1
        pos = start_pos

        while pos < len(content) and brace_count > 0:
            char = content[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            pos += 1

        return pos - 1 if brace_count == 0 else len(content)

    def _find_java_method_end(self, content: str, start_pos: int, start_line: int) -> int:
        """Find the end line of a Java method."""
        # Find the opening brace
        brace_pos = content.find('{', start_pos)
        if brace_pos == -1:
            return start_line + 1

        end_pos = self._find_closing_brace_pos(content, brace_pos + 1)
        return start_line + content[start_pos:end_pos].count('\n')

    def _parse_java_params(self, params_str: str) -> list[Parameter]:
        """Parse Java method parameters."""
        params = []
        if not params_str.strip():
            return params

        # Split by comma, but be careful with generics like Map<K, V>
        param_list = []
        current = ""
        angle_depth = 0

        for char in params_str:
            if char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth -= 1
            elif char == ',' and angle_depth == 0:
                param_list.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            param_list.append(current.strip())

        for param in param_list:
            param = param.strip()
            if not param:
                continue

            # Handle annotations like @NotNull String name or @RequestParam(defaultValue = "1") int page
            # Remove annotations with optional parameters: @Name or @Name(...) or @Name({...})
            param = re.sub(r'@\w+(?:\s*\([^)]*\))?\s*', '', param)

            # Handle final keyword
            param = re.sub(r'\bfinal\s+', '', param)

            # Split into type and name
            parts = param.rsplit(None, 1)
            if len(parts) == 2:
                type_hint, name = parts
                # Handle varargs
                if type_hint.endswith('...'):
                    type_hint = type_hint[:-3] + '[]'
                params.append(Parameter(
                    name=name,
                    type_hint=type_hint.strip()
                ))
            elif len(parts) == 1:
                # Just a name or malformed
                params.append(Parameter(name=parts[0]))

        return params

    def _extract_javadoc(self, content: str, method_start: int) -> str | None:
        """Extract Javadoc comment before a method or class."""
        # Look backwards from method_start for /** ... */
        search_area = content[max(0, method_start - 1000):method_start]

        # Find the last Javadoc comment
        javadoc_pattern = re.compile(r'/\*\*(.*?)\*/', re.DOTALL)
        matches = list(javadoc_pattern.finditer(search_area))

        if matches:
            last_match = matches[-1]
            # Check if there's only whitespace between javadoc and method
            between = search_area[last_match.end():]
            if between.strip() == '' or re.match(r'^[\s@\w()]*$', between):
                doc = last_match.group(1)
                # Clean up the javadoc
                lines = doc.split('\n')
                cleaned = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('*'):
                        line = line[1:].strip()
                    if line and not line.startswith('@'):
                        cleaned.append(line)
                return ' '.join(cleaned) if cleaned else None

        return None
