"""
Code Post-Processor - Fixes common LLM-generated invalid Java patterns.

This module runs AFTER the LLM generates code but BEFORE compilation,
automatically fixing known hallucination patterns that Grok and other
LLMs commonly produce.

Known patterns fixed:
1. Inline semicolons: instance.method(Type x = new Type(); x.foo();)
2. Variable declarations in method calls: instance.method(Type x = new Type())
3. Class names as method names: instance.Owner(...) -> instance.addPet(...)
4. Python booleans: True/False -> true/false
5. Python None: None -> null
"""

import re
from dataclasses import dataclass


@dataclass
class PostProcessResult:
    """Result of post-processing with changes made."""
    original_code: str
    fixed_code: str
    changes_made: list[str]
    was_modified: bool


class JavaCodePostProcessor:
    """
    Post-processes LLM-generated Java code to fix common syntax errors.

    Usage:
        processor = JavaCodePostProcessor()
        result = processor.process(test_code)
        if result.was_modified:
            print(f"Fixed {len(result.changes_made)} issues")
        use_code = result.fixed_code
    """

    # Python to Java literal mappings
    PYTHON_LITERALS = {
        r'\bTrue\b': 'true',
        r'\bFalse\b': 'false',
        r'\bNone\b': 'null',
    }

    def process(self, code: str) -> PostProcessResult:
        """
        Process Java code and fix common LLM patterns.

        Args:
            code: The Java test code to process

        Returns:
            PostProcessResult with fixed code and list of changes
        """
        original = code
        changes = []

        # Fix 0: Malformed import lines (must run first)
        code, import_changes = self._fix_malformed_imports(code)
        changes.extend(import_changes)

        # Fix 1: Python literals
        code, python_changes = self._fix_python_literals(code)
        changes.extend(python_changes)

        # Fix 2: Inline statements in method calls
        code, inline_changes = self._fix_inline_statements(code)
        changes.extend(inline_changes)

        # Fix 3: Class names used as methods
        code, method_changes = self._fix_class_as_method(code)
        changes.extend(method_changes)

        # Fix 4: Empty collection literals
        code, collection_changes = self._fix_empty_collections(code)
        changes.extend(collection_changes)

        # Fix 5: String-quoted expected values
        code, string_changes = self._fix_string_assertions(code)
        changes.extend(string_changes)

        # Fix 6: Unquoted path-like assertions (vets / vetList -> "vets/vetList")
        code, path_changes = self._fix_unquoted_paths(code)
        changes.extend(path_changes)

        # Fix 7: Missing semicolons after object instantiation
        code, semicolon_changes = self._fix_missing_semicolons(code)
        changes.extend(semicolon_changes)

        # Fix 8: Instance vs local variable mismatch
        code, instance_changes = self._fix_instance_variable_mismatch(code)
        changes.extend(instance_changes)

        # Fix 9: Method calls on wrong object (parameter instead of test instance)
        code, wrong_object_changes = self._fix_method_on_wrong_object(code)
        changes.extend(wrong_object_changes)

        # Fix 10: Mock variable name mismatch (LLM declares @Mock types but uses petTypeRepository)
        code, mock_var_changes = self._fix_mock_variable_mismatch(code)
        changes.extend(mock_var_changes)

        # Fix 11: Broken method chains (semicolon in middle of fluent chain)
        code, chain_changes = self._fix_broken_method_chains(code)
        changes.extend(chain_changes)

        # Fix 12: Missing @Mock/@InjectMocks is now handled in DirectCodeGenerator
        # which has access to source class dependencies

        return PostProcessResult(
            original_code=original,
            fixed_code=code,
            changes_made=changes,
            was_modified=code != original
        )

    def _fix_malformed_imports(self, code: str) -> tuple[str, list[str]]:
        """
        Fix malformed import lines where LLM concatenated multiple imports.

        Fixes patterns like:
            import foo.Bar;import baz.Qux;java.util.List;
        Into:
            import foo.Bar;
            import baz.Qux;
            import java.util.List;
        """
        changes = []
        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            # Check if line has multiple imports concatenated
            if line.count('import ') > 1 or (line.startswith('import ') and ';' in line and line.rstrip(';').count(';') > 0):
                # Split by semicolons and reconstruct
                parts = line.split(';')
                new_imports = []

                for part in parts:
                    part = part.strip()
                    if not part:
                        continue

                    # Check if part starts with 'import'
                    if part.startswith('import '):
                        new_imports.append(part + ';')
                    elif part.startswith('static '):
                        # Handle 'import static' that was split
                        new_imports.append('import ' + part + ';')
                    elif '.' in part and not part.startswith('//'):
                        # Looks like a package path without 'import' keyword
                        # e.g., "java.util.List"
                        new_imports.append('import ' + part + ';')

                if len(new_imports) > 1:
                    changes.append(f"Split {len(new_imports)} concatenated imports")
                    fixed_lines.extend(new_imports)
                elif len(new_imports) == 1:
                    fixed_lines.append(new_imports[0])
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines), changes

    def _fix_python_literals(self, code: str) -> tuple[str, list[str]]:
        """Replace Python True/False/None with Java equivalents."""
        changes = []

        for pattern, replacement in self.PYTHON_LITERALS.items():
            if re.search(pattern, code):
                code = re.sub(pattern, replacement, code)
                changes.append(f"Replaced Python literal with Java '{replacement}'")

        return code, changes

    def _fix_inline_statements(self, code: str) -> tuple[str, list[str]]:
        """
        Fix inline statements inside method call parentheses.

        Removes entire test methods that contain invalid Java syntax like:
            instance.method(Type x = new Type(); x.setFoo("bar");)

        Uses parenthesis-depth tracking to handle nested parens like new Type().
        """
        changes = []

        # First pass: identify lines with invalid Java syntax
        invalid_lines = set()
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Check for semicolons inside parentheses
            if ';' in line and '(' in line:
                if self._has_semicolon_in_parens(line):
                    invalid_lines.add(i)
            # Check for plain English in assertions (not valid Java)
            if 'assertEquals(' in line or 'assertNotNull(' in line:
                if self._has_invalid_assertion(line):
                    invalid_lines.add(i)
            # Check for inline method definitions (Python-like nested functions)
            # Pattern: TypeName methodName(params) { ... };
            if self._has_inline_method_definition(line):
                invalid_lines.add(i)

        if not invalid_lines:
            return code, changes

        # Second pass: remove entire test methods containing invalid lines
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is start of a test method
            if '@Test' in line:
                # Find the end of this test method
                method_start = i
                method_end = self._find_method_end(lines, i)

                # Check if any line in this method has invalid syntax
                method_has_invalid = any(
                    j in invalid_lines
                    for j in range(method_start, method_end + 1)
                )

                if method_has_invalid:
                    # Skip this entire test method
                    test_name = self._extract_test_name(lines, method_start, method_end)
                    changes.append(f"Removed invalid test method: {test_name}")
                    i = method_end + 1
                    continue

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines), changes

    def _has_semicolon_in_parens(self, line: str) -> bool:
        """Check if line has semicolons inside parentheses (not in strings)."""
        paren_depth = 0
        in_string = False

        for i, c in enumerate(line):
            if c == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if in_string:
                continue

            if c == '(':
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
            elif c == ';' and paren_depth > 0:
                return True

        return False

    def _has_invalid_assertion(self, line: str) -> bool:
        """Check if line has invalid assertion patterns (plain English, nested quotes, Python syntax)."""
        # Pattern 1: Plain English starting assertions (no quotes at start of expected)
        # e.g., assertEquals(a Collection<Visit> containing...
        if re.search(r'assertEquals\s*\(\s*[a-z]+\s+[A-Z]', line):
            return True

        # Pattern 2: Nested quotes in string literals
        # e.g., assertEquals("text with "nested" quotes", result)
        if re.search(r'assertEquals\s*\(\s*"[^"]*"[^"]*"', line):
            return True

        # Pattern 3: Descriptive strings that aren't valid expected values
        # e.g., assertEquals("a PetType object with name...", result)
        if re.search(r'assertEquals\s*\(\s*"a\s+\w+\s+object', line, re.IGNORECASE):
            return True

        # Pattern 4: Assert with angle brackets outside of generics context
        # e.g., assertEquals(Collection<Visit>, result)
        if re.search(r'assertEquals\s*\([^"]*<[^>]+>[^"]*,', line):
            # But allow things like List.of() or actual generics
            if not re.search(r'assertEquals\s*\(\s*(new\s+|List\.|Set\.|Arrays\.)', line):
                return True

        # Pattern 5: Python dict/list syntax in Java
        # e.g., assertEquals([{'key': 'value'}], result)
        if re.search(r"assertEquals\s*\(\s*\[?\s*\{'\w+':", line):
            return True

        # Pattern 6: Python-style single quotes for strings
        # e.g., assertEquals('value', result)
        if re.search(r"assertEquals\s*\(\s*'[^']+'\s*,", line):
            return True

        # Pattern 7: Python list syntax start - assertEquals([
        # This is ALWAYS invalid in Java (Java uses List.of() or Arrays.asList())
        if re.search(r'assertEquals\s*\(\s*\[', line):
            return True

        # Pattern 8: Multi-line code block as expected value (has variable declarations)
        # e.g., assertEquals(Specialty s1 = new Specialty(); ...
        if re.search(r'assertEquals\s*\(\s*[A-Z][a-z]+\s+\w+\s*=', line):
            return True

        return False

    def _has_inline_method_definition(self, line: str) -> bool:
        """
        Check if line contains an inline method definition (Python-like nested function).

        This is invalid Java syntax. LLMs sometimes generate:
            PetType createPetType(String name) { PetType pt = new PetType(); pt.setName(name); return pt; };

        This looks like a Python nested function which Java doesn't support.
        Valid Java method definitions don't end with };
        """
        stripped = line.strip()

        # Quick filters
        if '(' not in stripped or '{' not in stripped or '}' not in stripped:
            return False

        # Skip if it's a lambda expression: () -> { ... }
        if '->' in stripped:
            return False

        # Skip if it starts with control flow keywords
        if re.match(r'^(if|else|while|for|switch|try|catch|finally)\s*[\({]', stripped):
            return False

        # Skip class/interface declarations
        if re.match(r'^(class|interface|enum|@interface)\s+', stripped):
            return False

        # Pattern: TypeName methodName(params) { ... };
        # This is the invalid pattern - a full method definition on one line ending with };
        # Valid patterns that look similar but are OK:
        #   - Lambda: () -> { ... }
        #   - Anonymous class: new Type() { ... }
        #   - Method reference within expression

        # Match: identifier identifier(params) { ... };
        # Where the second identifier is lowercase (method name)
        pattern = r'^[A-Z]\w*\s+[a-z]\w*\s*\([^)]*\)\s*\{.+\}\s*;'
        if re.match(pattern, stripped):
            # Additional check: must NOT be on first line of class (declaration)
            # and must contain code inside braces
            brace_content = re.search(r'\{(.+)\}', stripped)
            if brace_content:
                content = brace_content.group(1).strip()
                # If there's actual code (not just whitespace), it's an inline method
                if content and ';' in content:
                    return True

        # Also check for pattern with 'throws': ReturnType methodName(params) throws Exception { ... };
        pattern_throws = r'^[A-Z]\w*\s+[a-z]\w*\s*\([^)]*\)\s+throws\s+\w+\s*\{.+\}\s*;'
        if re.match(pattern_throws, stripped):
            return True

        return False

    def _find_method_end(self, lines: list[str], start: int) -> int:
        """Find the closing brace of a method starting at start."""
        brace_depth = 0
        found_open = False

        for i in range(start, len(lines)):
            for c in lines[i]:
                if c == '{':
                    brace_depth += 1
                    found_open = True
                elif c == '}':
                    brace_depth -= 1
                    if found_open and brace_depth == 0:
                        return i
        return len(lines) - 1

    def _extract_test_name(self, lines: list[str], start: int, end: int) -> str:
        """Extract test method name from a method block."""
        for i in range(start, min(end + 1, len(lines))):
            match = re.search(r'void\s+(\w+)\s*\(', lines[i])
            if match:
                return match.group(1)
        return "unknown"

    def _fix_line_inline_statements(self, line: str, changes: list) -> str:
        """Fix inline statements in a single line."""
        # Find patterns like: identifier.method(content with semicolons)
        # We need to track parenthesis depth

        # Simple check: does this line have ; inside () that's not in a string?
        paren_depth = 0
        in_string = False
        has_semi_in_parens = False
        method_call_start = -1

        for i, c in enumerate(line):
            if c == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if in_string:
                continue

            if c == '(':
                if paren_depth == 0:
                    method_call_start = i
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
            elif c == ';' and paren_depth > 0:
                has_semi_in_parens = True

        if not has_semi_in_parens:
            return line

        # Extract the method call content using depth tracking
        return self._extract_and_fix_method_call(line, changes)

    def _extract_and_fix_method_call(self, line: str, changes: list) -> str:
        """Extract method call with inline statements and fix it."""
        # Pattern: whitespace + identifier.method(...)
        # Find where the method call starts
        indent_match = re.match(r'^(\s*)', line)
        indent = indent_match.group(1) if indent_match else ''

        # Find object.method( pattern
        call_match = re.search(r'(\w+)\.(\w+)\s*\(', line)
        if not call_match:
            return line

        obj = call_match.group(1)
        method = call_match.group(2)
        paren_start = call_match.end() - 1  # Position of '('

        # Track paren depth to find matching ')'
        paren_depth = 0
        in_string = False
        paren_end = -1

        for i in range(paren_start, len(line)):
            c = line[i]
            if c == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if in_string:
                continue

            if c == '(':
                paren_depth += 1
            elif c == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    paren_end = i
                    break

        if paren_end == -1:
            return line  # Couldn't find matching paren

        # Extract content inside parentheses
        inner = line[paren_start + 1:paren_end]

        # Check if there are semicolons (not in strings)
        has_semi = False
        temp_in_string = False
        for i, c in enumerate(inner):
            if c == '"' and (i == 0 or inner[i-1] != '\\'):
                temp_in_string = not temp_in_string
            if c == ';' and not temp_in_string:
                has_semi = True
                break

        if not has_semi:
            return line

        # Split by semicolons (respecting strings and parens)
        statements = self._split_by_semicolons(inner)

        if len(statements) <= 1:
            return line

        # This is invalid Java - comment it out and add a TODO
        changes.append(f"Commented out invalid inline statements in {obj}.{method}()")

        # Create a safe replacement - just comment out the problematic line
        commented = f"{indent}// TODO: Fix invalid Java syntax - semicolons inside method call\n{indent}// {line.strip()}"
        return commented

    def _split_by_semicolons(self, content: str) -> list[str]:
        """Split content by semicolons, respecting strings and parens."""
        parts = []
        current = []
        paren_depth = 0
        in_string = False

        for i, c in enumerate(content):
            if c == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
                current.append(c)
            elif in_string:
                current.append(c)
            elif c == '(':
                paren_depth += 1
                current.append(c)
            elif c == ')':
                paren_depth -= 1
                current.append(c)
            elif c == ';' and paren_depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(c)

        if current:
            parts.append(''.join(current).strip())

        return [p for p in parts if p]

    def _fix_class_as_method(self, code: str) -> tuple[str, list[str]]:
        """
        Fix class names used as method names.

        Transforms:
            instance.Owner(...) -> instance.setOwner(...) or removes
            instance.Pet(...) -> instance.addPet(...)
        """
        changes = []

        # Pattern: instance.ClassName(...) where ClassName starts with uppercase
        pattern = r'instance\.([A-Z][a-zA-Z]+)\s*\(([^)]*)\)'

        def fix_class_method(match):
            class_name = match.group(1)
            args = match.group(2)

            # Common mappings based on Spring PetClinic patterns
            # Usually these are constructor-like calls that should be setters or additions
            if class_name in ['Owner', 'Pet', 'Visit', 'Vet', 'Specialty']:
                # If args look like they should set something, use setter
                if args and not args.startswith('new'):
                    changes.append(f"Removed invalid instance.{class_name}() call")
                    # This is likely a confused LLM call - comment it out
                    return f"// REMOVED: instance.{class_name}({args})"

            return match.group(0)

        code = re.sub(pattern, fix_class_method, code)
        return code, changes

    def _fix_empty_collections(self, code: str) -> tuple[str, list[str]]:
        """
        Fix Python-style empty collections.

        Transforms:
            assertEquals([], result) -> assertTrue(result.isEmpty())
            assertEquals({}, result) -> assertTrue(result.isEmpty())
        """
        changes = []

        # Fix empty list comparison
        if 'assertEquals([], ' in code:
            code = re.sub(
                r'assertEquals\(\[\],\s*(\w+)\)',
                r'assertTrue(\1.isEmpty())',
                code
            )
            changes.append("Fixed empty list assertion to use isEmpty()")

        # Fix empty dict/map comparison
        if 'assertEquals({}, ' in code:
            code = re.sub(
                r'assertEquals\(\{\},\s*(\w+)\)',
                r'assertTrue(\1.isEmpty())',
                code
            )
            changes.append("Fixed empty map assertion to use isEmpty()")

        return code, changes

    def _fix_string_assertions(self, code: str) -> tuple[str, list[str]]:
        """
        Fix string assertions that should be object checks.

        Transforms assertions comparing to descriptive strings like:
            assertEquals("Pet object with name", result)
        Into:
            assertNotNull(result)
        """
        changes = []

        # Pattern: assertEquals with descriptive string as expected value
        # These are LLM placeholders that won't work
        descriptive_patterns = [
            r'assertEquals\("([^"]*(?:object|collection|list|the|Pet|Owner|Vet)[^"]*)"\s*,\s*(\w+)\)',
            r'assertEquals\("([^"]*pet1[^"]*)"\s*,\s*(\w+)\)',
            r'assertEquals\("([a-z]+[A-Z][^"]*)"\s*,\s*(\w+)\)',  # camelCase strings like "mockPage"
        ]

        for pattern in descriptive_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for desc, var in matches:
                # Replace with assertNotNull as a safe default
                code = re.sub(
                    rf'assertEquals\("{re.escape(desc)}"\s*,\s*{var}\)',
                    f'assertNotNull({var})',
                    code
                )
                changes.append(f"Replaced string assertion '{desc[:30]}...' with assertNotNull")

        return code, changes

    def _fix_unquoted_paths(self, code: str) -> tuple[str, list[str]]:
        """
        Fix unquoted path-like expressions in assertions.

        LLMs sometimes output:
            assertEquals(vets / vetList, result)
        Which should be:
            assertEquals("vets/vetList", result)
        """
        changes = []

        # Pattern: assertEquals(identifier / identifier, variable)
        # This is invalid Java - looks like the LLM tried to express a path
        pattern = r'assertEquals\(\s*(\w+)\s*/\s*(\w+)\s*,\s*(\w+)\s*\)'

        def fix_path(match):
            path1 = match.group(1)
            path2 = match.group(2)
            var = match.group(3)
            changes.append(f"Fixed unquoted path '{path1}/{path2}' to string literal")
            return f'assertEquals("{path1}/{path2}", {var})'

        code = re.sub(pattern, fix_path, code)
        return code, changes

    def _fix_missing_semicolons(self, code: str) -> tuple[str, list[str]]:
        """
        Fix missing semicolons after object instantiation.

        LLMs sometimes output:
            Pet pet = new Pet()
            instance.setBirthDate(LocalDate.of(2020, 5, 15));
        Which should be:
            Pet pet = new Pet();
            instance.setBirthDate(LocalDate.of(2020, 5, 15));

        Also fixes:
            Type var = new Type()  (end of line without semicolon)
        """
        changes = []
        lines = code.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            stripped = line.rstrip()

            # Pattern 1: Line ends with ) but not ); and looks like instantiation or method call
            # e.g., "Pet pet = new Pet()" or "instance.setName("Buddy")"
            if stripped.endswith(')') and not stripped.endswith(');'):
                # Check if this looks like a statement (not a method signature or control structure)
                # Must not be: method declaration, if/while/for, class declaration, annotation
                if self._is_statement_line(stripped):
                    fixed_lines.append(line.rstrip() + ';')
                    changes.append(f"Added missing semicolon: {stripped[-40:]}...")
                    continue

            # Pattern 2: Variable assignment ending without semicolon
            # e.g., "String name = pet.getName()"
            if '=' in stripped and not stripped.endswith(';') and not stripped.endswith('{'):
                if self._is_assignment_without_semicolon(stripped):
                    fixed_lines.append(line.rstrip() + ';')
                    changes.append(f"Added missing semicolon to assignment")
                    continue

            fixed_lines.append(line)

        return '\n'.join(fixed_lines), changes

    def _is_statement_line(self, line: str) -> bool:
        """Check if line looks like a Java statement that needs a semicolon."""
        stripped = line.strip()

        # Skip method/constructor declarations
        if re.match(r'^\s*(public|private|protected|void|static|final|abstract)\s+', stripped):
            # But allow "Type var = new Type()" even with access modifiers
            if '=' in stripped and 'new ' in stripped:
                return True
            # Skip if it looks like a method signature (has throws or just modifiers before name)
            if 'throws' in stripped or re.match(r'^\s*(public|private|protected)\s+\w+\s*\(', stripped):
                return False

        # Skip control structures
        if re.match(r'^\s*(if|else|while|for|switch|try|catch|finally)\s*[\({]?', stripped):
            return False

        # Skip annotations
        if stripped.startswith('@'):
            return False

        # Skip class/interface declarations
        if re.match(r'^\s*(class|interface|enum)\s+', stripped):
            return False

        # Skip opening braces
        if stripped.endswith('{'):
            return False

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            return False

        # Skip lambda/anonymous class starts
        if stripped.endswith('->') or stripped.endswith('-> {'):
            return False

        # This looks like a statement
        # Must have balanced parentheses and end with )
        return self._has_balanced_parens(stripped)

    def _has_balanced_parens(self, line: str) -> bool:
        """Check if parentheses are balanced."""
        depth = 0
        in_string = False
        for i, c in enumerate(line):
            if c == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
            if in_string:
                continue
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
        return depth == 0

    def _is_assignment_without_semicolon(self, line: str) -> bool:
        """Check if line is a variable assignment missing a semicolon."""
        stripped = line.strip()

        # Must have = sign
        if '=' not in stripped:
            return False

        # Skip comparison operators
        if '==' in stripped or '!=' in stripped or '<=' in stripped or '>=' in stripped:
            return False

        # Skip augmented assignments that might be in for loops
        if '+=' in stripped or '-=' in stripped:
            return False

        # Must end with a value (not opening brace or arrow)
        if stripped.endswith('{') or stripped.endswith('->'):
            return False

        # Skip if it's a lambda or method reference
        if '->' in stripped:
            return False

        # Check for common patterns:
        # Type var = value
        # Type var = new Type()
        # var = value
        if re.match(r'^\s*\w+(\s*<[^>]+>)?\s+\w+\s*=', stripped):
            return True

        # Simple assignment: var = value
        if re.match(r'^\s*\w+\s*=\s*[^=]', stripped):
            return True

        return False

    def _fix_instance_variable_mismatch(self, code: str) -> tuple[str, list[str]]:
        """
        Fix tests that create a local variable but use 'instance' in assertions.

        LLMs sometimes output:
            Vet vet = new Vet();
            vet.addSpecialty(specialty);
            assertThrows(Exception.class, () -> { instance.getSpecialties(); });

        Which should use the local 'vet' variable instead of 'instance':
            assertThrows(Exception.class, () -> { vet.getSpecialties(); });
        """
        changes = []
        lines = code.split('\n')

        # Find test methods and process each one
        i = 0
        while i < len(lines):
            if '@Test' in lines[i]:
                method_start = i
                method_end = self._find_method_end(lines, i)

                # Extract the test method
                method_lines = lines[method_start:method_end + 1]
                fixed_method, method_changes = self._fix_method_instance_mismatch(method_lines)

                if method_changes:
                    # Replace the method lines
                    lines = lines[:method_start] + fixed_method + lines[method_end + 1:]
                    changes.extend(method_changes)
                    # Adjust index for any line count changes
                    i = method_start + len(fixed_method)
                else:
                    i = method_end + 1
            else:
                i += 1

        return '\n'.join(lines), changes

    def _fix_method_instance_mismatch(self, method_lines: list[str]) -> tuple[list[str], list[str]]:
        """
        Fix instance/local variable mismatch within a single test method.

        Returns the fixed lines and list of changes made.
        """
        changes = []
        method_text = '\n'.join(method_lines)

        # Find local variable declarations that create new instances of the same type
        # Pattern: Type varName = new Type(...)
        local_var_pattern = r'(\w+)\s+(\w+)\s*=\s*new\s+\1\s*\('

        local_vars = re.findall(local_var_pattern, method_text)
        if not local_vars:
            return method_lines, changes

        # For each local variable found
        for type_name, var_name in local_vars:
            # Skip if var_name is 'instance' (that's the fixture variable)
            if var_name == 'instance':
                continue

            # Check if this variable is used (modified) in the method
            # Look for patterns like: varName.setSomething() or varName.addSomething()
            var_usage = re.search(rf'{var_name}\.\w+\s*\(', method_text)
            if not var_usage:
                continue

            # Check if 'instance' is used in assertions/lambdas for the same type of operation
            # Pattern: instance.methodName() in an assertion context
            assertion_with_instance = re.search(
                r'(assert\w*|assertThrows)\s*\([^)]*\(\s*\)\s*->\s*\{?\s*instance\.\w+\s*\(',
                method_text
            )
            if assertion_with_instance:
                # Replace 'instance.' with 'varName.' in assertion lambdas
                # Be careful to only replace in the assertion context
                fixed_text = self._replace_instance_in_assertions(method_text, var_name)
                if fixed_text != method_text:
                    changes.append(f"Replaced 'instance' with '{var_name}' in assertion")
                    return fixed_text.split('\n'), changes

            # Also check for direct assertion patterns like:
            # var result = instance.getSpecialties();  (when vet was set up)
            result_from_instance = re.search(
                rf'(var|{type_name})\s+\w+\s*=\s*instance\.\w+\s*\(',
                method_text
            )
            if result_from_instance and var_usage:
                # The test sets up a local var but then calls instance
                fixed_text = method_text.replace(f'instance.', f'{var_name}.')
                if fixed_text != method_text:
                    changes.append(f"Replaced 'instance' with '{var_name}' throughout test")
                    return fixed_text.split('\n'), changes

        return method_lines, changes

    def _replace_instance_in_assertions(self, method_text: str, var_name: str) -> str:
        """
        Replace 'instance' with var_name specifically in assertion lambdas.

        Handles patterns like:
            assertThrows(Exception.class, () -> { instance.method(); });
            assertThrows(Exception.class, () -> instance.method());
        """
        # Pattern for lambda body with instance
        # Matches: () -> { instance.xxx } or () -> instance.xxx
        patterns = [
            # Lambda with braces: () -> { instance.method() }
            (r'(\(\s*\)\s*->\s*\{[^}]*?)instance\.', rf'\1{var_name}.'),
            # Lambda without braces: () -> instance.method()
            (r'(\(\s*\)\s*->\s*)instance\.', rf'\1{var_name}.'),
        ]

        for pattern, replacement in patterns:
            method_text = re.sub(pattern, replacement, method_text)

        return method_text


    def _fix_method_on_wrong_object(self, code: str) -> tuple[str, list[str]]:
        """
        Fix method calls made on wrong objects (parameters instead of test instance).

        LLMs sometimes call the method under test on a parameter object:
            var result = petType.print(petType, locale);  // WRONG
        Should be:
            var result = controller.print(petType, locale);  // CORRECT

        Also fixes mismatch between 'instance' and 'controller':
            instance.print(...)  // WRONG if class has controller
        Should be:
            controller.print(...)  // CORRECT

        This fix detects the test instance variable (instance/controller) and replaces
        method calls on parameter objects with calls on the test instance.
        """
        changes = []

        # Find test class name
        class_match = re.search(r'class\s+(\w+)Test\s*\{', code)
        if not class_match:
            return code, changes

        class_under_test = class_match.group(1)

        # Find the test instance variable name (instance or controller)
        instance_var = None
        has_controller = re.search(rf'private\s+{class_under_test}\s+controller\s*;', code)
        has_instance = re.search(rf'private\s+{class_under_test}\s+instance\s*;', code)

        if has_controller:
            instance_var = 'controller'
            # Also fix any 'instance.' calls to use 'controller.' since only controller exists
            if 'instance.' in code and not has_instance:
                code = re.sub(r'\binstance\.', 'controller.', code)
                changes.append("Replaced 'instance.' with 'controller.' throughout")
        elif has_instance:
            instance_var = 'instance'
        else:
            return code, changes

        # Find methods being tested by looking for method names that appear after @DisplayName or @Test
        # Also look at the actual method calls to find patterns like:
        # var result = paramVar.methodName(...)
        # Where paramVar is NOT the test instance

        # Pattern: var result = someVar.methodName(someVar, ...);
        # This is suspicious when:
        # 1. someVar appears as both the object AND a parameter (e.g., petType.print(petType, ...))
        # 2. The object is a lowercase variable that's NOT instance/controller

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            fixed_line = line

            # Pattern: var result = wrongVar.methodName(wrongVar, ...)
            # where wrongVar is not the test instance
            method_call_match = re.search(
                r'var\s+result\s*=\s*(\w+)\.(\w+)\s*\((\1)\s*,',
                line
            )
            if method_call_match:
                wrong_var = method_call_match.group(1)
                method_name = method_call_match.group(2)

                # Check if this is NOT the test instance
                if wrong_var not in ('instance', 'controller', instance_var):
                    # This is likely calling a method on a parameter object
                    # Replace wrongVar.methodName( with instance_var.methodName(
                    old_call = f'{wrong_var}.{method_name}('
                    new_call = f'{instance_var}.{method_name}('
                    fixed_line = line.replace(old_call, new_call)
                    if fixed_line != line:
                        changes.append(f"Fixed method call: {wrong_var}.{method_name} -> {instance_var}.{method_name}")

            # Also fix pattern: wrongVar.methodName("text", ...) where wrongVar should be test instance
            # but only if we can determine the method belongs to the class under test
            if fixed_line == line:  # Only if not already fixed
                # Pattern: someVar.parse(...) or someVar.print(...) where someVar is a lowercase parameter
                method_call_match2 = re.search(
                    r'(\w+)\.(\w+)\s*\(((?!this)[^)]+)\)',
                    line
                )
                if method_call_match2:
                    obj_var = method_call_match2.group(1)
                    method_name = method_call_match2.group(2)

                    # Check if this looks like a parameter object (lowercase, not instance/controller)
                    # and the method name is NOT a common object method
                    common_methods = {'get', 'set', 'add', 'remove', 'size', 'isEmpty', 'contains',
                                     'toString', 'equals', 'hashCode', 'getClass', 'getName',
                                     'setName', 'getId', 'setId', 'getValue', 'setValue'}

                    if (obj_var not in ('instance', 'controller', instance_var) and
                        obj_var[0].islower() and
                        method_name not in common_methods and
                        not method_name.startswith('get') and
                        not method_name.startswith('set')):

                        # Check if there's a local variable declaration for this obj_var
                        # Pattern: TypeName objVar = new TypeName()
                        var_decl_pattern = rf'(\w+)\s+{obj_var}\s*='
                        var_decl = re.search(var_decl_pattern, code)

                        if var_decl:
                            # This is a parameter object - replace with test instance
                            old_call = f'{obj_var}.{method_name}('
                            new_call = f'{instance_var}.{method_name}('
                            fixed_line = line.replace(old_call, new_call)
                            if fixed_line != line:
                                changes.append(f"Fixed method call: {obj_var}.{method_name} -> {instance_var}.{method_name}")

            fixed_lines.append(fixed_line)

        return '\n'.join(fixed_lines), changes

    def _fix_mock_variable_mismatch(self, code: str) -> tuple[str, list[str]]:
        """
        Fix mock variable name mismatches in test code.

        LLMs sometimes declare a mock with one name but use a different name:
            @Mock
            private PetTypeRepository types;  // Declared as 'types'
            ...
            when(petTypeRepository.findPetTypes())  // Used as 'petTypeRepository'

        This fix:
        1. Finds all @Mock declarations: @Mock private TypeName varName;
        2. Builds a map of TypeName -> declared varName
        3. Finds usages of wrong variable names (TypeName lowercased as variable)
        4. Replaces with the correct declared variable name
        """
        changes = []

        # Find all @Mock declarations
        # Pattern: @Mock followed by private TypeName varName;
        # Can be on same line or multiple lines
        mock_pattern = r'@Mock\s+(?:private\s+)?(\w+)\s+(\w+)\s*;'
        mock_declarations = re.findall(mock_pattern, code)

        if not mock_declarations:
            return code, changes

        # Build map: TypeName -> declared_var_name
        # Also track common wrong names (type lowercased with Repository suffix)
        type_to_var = {}
        wrong_to_correct = {}

        for type_name, var_name in mock_declarations:
            type_to_var[type_name] = var_name

            # Generate possible wrong variable names that LLM might use
            # Common patterns:
            # 1. PetTypeRepository -> petTypeRepository (standard camelCase)
            # 2. OwnerRepository -> ownerRepository
            # 3. VetRepository -> vetRepository
            # 4. TypeRepository -> typeRepository (remove prefix)

            # Generate camelCase from type name
            camel_case = type_name[0].lower() + type_name[1:]
            if camel_case != var_name:
                wrong_to_correct[camel_case] = var_name

            # Also handle Repository suffix being added
            if type_name.endswith('Repository'):
                # e.g., PetTypeRepository with var 'types' -> LLM might use 'petTypeRepository'
                base = type_name[:-len('Repository')]
                wrong_repo_var = base[0].lower() + base[1:] + 'Repository'
                if wrong_repo_var != var_name:
                    wrong_to_correct[wrong_repo_var] = var_name

        if not wrong_to_correct:
            return code, changes

        # Replace wrong variable names with correct ones
        for wrong_var, correct_var in wrong_to_correct.items():
            # Only replace if the wrong variable is actually used but not declared
            # Check if wrong_var appears but is not declared as a field
            if re.search(rf'\b{wrong_var}\s*\.', code):
                # Check it's not declared somewhere
                if not re.search(rf'(private|protected|public)?\s+\w+\s+{wrong_var}\s*[;=]', code):
                    # Replace wrongVar. with correctVar.
                    code = re.sub(rf'\b{wrong_var}\.', f'{correct_var}.', code)
                    changes.append(f"Fixed mock variable: {wrong_var} -> {correct_var}")

        return code, changes

    def _fix_broken_method_chains(self, code: str) -> tuple[str, list[str]]:
        """
        Fix broken method chains where LLM puts semicolon mid-chain.

        LLMs sometimes output:
            assertThatThrownBy(() -> controller.method());
                .isInstanceOf(Exception.class);

        Which should be:
            assertThatThrownBy(() -> controller.method())
                .isInstanceOf(Exception.class);

        The issue is the semicolon after the first part breaks the fluent chain.
        """
        changes = []

        # Pattern: );\n followed by whitespace and .methodName(
        # This is a broken method chain where semicolon should not be there
        pattern = r'\);\s*\n(\s*)\.(\w+)\s*\('

        def fix_chain(match):
            indent = match.group(1)
            method = match.group(2)
            changes.append(f"Fixed broken method chain before .{method}()")
            return f')\n{indent}.{method}('

        fixed_code = re.sub(pattern, fix_chain, code)

        return fixed_code, changes


def postprocess_java_code(code: str) -> PostProcessResult:
    """
    Convenience function to post-process Java code.

    Usage:
        result = postprocess_java_code(test_code)
        clean_code = result.fixed_code
    """
    processor = JavaCodePostProcessor()
    return processor.process(code)
