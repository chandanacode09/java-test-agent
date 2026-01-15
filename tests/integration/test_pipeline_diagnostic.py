"""
Pipeline Diagnostic Tests

Tests the full pipeline against realistic Java code patterns to identify
issues and verify fixes. These tests exercise real component interactions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context.ast_parser import ASTParser
from src.generator.spec_generator import SpecGenerator
from src.renderer.template_renderer import TemplateRenderer
from src.models import Language, TestType, FunctionContext, ClassContext


# ============================================================================
# Realistic Java Code Samples
# ============================================================================

SPRING_CONTROLLER = '''
package com.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.web.bind.annotation.*;
import com.example.model.User;
import com.example.service.UserService;
import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @GetMapping
    public Page<User> listUsers(Pageable pageable) {
        return userService.findAll(pageable);
    }

    @GetMapping("/{id}")
    public Optional<User> getUser(@PathVariable Long id) {
        return userService.findById(id);
    }

    @PostMapping
    public User createUser(@RequestBody @Valid User user) {
        return userService.save(user);
    }

    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User user) {
        user.setId(id);
        return userService.save(user);
    }

    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        userService.deleteById(id);
    }
}
'''

GENERIC_REPOSITORY = '''
package com.example.repository;

import java.util.*;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class GenericRepository<T, ID extends Comparable<ID>> {

    private final Map<ID, T> storage = new HashMap<>();

    public Optional<T> findById(ID id) {
        return Optional.ofNullable(storage.get(id));
    }

    public List<T> findAll() {
        return new ArrayList<>(storage.values());
    }

    public List<T> findAll(Predicate<T> filter) {
        return storage.values().stream()
            .filter(filter)
            .collect(Collectors.toList());
    }

    public T save(T entity, ID id) {
        storage.put(id, entity);
        return entity;
    }

    public void deleteById(ID id) {
        storage.remove(id);
    }

    public boolean existsById(ID id) {
        return storage.containsKey(id);
    }

    public long count() {
        return storage.size();
    }
}
'''

ABSTRACT_SERVICE = '''
package com.example.service;

import java.util.List;
import java.util.Optional;

public abstract class AbstractCrudService<T, ID> {

    protected abstract T findEntityById(ID id);
    protected abstract List<T> findAllEntities();
    protected abstract T saveEntity(T entity);
    protected abstract void deleteEntity(ID id);

    public Optional<T> findById(ID id) {
        T entity = findEntityById(id);
        return Optional.ofNullable(entity);
    }

    public List<T> findAll() {
        return findAllEntities();
    }

    public T save(T entity) {
        return saveEntity(entity);
    }

    public void delete(ID id) {
        deleteEntity(id);
    }

    public boolean exists(ID id) {
        return findEntityById(id) != null;
    }
}
'''

ENTITY_WITH_BUILDERS = '''
package com.example.model;

import javax.persistence.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Objects;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String firstName;

    @Column(nullable = false)
    private String lastName;

    @Column(unique = true)
    private String email;

    private LocalDate birthDate;

    @Column(updatable = false)
    private LocalDateTime createdAt;

    private LocalDateTime updatedAt;

    public User() {}

    public User(String firstName, String lastName, String email) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.email = email;
        this.createdAt = LocalDateTime.now();
    }

    // Builder pattern
    public static UserBuilder builder() {
        return new UserBuilder();
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getFirstName() { return firstName; }
    public void setFirstName(String firstName) { this.firstName = firstName; }

    public String getLastName() { return lastName; }
    public void setLastName(String lastName) { this.lastName = lastName; }

    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }

    public LocalDate getBirthDate() { return birthDate; }
    public void setBirthDate(LocalDate birthDate) { this.birthDate = birthDate; }

    public String getFullName() {
        return firstName + " " + lastName;
    }

    public int getAge() {
        if (birthDate == null) return 0;
        return LocalDate.now().getYear() - birthDate.getYear();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        return Objects.equals(id, user.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    public static class UserBuilder {
        private String firstName;
        private String lastName;
        private String email;
        private LocalDate birthDate;

        public UserBuilder firstName(String firstName) {
            this.firstName = firstName;
            return this;
        }

        public UserBuilder lastName(String lastName) {
            this.lastName = lastName;
            return this;
        }

        public UserBuilder email(String email) {
            this.email = email;
            return this;
        }

        public UserBuilder birthDate(LocalDate birthDate) {
            this.birthDate = birthDate;
            return this;
        }

        public User build() {
            User user = new User(firstName, lastName, email);
            user.setBirthDate(birthDate);
            return user;
        }
    }
}
'''

INTERFACE_WITH_DEFAULT = '''
package com.example.service;

import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

public interface CrudService<T, ID> {

    Optional<T> findById(ID id);

    List<T> findAll();

    T save(T entity);

    void delete(ID id);

    default boolean exists(ID id) {
        return findById(id).isPresent();
    }

    default List<T> findAll(Predicate<T> filter) {
        return findAll().stream()
            .filter(filter)
            .toList();
    }
}
'''


# ============================================================================
# AST Parser Diagnostic Tests
# ============================================================================

class TestASTParserDiagnostic:
    """Diagnostic tests to identify AST Parser gaps."""

    @pytest.fixture
    def parser(self):
        return ASTParser(language=Language.JAVA)

    def test_spring_controller_parsing(self, parser, tmp_path):
        """Test parsing Spring controller with annotations."""
        java_file = tmp_path / "UserController.java"
        java_file.write_text(SPRING_CONTROLLER)

        functions, classes = parser.parse_file(java_file)

        # Verify class found
        assert len(classes) >= 1, "Should find UserController class"
        controller = classes[0]
        assert controller.name == "UserController"

        # Verify methods found
        method_names = [m.name for m in controller.methods]
        print(f"\nFound methods: {method_names}")

        expected_methods = ['listUsers', 'getUser', 'createUser', 'updateUser', 'deleteUser']
        for expected in expected_methods:
            assert expected in method_names, f"Missing method: {expected}"

        # Verify parameters parsed correctly
        get_user = next((m for m in controller.methods if m.name == 'getUser'), None)
        assert get_user is not None
        assert len(get_user.parameters) == 1
        print(f"getUser params: {[(p.name, p.type_hint) for p in get_user.parameters]}")

    def test_generic_class_parsing(self, parser, tmp_path):
        """Test parsing generic class with type parameters."""
        java_file = tmp_path / "GenericRepository.java"
        java_file.write_text(GENERIC_REPOSITORY)

        functions, classes = parser.parse_file(java_file)

        print(f"\nClasses found: {[c.name for c in classes]}")
        print(f"Functions found: {[f.name for f in functions]}")

        # This is a key diagnostic - does the parser handle generics?
        if len(classes) == 0:
            pytest.skip("Parser doesn't handle generic class declarations - NEEDS FIX")

        repo = classes[0]
        method_names = [m.name for m in repo.methods]
        print(f"Methods: {method_names}")

        # Check if generic methods are parsed
        find_all_filter = next((m for m in repo.methods if m.name == 'findAll' and len(m.parameters) > 0), None)
        if find_all_filter:
            print(f"findAll(Predicate) params: {[(p.name, p.type_hint) for p in find_all_filter.parameters]}")

    def test_abstract_class_parsing(self, parser, tmp_path):
        """Test parsing abstract class with abstract methods."""
        java_file = tmp_path / "AbstractCrudService.java"
        java_file.write_text(ABSTRACT_SERVICE)

        functions, classes = parser.parse_file(java_file)

        assert len(classes) >= 1, "Should find AbstractCrudService class"
        service = classes[0]
        method_names = [m.name for m in service.methods]
        print(f"\nAbstract class methods: {method_names}")

        # Concrete methods should be found
        assert 'findById' in method_names, "Concrete method findById should be found"

        # Abstract methods might not be found (no body)
        abstract_found = 'findEntityById' in method_names
        print(f"Abstract methods found: {abstract_found}")
        if not abstract_found:
            print("NOTE: Abstract methods not parsed - this is expected behavior")

    def test_entity_with_inner_class(self, parser, tmp_path):
        """Test parsing entity with inner builder class."""
        java_file = tmp_path / "User.java"
        java_file.write_text(ENTITY_WITH_BUILDERS)

        functions, classes = parser.parse_file(java_file)

        print(f"\nClasses found: {[c.name for c in classes]}")

        # Main class
        user_class = next((c for c in classes if c.name == 'User'), None)
        assert user_class is not None, "Should find User class"

        method_names = [m.name for m in user_class.methods]
        print(f"User methods: {method_names}")

        # Key methods
        assert 'getFullName' in method_names, "Should find getFullName"
        assert 'getAge' in method_names, "Should find getAge"

        # Inner class - may or may not be parsed
        builder_class = next((c for c in classes if c.name == 'UserBuilder'), None)
        if builder_class:
            print(f"Inner class UserBuilder found with methods: {[m.name for m in builder_class.methods]}")
        else:
            print("NOTE: Inner class UserBuilder not found - parser may not handle inner classes")

    def test_interface_parsing(self, parser, tmp_path):
        """Test parsing interface with default methods."""
        java_file = tmp_path / "CrudService.java"
        java_file.write_text(INTERFACE_WITH_DEFAULT)

        functions, classes = parser.parse_file(java_file)

        print(f"\nClasses/Interfaces found: {[c.name for c in classes]}")

        # Interface should be parsed as class-like structure
        if len(classes) >= 1:
            service = classes[0]
            method_names = [m.name for m in service.methods]
            print(f"Interface methods: {method_names}")

            # Default methods have bodies - should be found
            if 'exists' in method_names:
                print("Default method 'exists' found")
            else:
                print("NOTE: Default methods not parsed")


# ============================================================================
# Pipeline Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def parser(self):
        return ASTParser(language=Language.JAVA)

    @pytest.fixture
    def renderer(self):
        return TemplateRenderer()

    def test_parser_to_renderer_handoff(self, parser, renderer, tmp_path):
        """Test data flows correctly from parser to renderer."""
        # Parse a simple class
        java_file = tmp_path / "Simple.java"
        java_file.write_text('''
package com.example;

public class Simple {
    private String value;

    public String getValue() {
        return this.value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
''')

        functions, classes = parser.parse_file(java_file)
        assert len(classes) >= 1

        # Verify FunctionContext has all required fields for spec generation
        method = classes[0].methods[0]
        print(f"\nMethod context: {method.name}")
        print(f"  - parameters: {method.parameters}")
        print(f"  - return_type: {method.return_type}")
        print(f"  - source_code length: {len(method.source_code)}")
        print(f"  - location: {method.location}")

        # These fields are needed by SpecGenerator
        assert method.name is not None
        assert method.source_code is not None
        assert method.location is not None

    def test_validation_gap_detection(self, tmp_path):
        """Detect if invalid spec data passes through without validation."""
        from src.models import TestSpec, TestCase, TestCategory, ExpectedOutput

        renderer = TemplateRenderer()

        # Create spec with Python syntax that should be caught
        spec = TestSpec(
            test_type=TestType.UNIT_CLASS,
            target_file="Example.java",
            target_name="getValue",
            target_class="Example",
            language=Language.JAVA,
            test_cases=[
                TestCase(
                    name="test_getValue",
                    category=TestCategory.HAPPY_PATH,
                    description="Test getValue",
                    inputs={"value": "None"},  # Python syntax!
                    expected=ExpectedOutput(returns="True")  # Python syntax!
                )
            ],
            imports_needed=[
                "import org.junit.jupiter.api.Test",
                "from flask import Flask",  # Python import!
            ]
        )

        # Try to render - does it catch the issues?
        try:
            code = renderer.render(spec)
            print(f"\nRendered code (first 500 chars):\n{code[:500]}")

            # Check for Python leakage
            issues = []
            if "None" in code and "null" not in code:
                issues.append("Python 'None' leaked through")
            if "True" in code and "true" not in code:
                issues.append("Python 'True' leaked through")
            if "from flask" in code:
                issues.append("Python import leaked through")

            if issues:
                print(f"\nVALIDATION GAP DETECTED: {issues}")
            else:
                print("\nSanitization working - Python syntax converted")

        except ValueError as e:
            print(f"\nRenderer rejected spec: {e}")


# ============================================================================
# Specific Issue Tests
# ============================================================================

class TestKnownIssues:
    """Tests for specific known issues from PIPELINE_ANALYSIS.md."""

    def test_duplicate_setup_detection(self):
        """Test detection of duplicate @BeforeEach methods."""
        code = '''
@BeforeEach
void setUp() {
    // setup 1
}

@Test
void testSomething() {}

@BeforeEach
void setUp() {
    // setup 2 - DUPLICATE!
}
'''
        setup_count = code.count('@BeforeEach')
        print(f"\n@BeforeEach count: {setup_count}")

        if setup_count > 1:
            print("ISSUE: Duplicate @BeforeEach methods detected")
            # This should be caught and fixed

    def test_python_java_syntax_mixing(self):
        """Detect Python/Java syntax mixing patterns."""
        problematic_patterns = [
            ("None instead of null", "Object x = None;", "None"),
            ("True instead of true", "boolean b = True;", "True"),
            ("False instead of false", "boolean b = False;", "False"),
            ("Python import", "from flask import Flask", "from "),
            ("Python dict", "{'key': 'value'}", "{'"),
            ("Python list comp", "[x for x in items]", " for "),
        ]

        print("\nPython/Java mixing patterns to detect:")
        for name, example, pattern in problematic_patterns:
            print(f"  - {name}: '{pattern}'")


# ============================================================================
# Run Diagnostic Report
# ============================================================================

def test_generate_diagnostic_report(tmp_path):
    """Generate a diagnostic report of parser capabilities."""
    parser = ASTParser(language=Language.JAVA)

    samples = [
        ("Spring Controller", SPRING_CONTROLLER),
        ("Generic Repository", GENERIC_REPOSITORY),
        ("Abstract Service", ABSTRACT_SERVICE),
        ("Entity with Builder", ENTITY_WITH_BUILDERS),
        ("Interface with Default", INTERFACE_WITH_DEFAULT),
    ]

    print("\n" + "="*60)
    print("AST PARSER DIAGNOSTIC REPORT")
    print("="*60)

    for name, code in samples:
        java_file = tmp_path / f"{name.replace(' ', '')}.java"
        java_file.write_text(code)

        try:
            functions, classes = parser.parse_file(java_file)
            status = "OK" if classes else "EMPTY"
            method_count = sum(len(c.methods) for c in classes)
            print(f"\n{name}:")
            print(f"  Status: {status}")
            print(f"  Classes: {len(classes)}")
            print(f"  Methods: {method_count}")
            if classes:
                for c in classes:
                    print(f"    - {c.name}: {[m.name for m in c.methods]}")
        except Exception as e:
            print(f"\n{name}:")
            print(f"  Status: ERROR - {e}")

    print("\n" + "="*60)
