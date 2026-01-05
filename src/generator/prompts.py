"""
Prompt templates for LLM test generation.

These prompts guide the LLM to output structured JSON test specifications.
"""

from ..models import FunctionContext, ClassContext, ProjectTestPatterns, ReturnSemantics


# =============================================================================
# OFFICIAL EXAMPLES LIBRARY
# Source: Flask docs (https://flask.palletsprojects.com/en/stable/testing/)
#         Python docs (https://docs.python.org/3/library/unittest.mock-examples.html)
# =============================================================================

FLASK_EXAMPLES = {
    "fixture_basic": '''
# From Flask Official Docs - Basic Fixtures
@pytest.fixture()
def app():
    app = Flask(__name__)
    app.config.update({"TESTING": True})
    yield app

@pytest.fixture()
def client(app):
    return app.test_client()
''',
    "app_context": '''
# From Flask Official Docs - Application Context
def test_db_model(app):
    with app.app_context():
        result = some_function_using_current_app()
        assert result is not None
''',
    "request_context": '''
# From Flask Official Docs - Request Context
def test_request_context(app):
    with app.test_request_context("/path", method="POST"):
        # Now request, session, g are available
        result = function_using_request()
        assert result == expected
''',
    "client_request": '''
# From Flask Official Docs - Test Client
def test_request_example(client):
    response = client.get("/posts")
    assert response.status_code == 200
    assert b"Hello" in response.data
''',
}

MOCK_EXAMPLES = {
    "basic_mock": '''
# From Python Official Docs - Basic Mock
from unittest.mock import Mock

mock = Mock()
mock.method.return_value = 3
result = mock.method()
assert result == 3
''',
    "mock_with_spec": '''
# From Python Official Docs - Mock with Specification
from unittest.mock import Mock

# Use spec= to create a mock that only allows attributes from the real class
mock_loader = Mock(spec=BaseLoader)
mock_loader.some_attribute = "value"  # Set attributes AFTER creation
mock_loader.load.return_value = "loaded_data"
''',
    "mock_attributes": '''
# From Python Official Docs - Setting Mock Attributes
from unittest.mock import Mock

mock = Mock()
mock.x = 3                           # Set attribute directly
mock.method.return_value = "result"  # Set method return value
mock.name = "test_name"              # Set another attribute
''',
    "patch_decorator": '''
# From Python Official Docs - Patch Decorator
from unittest.mock import patch

@patch('module.ClassName')
def test_something(MockClass):
    instance = MockClass.return_value
    instance.method.return_value = "mocked"
    result = function_under_test()
    assert result == "mocked"
''',
    "patch_context": '''
# From Python Official Docs - Patch Context Manager
from unittest.mock import patch

def test_with_patch():
    with patch('module.function') as mock_func:
        mock_func.return_value = "mocked_value"
        result = code_that_calls_function()
        assert result == "mocked_value"
        mock_func.assert_called_once()
''',
}

GENERATOR_EXAMPLES = {
    "mock_generator": '''
# From Python Official Docs - Mocking Generators
from unittest.mock import Mock

mock = Mock()
mock.iter.return_value = iter([1, 2, 3])  # Use iter() for generator
result = list(mock.iter())
assert result == [1, 2, 3]
''',
    "test_generator": '''
# Testing a Generator Function
def test_generator_function():
    result = my_generator_function(arg1, arg2)

    # Verify it's a generator/iterator
    assert hasattr(result, '__iter__')

    # Consume and verify output
    items = list(result)
    assert items == ["expected1", "expected2"]
''',
}

ASYNC_EXAMPLES = {
    "async_test": '''
# From Python Official Docs - Async Testing
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function_under_test()
    assert result == expected_value
''',
    "mock_async": '''
# From Python Official Docs - Mocking Async
from unittest.mock import AsyncMock

mock = AsyncMock(return_value="async_result")
result = await mock()
assert result == "async_result"
''',
}

# JSON output examples showing CORRECT format
JSON_OUTPUT_EXAMPLES = {
    "basic": '''
CORRECT JSON OUTPUT FORMAT:
{
  "test_type": "unit_mocked",
  "test_cases": [{
    "name": "test_function_returns_expected",
    "inputs": {"param1": "string_value", "param2": 123, "flag": true},
    "expected": {"returns": "expected_string"}
  }]
}

IMPORTANT RULES:
- String values MUST be quoted: "hello" not hello
- Use true/false for booleans (JSON format), they will be converted to Python True/False
- Use null for None values
- Numbers don't need quotes: 123 not "123"
''',
    "mock_input": '''
CORRECT - How to specify Mock objects in inputs:
{
  "inputs": {
    "loader": "Mock(spec=BaseLoader)",
    "request": "Mock(spec=Request)",
    "simple_string": "hello",
    "number": 42
  }
}

WRONG - Never do this:
{
  "inputs": {
    "loader": "MockLoader()",           // DON'T invent class names
    "loader": "FakeLoader()",           // DON'T use Fake* names
    "loader": Mock(spec=BaseLoader)     // DON'T forget quotes around Mock()
  }
}
''',
}

# =============================================================================
# JAVA/JUNIT 5 EXAMPLES
# =============================================================================

JAVA_JUNIT_EXAMPLES = {
    "basic_test": '''
// JUnit 5 Basic Test
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAddition() {
        Calculator calc = new Calculator();
        int result = calc.add(2, 3);
        assertEquals(5, result);
    }
}
''',
    "exception_test": '''
// JUnit 5 Exception Testing
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

@Test
void testThrowsException() {
    assertThrows(IllegalArgumentException.class, () -> {
        service.processInvalidInput(null);
    });
}
''',
    "mockito_basic": '''
// Mockito Basic Mocking
import org.mockito.Mock;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;

@Mock
private UserRepository userRepository;

@Test
void testFindUser() {
    // Setup mock
    when(userRepository.findById(1L)).thenReturn(new User("John"));

    // Test
    User result = userService.getUser(1L);
    assertEquals("John", result.getName());

    // Verify
    verify(userRepository).findById(1L);
}
''',
    "setup_teardown": '''
// JUnit 5 Setup/Teardown
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;

private Service service;

@BeforeEach
void setUp() {
    service = new Service();
}

@AfterEach
void tearDown() {
    service.cleanup();
}
''',
}

JAVA_JSON_EXAMPLES = {
    "basic": '''
CORRECT JSON OUTPUT FORMAT FOR JAVA:
{
  "test_type": "unit_class",
  "language": "java",
  "test_cases": [{
    "name": "testMethodReturnsExpected",
    "inputs": {"param1": "\\"stringValue\\"", "param2": 123, "flag": true},
    "expected": {"returns": "\\"expectedString\\""}
  }]
}

IMPORTANT RULES FOR JAVA:
- Use null (not None) for null values
- Use true/false (not True/False) for booleans
- Use camelCase for test names: testGetUserById not test_get_user_by_id
- String values in Java need escaped quotes: "\\"hello\\""
- For mock objects use: "mock(ClassName.class)" (Mockito syntax)
''',
    "mock_input": '''
CORRECT - How to specify Mock objects for Java/Mockito:
{
  "inputs": {
    "pet": "mock(Pet.class)",
    "visit": "new Visit()",
    "petId": 1,
    "name": "\\"Fluffy\\""
  }
}

WRONG - Never use Python syntax for Java:
{
  "inputs": {
    "pet": "Mock()",                    // DON'T use Python Mock
    "pet": "Mock(spec=Pet)",            // DON'T use Python spec syntax
    "name": "None",                     // DON'T use Python None, use null
    "flag": "True"                      // DON'T use Python True, use true
  }
}
''',
}

# =============================================================================
# SPRING MVC CONTROLLER EXAMPLES
# =============================================================================

SPRING_CONTROLLER_EXAMPLES = {
    "controller_test_structure": '''
// Spring MVC Controller Test Pattern - MockitoExtension with @Mock/@InjectMocks
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

@ExtendWith(MockitoExtension.class)
class OwnerControllerTest {

    @Mock
    private OwnerRepository owners;  // Mock the injected dependency

    @InjectMocks
    private OwnerController controller;  // Inject mocks into controller

    private Owner testOwner;

    @BeforeEach
    void setUp() {
        testOwner = new Owner();
        testOwner.setId(1);
        testOwner.setFirstName("John");
        testOwner.setLastName("Doe");
    }
}
''',
    "controller_mock_repository": '''
// Mocking repository methods in controller tests
@Test
void testFindOwnerWithValidId() {
    // Setup: Mock repository to return our test object
    when(owners.findById(1)).thenReturn(Optional.of(testOwner));

    // Act: Call controller method
    Owner result = controller.findOwner(1);

    // Assert: Verify behavior
    assertEquals(testOwner, result);
    verify(owners).findById(1);
}

@Test
void testProcessCreationFormSuccess() {
    BindingResult result = mock(BindingResult.class);
    RedirectAttributes redirectAttributes = mock(RedirectAttributes.class);
    when(result.hasErrors()).thenReturn(false);
    when(owners.save(testOwner)).thenReturn(testOwner);

    String view = controller.processCreationForm(testOwner, result, redirectAttributes);

    assertEquals("redirect:/owners/" + testOwner.getId(), view);
    verify(owners).save(testOwner);
}
''',
    "controller_validation_errors": '''
// Testing validation error scenarios
@Test
void testProcessCreationFormWithErrors() {
    BindingResult result = mock(BindingResult.class);
    RedirectAttributes redirectAttributes = mock(RedirectAttributes.class);
    when(result.hasErrors()).thenReturn(true);  // Simulate validation error

    String view = controller.processCreationForm(testOwner, result, redirectAttributes);

    assertEquals("owners/createOrUpdateOwnerForm", view);
    verify(owners, never()).save(any());  // Repository should NOT be called
}
''',
    "controller_page_mocking": '''
// Mocking Page<T> for pagination tests
@Test
void testProcessFindFormMultipleOwnersFound() {
    Page<Owner> testPage = mock(Page.class);
    Model model = mock(Model.class);
    BindingResult result = mock(BindingResult.class);
    Owner searchOwner = new Owner();

    when(testPage.isEmpty()).thenReturn(false);
    when(testPage.getTotalElements()).thenReturn(2L);
    when(owners.findByLastNameStartingWith(any(), any(Pageable.class))).thenReturn(testPage);

    String view = controller.processFindForm(1, searchOwner, result, model);
    assertEquals("owners/ownersList", view);
}
''',
}

SPRING_CONTROLLER_JSON_EXAMPLES = {
    "controller_spec": '''
CORRECT JSON FORMAT FOR SPRING MVC CONTROLLER TESTS:
{
  "test_type": "unit_mocked",
  "language": "java",
  "target_class": "OwnerController",
  "framework_hints": ["spring_mvc", "controller"],
  "imports_needed": [
    "import org.junit.jupiter.api.extension.ExtendWith",
    "import org.mockito.InjectMocks",
    "import org.mockito.Mock",
    "import org.mockito.junit.jupiter.MockitoExtension",
    "import static org.mockito.Mockito.*",
    "import static org.mockito.ArgumentMatchers.*"
  ],
  "test_cases": [{
    "name": "testFindOwnerWithValidId",
    "category": "happy_path",
    "description": "Returns owner when found by ID",
    "inputs": {"ownerId": 1},
    "setup": [
      "when(owners.findById(1)).thenReturn(Optional.of(testOwner))"
    ],
    "expected": {"returns": "testOwner"}
  }]
}

CRITICAL FOR CONTROLLERS:
- Mock all injected dependencies (repositories, services)
- Use when().thenReturn() to setup mock behavior
- Use verify() to check mock interactions
- Test both success and error paths
- Mock BindingResult for form validation tests
''',
}


class TestPromptBuilder:
    """Build prompts for test specification generation."""

    # Spring annotations that indicate a component requiring dependency injection
    SPRING_ANNOTATIONS = [
        '@Controller', '@RestController', '@Service', '@Repository',
        '@Component', '@Configuration'
    ]

    # Spring framework hints
    SPRING_FRAMEWORK_HINTS = ['spring_mvc', 'spring', 'controller', 'service', 'repository']

    @classmethod
    def _is_spring_component(cls, func: FunctionContext) -> bool:
        """Detect if this function belongs to a Spring component (controller, service, etc.).

        Detection methods:
        1. Source code contains Spring annotations
        2. Framework hints indicate Spring
        3. Class name ends with Controller, Service, Repository
        4. Constructor has injected dependencies (common Spring pattern)
        """
        # Check framework hints
        if func.framework_hints:
            for hint in func.framework_hints:
                if hint.lower() in cls.SPRING_FRAMEWORK_HINTS:
                    return True

        # Check source code for Spring annotations
        source = func.source_code or ''
        for annotation in cls.SPRING_ANNOTATIONS:
            if annotation in source:
                return True

        # Check class name patterns
        class_name = func.class_name or ''
        if class_name.endswith(('Controller', 'Service', 'Repository', 'RestController')):
            return True

        # Check for constructor injection pattern (private final field + constructor)
        if 'private final' in source and 'public ' + class_name + '(' in source:
            return True

        return False

    @classmethod
    def _select_examples(cls, func: FunctionContext, language: str = "python") -> str:
        """Select relevant examples based on function characteristics and language.

        Returns 2-4 examples most relevant to the function being tested.
        """
        examples = []

        # Java-specific examples
        if language == "java":
            # Check if this is a Spring component (controller, service, etc.)
            is_spring_component = cls._is_spring_component(func)

            if is_spring_component:
                # Use Spring MVC controller examples
                examples.append(SPRING_CONTROLLER_JSON_EXAMPLES["controller_spec"])
                examples.append(SPRING_CONTROLLER_EXAMPLES["controller_test_structure"])
                examples.append(SPRING_CONTROLLER_EXAMPLES["controller_mock_repository"])
                examples.append(SPRING_CONTROLLER_EXAMPLES["controller_validation_errors"])

                header = """
═══════════════════════════════════════════════════════════════════════════════
SPRING MVC CONTROLLER EXAMPLES - Use @Mock/@InjectMocks pattern:
═══════════════════════════════════════════════════════════════════════════════
"""
                return header + "\n".join(examples[:4])

            # Standard Java examples for non-Spring classes
            examples.append(JAVA_JSON_EXAMPLES["basic"])
            examples.append(JAVA_JUNIT_EXAMPLES["basic_test"])

            # Mock examples for complex params
            has_complex_params = any(
                param.type_hint and param.type_hint not in ('int', 'String', 'boolean', 'long', 'double')
                for param in func.parameters
            )
            if has_complex_params:
                examples.append(JAVA_JUNIT_EXAMPLES["mockito_basic"])
                examples.append(JAVA_JSON_EXAMPLES["mock_input"])

            # Exception testing
            examples.append(JAVA_JUNIT_EXAMPLES["exception_test"])

            header = """
═══════════════════════════════════════════════════════════════════════════════
JAVA/JUNIT 5 EXAMPLES - Follow these patterns:
═══════════════════════════════════════════════════════════════════════════════
"""
            return header + "\n".join(examples[:4])

        # Python examples (default)
        examples.append(JSON_OUTPUT_EXAMPLES["basic"])

        # Flask-specific examples
        if 'flask' in func.framework_hints or 'flask_app' in func.requires_context:
            examples.append(FLASK_EXAMPLES["fixture_basic"])
            if 'request_context' in func.requires_context:
                examples.append(FLASK_EXAMPLES["request_context"])
            else:
                examples.append(FLASK_EXAMPLES["app_context"])

        # Generator examples
        if func.is_generator:
            examples.append(GENERATOR_EXAMPLES["test_generator"])

        # Mock examples for complex parameters
        has_complex_params = any(
            param.type_hint and any(
                pattern in param.type_hint
                for pattern in cls.COMPLEX_TYPE_PATTERNS
            )
            for param in func.parameters
        )
        if has_complex_params:
            examples.append(MOCK_EXAMPLES["mock_with_spec"])
            examples.append(JSON_OUTPUT_EXAMPLES["mock_input"])
        elif func.calls:  # Has function calls that might need mocking
            examples.append(MOCK_EXAMPLES["patch_decorator"])

        # Async examples
        if func.is_async:
            examples.append(ASYNC_EXAMPLES["async_test"])

        # Limit to 4 examples max to avoid token bloat
        examples = examples[:4]

        if not examples:
            return ""

        header = """
═══════════════════════════════════════════════════════════════════════════════
OFFICIAL EXAMPLES - Follow these patterns from Flask & Python documentation:
═══════════════════════════════════════════════════════════════════════════════
"""
        return header + "\n".join(examples)

    SYSTEM_PROMPT_PYTHON = """You are a senior test engineer who writes comprehensive test specifications for PYTHON code.

Your task is to analyze code and output a JSON specification for test cases.
You do NOT write actual test code - you specify WHAT should be tested.

Guidelines:
1. Cover happy paths first, then edge cases, then error handling
2. Be specific about inputs and expected outputs
3. Identify what needs to be mocked
4. Consider boundary conditions and null/empty inputs
5. Output valid JSON that matches the schema exactly
6. IMPORTANT: Pay attention to semantic hints about generators, context requirements, and side effects

CRITICAL RULES TO AVOID HALLUCINATIONS:
- NEVER invent class names that don't exist (no MockLoader, FakeRequest, etc.)
- For complex types, use: Mock(spec=TypeName) from unittest.mock
- ALL string values MUST be quoted: "hello" not hello
- Only use imports that actually exist in Python/the framework
- If unsure about expected output, use null for 'returns' and add a description

You must output ONLY valid JSON - no markdown, no explanations."""

    SYSTEM_PROMPT_JAVA = """You are a senior test engineer who writes comprehensive test specifications for JAVA code.

Your task is to analyze Java code and output a JSON specification for JUnit 5 test cases.
You do NOT write actual test code - you specify WHAT should be tested.

Guidelines:
1. Cover happy paths first, then edge cases, then error handling
2. Be specific about inputs and expected outputs
3. Identify what needs to be mocked using Mockito patterns
4. Consider boundary conditions and null inputs
5. Output valid JSON that matches the schema exactly

CRITICAL RULES FOR JAVA - DO NOT USE PYTHON SYNTAX:
- Use null (NOT None) for null values
- Use true/false (NOT True/False) for booleans
- Use camelCase for test names: testGetUserById NOT test_get_user_by_id
- For mock objects use Mockito: mock(ClassName.class) NOT Mock(spec=ClassName)
- String values need quotes: "hello"

IMPORTANT - HOW TO CREATE OBJECTS:
Most Java domain objects have NO-ARG constructors. You must create them and then call setters:

CORRECT way to create objects:
  Pet pet = new Pet();
  pet.setName("Fluffy");
  owner.addPet(pet);

  Specialty specialty = new Specialty();
  specialty.setName("Cardiology");
  vet.addSpecialty(specialty);

  Visit visit = new Visit();
  visit.setDescription("checkup");

WRONG - Do NOT use constructors with arguments unless you see them in the source:
  new Pet("Fluffy")           // WRONG - Pet has no String constructor
  new Specialty("Cardiology") // WRONG - Specialty has no String constructor
  new Visit("checkup")        // WRONG - Visit has no String constructor

SPRING MVC CONTROLLER TESTING (when class has @Controller or ends with 'Controller'):
- Use @ExtendWith(MockitoExtension.class) at class level
- Use @Mock for injected dependencies (repositories, services)
- Use @InjectMocks for the controller under test
- Setup mock behavior with when().thenReturn() BEFORE calling controller methods
- Use mock(BindingResult.class), mock(Model.class), mock(RedirectAttributes.class) for Spring MVC types
- For Page<T> results, use mock(Page.class) and configure isEmpty(), getTotalElements(), iterator()
- Verify interactions with verify(mockObject).method()

ONLY use method signatures that exist in the source code you are given.

You must output ONLY valid JSON - no markdown, no explanations."""

    # Alias for backward compatibility
    SYSTEM_PROMPT = SYSTEM_PROMPT_PYTHON

    @classmethod
    def get_system_prompt(cls, language: str = "python") -> str:
        """Get the appropriate system prompt for the language."""
        if language == "java":
            return cls.SYSTEM_PROMPT_JAVA
        return cls.SYSTEM_PROMPT_PYTHON

    # Primitive types that can have literal values in tests
    PRIMITIVE_TYPES = {'str', 'int', 'float', 'bool', 'None', 'list', 'dict', 'tuple', 'set'}

    # Types that should always use Mock()
    COMPLEX_TYPE_PATTERNS = {
        'Request', 'Response', 'Session', 'Connection', 'Client', 'Server',
        'Loader', 'Handler', 'Manager', 'Service', 'Controller', 'Provider',
        'Reader', 'Writer', 'Stream', 'Socket', 'Database', 'Model'
    }

    @classmethod
    def _get_mock_guidance(cls, func: FunctionContext) -> str:
        """Generate guidance on how to mock complex parameters."""
        guidance_lines = []
        needs_mock_import = False

        for param in func.parameters:
            if not param.type_hint:
                continue

            type_hint = param.type_hint

            # Check if it's a complex type that needs mocking
            is_complex = any(pattern in type_hint for pattern in cls.COMPLEX_TYPE_PATTERNS)
            is_primitive = any(prim in type_hint.lower() for prim in cls.PRIMITIVE_TYPES)

            if is_complex and not is_primitive:
                needs_mock_import = True
                guidance_lines.append(
                    f'   - {param.name}: Use Mock(spec={type_hint}) - do NOT invent fake classes'
                )

        if not guidance_lines:
            return ""

        header = """
⚠️ MOCK USAGE REQUIRED (DO NOT HALLUCINATE CLASSES):
   For complex parameter types, use unittest.mock.Mock:"""

        mock_import = '\n   - Add "from unittest.mock import Mock" to imports_needed' if needs_mock_import else ""

        return header + "\n" + "\n".join(guidance_lines) + mock_import

    @classmethod
    def _get_fixture_patterns(cls, func: FunctionContext) -> str:
        """Generate fixture patterns based on detected requirements."""
        fixtures = []

        # Flask fixtures
        if 'flask_app' in func.requires_context:
            fixtures.append('''
FIXTURE PATTERN - Flask App:
```python
@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def app_context(app):
    with app.app_context():
        yield
```
Use fixture name "app_context" in your test.''')

        # Request context
        if 'request_context' in func.requires_context:
            fixtures.append('''
FIXTURE PATTERN - Request Context:
```python
@pytest.fixture
def request_context(app):
    with app.test_request_context():
        yield
```''')

        # Complex parameter mocking
        for param in func.parameters:
            if param.type_hint and any(p in param.type_hint for p in cls.COMPLEX_TYPE_PATTERNS):
                fixtures.append(f'''
FIXTURE PATTERN - {param.name}:
```python
@pytest.fixture
def mock_{param.name}():
    mock = Mock(spec={param.type_hint})
    # Configure mock attributes as needed
    mock.some_attr = "value"
    return mock
```
Reference as "mock_{param.name}" in fixtures_needed.''')

        if not fixtures:
            return ""

        return "\n".join(["AVAILABLE FIXTURE PATTERNS:"] + fixtures)

    @staticmethod
    def _get_safe_imports(func: FunctionContext) -> str:
        """List imports that are safe to use (actually exist)."""
        safe_imports = [
            "pytest",
            "unittest.mock.Mock",
            "unittest.mock.MagicMock",
            "unittest.mock.patch",
            "unittest.mock.PropertyMock",
        ]

        # Add framework-specific imports
        if 'flask' in func.framework_hints:
            safe_imports.extend([
                "flask.Flask",
                "flask.testing.FlaskClient",
            ])

        if 'werkzeug' in func.framework_hints:
            safe_imports.extend([
                "werkzeug.test.Client",
                "werkzeug.wrappers.Response",
            ])

        return f"""
SAFE IMPORTS (only use these, do not invent others):
{chr(10).join(f'   - {imp}' for imp in safe_imports)}

DO NOT USE: MockLoader, FakeRequest, TestResponse, or any invented class names."""

    @staticmethod
    def _build_semantic_hints(func: FunctionContext) -> str:
        """Build semantic hints section based on detected function characteristics."""
        hints = []

        # Generator/Iterator hint
        if func.is_generator:
            hints.append("""⚠️ GENERATOR FUNCTION:
   - This function uses 'yield' and returns a generator/iterator
   - Tests should consume the generator with list() before asserting
   - Expected 'returns' should be a list of yielded values
   - Example: result = list(func()); assert result == [item1, item2]""")
        elif func.return_semantics == ReturnSemantics.ITERATOR:
            hints.append("""⚠️ ITERATOR RETURN:
   - This function returns an Iterator/Iterable type
   - Tests should convert to list before comparing: list(result)
   - Expected 'returns' should be a list""")

        # Flask context hint
        if 'flask_app' in func.requires_context:
            hints.append("""⚠️ FLASK CONTEXT REQUIRED:
   - This function uses Flask globals (current_app, request, g, session)
   - Tests MUST run inside 'with app.app_context():'
   - Add fixture: 'flask_app' to fixtures_needed
   - Test setup should include: app = Flask(__name__)""")

        if 'request_context' in func.requires_context:
            hints.append("""⚠️ REQUEST CONTEXT REQUIRED:
   - This function requires an active Flask request context
   - Tests should use: 'with app.test_request_context():'""")

        # Side effect hint
        if func.mutates_args:
            hints.append("""⚠️ SIDE EFFECT FUNCTION:
   - This function modifies its arguments (does not just return a value)
   - Return value may be None - test the MUTATION instead
   - Verify object state BEFORE and AFTER the call
   - Example: assert obj.attr_changed == new_value""")

        # Database session hint
        if 'db_session' in func.requires_context:
            hints.append("""⚠️ DATABASE SESSION REQUIRED:
   - This function requires a database session
   - Mock the session or use a test database
   - Add 'db_session' to fixtures_needed""")

        # Framework-specific hints
        if 'flask' in func.framework_hints:
            hints.append("""📦 FLASK FRAMEWORK:
   - Use Flask test patterns and fixtures
   - Consider using flask.testing.FlaskClient for request testing""")

        if 'werkzeug' in func.framework_hints:
            hints.append("""📦 WERKZEUG UTILITIES:
   - This uses Werkzeug classes (Request, Response, MultiDict)
   - Use werkzeug.test for creating test Request objects""")

        if not hints:
            return ""

        return "\n\n".join(["SEMANTIC HINTS (IMPORTANT - READ CAREFULLY):"] + hints)

    @classmethod
    def _detect_language(cls, file_path: str) -> str:
        """Detect the programming language from the file extension."""
        path = str(file_path).lower()
        if path.endswith('.java'):
            return 'java'
        elif path.endswith('.ts') or path.endswith('.tsx'):
            return 'typescript'
        elif path.endswith('.js') or path.endswith('.jsx'):
            return 'javascript'
        else:
            return 'python'

    @classmethod
    def build_function_prompt(
        cls,
        func: FunctionContext,
        patterns: ProjectTestPatterns | None = None,
        existing_tests: str | None = None
    ) -> str:
        """Build a prompt for generating test specs for a function."""

        # Detect language from file path
        language = cls._detect_language(str(func.location.file_path))

        # Build parameter info
        params_info = []
        for p in func.parameters:
            param_str = f"- {p.name}"
            if p.type_hint:
                param_str += f": {p.type_hint}"
            if p.default_value:
                param_str += f" = {p.default_value}"
            if p.is_optional:
                param_str += " (optional)"
            params_info.append(param_str)

        params_section = "\n".join(params_info) if params_info else "None"

        # Build pattern hints
        pattern_hints = ""
        if patterns:
            if patterns.fixtures:
                pattern_hints += f"\nAvailable fixtures: {', '.join(patterns.fixtures[:10])}"
            if patterns.common_mocks:
                pattern_hints += f"\nCommon mocks used: {', '.join(patterns.common_mocks[:5])}"

        # Build semantic hints
        semantic_hints = cls._build_semantic_hints(func)

        # Build anti-hallucination guidance (Options 1, 2, 3)
        mock_guidance = cls._get_mock_guidance(func)
        fixture_patterns = cls._get_fixture_patterns(func)
        safe_imports = cls._get_safe_imports(func)

        # Select relevant official examples (NEW - one-shot examples)
        official_examples = cls._select_examples(func, language)

        # Semantic metadata for the prompt
        semantic_metadata = f"""
SEMANTIC ANALYSIS:
- Is Generator: {func.is_generator}
- Return Semantics: {func.return_semantics.value}
- Requires Context: {', '.join(func.requires_context) if func.requires_context else 'None'}
- Mutates Arguments: {func.mutates_args}
- Framework: {', '.join(func.framework_hints) if func.framework_hints else 'None'}"""

        # Language-specific code fence
        code_lang = "java" if language == "java" else "python"

        prompt = f"""Analyze this {language.upper()} function and generate a test specification JSON.

FUNCTION TO TEST:
```{code_lang}
{func.source_code}
```

FUNCTION DETAILS:
- Name: {func.name}
- File: {func.location.file_path}
- Is async: {func.is_async}
- Is method: {func.is_method}
- Class: {func.class_name or 'N/A'}
- Return type: {func.return_type or 'Unknown'}
{semantic_metadata}

PARAMETERS:
{params_section}

FUNCTION CALLS (potential mock targets):
{', '.join(func.calls) if func.calls else 'None identified'}

{pattern_hints}

{semantic_hints}

{mock_guidance}

{fixture_patterns}

{safe_imports}

{official_examples}

{f'EXISTING TEST PATTERNS IN PROJECT:{chr(10)}{existing_tests}' if existing_tests else ''}

Generate a JSON test specification with:
1. 2-4 test cases covering:
   - At least 1 happy path test
   - At least 1 edge case or error handling test
2. Specific input values (not placeholders) - USE QUOTED STRINGS for string values
3. Expected outputs or exceptions
4. Any mocks needed
5. For generators: expected "returns" should be a LIST of values
6. For Flask functions: include "flask_app" in fixtures_needed

OUTPUT JSON SCHEMA:
{{
  "test_type": "unit_pure|unit_class|unit_mocked|edge_case|custom",
  "target_file": "{func.location.file_path}",
  "target_name": "{func.name}",
  "target_class": {f'"{func.class_name}"' if func.class_name else 'null'},
  "language": "{language}",
  "test_cases": [
    {{
      "name": "descriptive_test_name",
      "category": "happy_path|edge_case|error_handling|boundary|null_empty",
      "description": "What this test verifies",
      "inputs": {{"param1": "value1"}},
      "expected": {{
        "returns": "expected_value",  // OR for generators: ["item1", "item2"]
        "raises": "ExceptionType",
        "raises_message": "partial message"
      }},
      "mocks": [
        {{
          "target": "module.function",
          "return_value": "mocked_value"
        }}
      ]
    }}
  ],
  "fixtures_needed": ["flask_app"],  // Add if Flask context required
  "imports_needed": ["from module import thing"],
  "complexity_score": 1-10,
  "requires_custom_generation": false,
  "is_generator": {str(func.is_generator).lower()},
  "requires_context": {func.requires_context if func.requires_context else []},
  "framework_hints": {func.framework_hints if func.framework_hints else []}
}}

Output ONLY the JSON, no other text:"""

        return prompt

    @staticmethod
    def build_class_prompt(
        cls: ClassContext,
        patterns: ProjectTestPatterns | None = None
    ) -> str:
        """Build a prompt for generating test specs for a class."""

        methods_info = []
        for method in cls.methods:
            params = ", ".join(
                f"{p.name}: {p.type_hint or 'Any'}"
                for p in method.parameters
            )
            methods_info.append(f"- {method.name}({params}) -> {method.return_type or 'Unknown'}")

        methods_section = "\n".join(methods_info) if methods_info else "No methods"

        prompt = f"""Analyze this class and generate test specifications for its key methods.

CLASS TO TEST:
```python
{cls.docstring or 'No docstring'}

class {cls.name}({', '.join(cls.base_classes) if cls.base_classes else ''}):
    ...
```

CLASS DETAILS:
- Name: {cls.name}
- File: {cls.location.file_path}
- Base classes: {', '.join(cls.base_classes) if cls.base_classes else 'None'}

METHODS:
{methods_section}

Focus on the most important methods that need testing.
Generate a JSON array of test specifications, one per method.

Output ONLY the JSON array, no other text:"""

        return prompt

    @staticmethod
    def build_coverage_improvement_prompt(
        func: FunctionContext,
        current_coverage: float,
        missing_lines: list[int],
        existing_test_spec: dict
    ) -> str:
        """Build a prompt to improve test coverage."""

        prompt = f"""The existing tests for this function only achieve {current_coverage:.1f}% coverage.

FUNCTION:
```python
{func.source_code}
```

MISSING COVERAGE (lines {missing_lines}):
These lines are not covered by current tests.

CURRENT TEST SPECIFICATION:
{existing_test_spec}

Generate ADDITIONAL test cases to cover the missing lines.
Focus on:
1. Branches not taken (if/else, try/except)
2. Edge cases that trigger different code paths
3. Error conditions

Output a JSON object with ONLY the new test_cases to add:
{{
  "additional_test_cases": [
    {{
      "name": "...",
      "category": "...",
      "description": "Covers lines X-Y",
      "inputs": {{}},
      "expected": {{}}
    }}
  ]
}}

Output ONLY the JSON, no other text:"""

        return prompt

    @staticmethod
    def build_custom_generation_prompt(
        func: FunctionContext,
        reason: str
    ) -> str:
        """Build a prompt for custom test generation (fallback when templates don't fit)."""

        prompt = f"""This function requires custom test generation because: {reason}

FUNCTION:
```python
{func.source_code}
```

FUNCTION DETAILS:
- Name: {func.name}
- Is async: {func.is_async}
- Parameters: {[p.name for p in func.parameters]}

Generate complete pytest test code (not a specification).
Include:
1. All necessary imports
2. Any required fixtures
3. 3-5 test cases with descriptive names
4. Clear assertions

Output ONLY the Python test code, no explanations:
```python
"""

        return prompt
