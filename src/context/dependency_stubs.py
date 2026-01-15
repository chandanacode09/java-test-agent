"""
Dependency stubs for common Java APIs.

Provides method signatures for common libraries that the LLM needs to know about
when generating tests, but which aren't present in the parsed source code.
"""

from typing import Dict, List

# Java Standard Library stubs
JAVA_STDLIB_STUBS: Dict[str, Dict[str, List[str]]] = {
    "List": {
        "package": "java.util",
        "methods": [
            "add(E element)",
            "add(int index, E element)",
            "addAll(Collection<? extends E> c)",
            "get(int index)",
            "set(int index, E element)",
            "remove(int index)",
            "remove(Object o)",
            "size()",
            "isEmpty()",
            "contains(Object o)",
            "clear()",
            "indexOf(Object o)",
            "toArray()",
            "stream()",
            "forEach(Consumer<? super E> action)",
            "iterator()",
            "subList(int fromIndex, int toIndex)",
        ],
    },
    "ArrayList": {
        "package": "java.util",
        "extends": "List",
        "methods": [
            "ArrayList()",
            "ArrayList(int initialCapacity)",
            "ArrayList(Collection<? extends E> c)",
            "trimToSize()",
            "ensureCapacity(int minCapacity)",
        ],
    },
    "Set": {
        "package": "java.util",
        "methods": [
            "add(E element)",
            "remove(Object o)",
            "contains(Object o)",
            "size()",
            "isEmpty()",
            "clear()",
            "iterator()",
            "toArray()",
            "stream()",
        ],
    },
    "HashSet": {
        "package": "java.util",
        "extends": "Set",
        "methods": [
            "HashSet()",
            "HashSet(Collection<? extends E> c)",
            "HashSet(int initialCapacity)",
        ],
    },
    "Map": {
        "package": "java.util",
        "methods": [
            "put(K key, V value)",
            "get(Object key)",
            "remove(Object key)",
            "containsKey(Object key)",
            "containsValue(Object value)",
            "size()",
            "isEmpty()",
            "clear()",
            "keySet()",
            "values()",
            "entrySet()",
            "getOrDefault(Object key, V defaultValue)",
            "putIfAbsent(K key, V value)",
            "forEach(BiConsumer<? super K, ? super V> action)",
        ],
    },
    "HashMap": {
        "package": "java.util",
        "extends": "Map",
        "methods": [
            "HashMap()",
            "HashMap(int initialCapacity)",
            "HashMap(Map<? extends K, ? extends V> m)",
        ],
    },
    "Optional": {
        "package": "java.util",
        "methods": [
            "of(T value)",
            "ofNullable(T value)",
            "empty()",
            "get()",
            "isPresent()",
            "isEmpty()",
            "ifPresent(Consumer<? super T> action)",
            "orElse(T other)",
            "orElseGet(Supplier<? extends T> supplier)",
            "orElseThrow()",
            "orElseThrow(Supplier<? extends X> exceptionSupplier)",
            "map(Function<? super T, ? extends U> mapper)",
            "flatMap(Function<? super T, ? extends Optional<? extends U>> mapper)",
            "filter(Predicate<? super T> predicate)",
            "stream()",
        ],
    },
    "Stream": {
        "package": "java.util.stream",
        "methods": [
            "filter(Predicate<? super T> predicate)",
            "map(Function<? super T, ? extends R> mapper)",
            "flatMap(Function<? super T, ? extends Stream<? extends R>> mapper)",
            "distinct()",
            "sorted()",
            "sorted(Comparator<? super T> comparator)",
            "peek(Consumer<? super T> action)",
            "limit(long maxSize)",
            "skip(long n)",
            "forEach(Consumer<? super T> action)",
            "toArray()",
            "reduce(BinaryOperator<T> accumulator)",
            "collect(Collector<? super T, A, R> collector)",
            "count()",
            "findFirst()",
            "findAny()",
            "anyMatch(Predicate<? super T> predicate)",
            "allMatch(Predicate<? super T> predicate)",
            "noneMatch(Predicate<? super T> predicate)",
        ],
    },
    "Collectors": {
        "package": "java.util.stream",
        "methods": [
            "toList()",
            "toSet()",
            "toMap(Function keyMapper, Function valueMapper)",
            "joining()",
            "joining(CharSequence delimiter)",
            "groupingBy(Function classifier)",
            "counting()",
            "summingInt(ToIntFunction mapper)",
            "averagingDouble(ToDoubleFunction mapper)",
        ],
    },
    "String": {
        "package": "java.lang",
        "methods": [
            "length()",
            "charAt(int index)",
            "substring(int beginIndex)",
            "substring(int beginIndex, int endIndex)",
            "contains(CharSequence s)",
            "startsWith(String prefix)",
            "endsWith(String suffix)",
            "indexOf(String str)",
            "lastIndexOf(String str)",
            "replace(CharSequence target, CharSequence replacement)",
            "replaceAll(String regex, String replacement)",
            "split(String regex)",
            "trim()",
            "strip()",
            "toLowerCase()",
            "toUpperCase()",
            "equals(Object obj)",
            "equalsIgnoreCase(String anotherString)",
            "isEmpty()",
            "isBlank()",
            "format(String format, Object... args)",
            "valueOf(Object obj)",
            "join(CharSequence delimiter, CharSequence... elements)",
        ],
    },
    "Integer": {
        "package": "java.lang",
        "methods": [
            "parseInt(String s)",
            "valueOf(int i)",
            "valueOf(String s)",
            "toString(int i)",
            "intValue()",
            "compareTo(Integer anotherInteger)",
            "equals(Object obj)",
            "hashCode()",
            "MAX_VALUE",
            "MIN_VALUE",
        ],
    },
    "Long": {
        "package": "java.lang",
        "methods": [
            "parseLong(String s)",
            "valueOf(long l)",
            "valueOf(String s)",
            "toString(long l)",
            "longValue()",
            "compareTo(Long anotherLong)",
            "equals(Object obj)",
            "hashCode()",
            "MAX_VALUE",
            "MIN_VALUE",
        ],
    },
    "LocalDate": {
        "package": "java.time",
        "methods": [
            "now()",
            "of(int year, int month, int dayOfMonth)",
            "of(int year, Month month, int dayOfMonth)",
            "parse(CharSequence text)",
            "getYear()",
            "getMonth()",
            "getMonthValue()",
            "getDayOfMonth()",
            "getDayOfWeek()",
            "getDayOfYear()",
            "plusDays(long daysToAdd)",
            "plusWeeks(long weeksToAdd)",
            "plusMonths(long monthsToAdd)",
            "plusYears(long yearsToAdd)",
            "minusDays(long daysToSubtract)",
            "minusMonths(long monthsToSubtract)",
            "isBefore(ChronoLocalDate other)",
            "isAfter(ChronoLocalDate other)",
            "isEqual(ChronoLocalDate other)",
            "format(DateTimeFormatter formatter)",
        ],
    },
    "LocalDateTime": {
        "package": "java.time",
        "methods": [
            "now()",
            "of(int year, int month, int dayOfMonth, int hour, int minute)",
            "of(LocalDate date, LocalTime time)",
            "parse(CharSequence text)",
            "toLocalDate()",
            "toLocalTime()",
            "plusHours(long hours)",
            "plusMinutes(long minutes)",
            "plusSeconds(long seconds)",
            "format(DateTimeFormatter formatter)",
        ],
    },
}

# Spring Data JPA stubs
SPRING_DATA_STUBS: Dict[str, Dict[str, List[str]]] = {
    "JpaRepository": {
        "package": "org.springframework.data.jpa.repository",
        "extends": "PagingAndSortingRepository, QueryByExampleExecutor",
        "methods": [
            "save(S entity)",
            "saveAll(Iterable<S> entities)",
            "findById(ID id)",
            "existsById(ID id)",
            "findAll()",
            "findAll(Pageable pageable)",
            "findAll(Sort sort)",
            "findAllById(Iterable<ID> ids)",
            "count()",
            "deleteById(ID id)",
            "delete(T entity)",
            "deleteAllById(Iterable<? extends ID> ids)",
            "deleteAll(Iterable<? extends T> entities)",
            "deleteAll()",
            "flush()",
            "saveAndFlush(S entity)",
            "saveAllAndFlush(Iterable<S> entities)",
            "deleteAllInBatch(Iterable<T> entities)",
            "deleteAllByIdInBatch(Iterable<ID> ids)",
            "deleteAllInBatch()",
            "getById(ID id)",
            "getReferenceById(ID id)",
            "findAll(Example<S> example)",
            "findAll(Example<S> example, Sort sort)",
        ],
    },
    "CrudRepository": {
        "package": "org.springframework.data.repository",
        "methods": [
            "save(S entity)",
            "saveAll(Iterable<S> entities)",
            "findById(ID id)",
            "existsById(ID id)",
            "findAll()",
            "findAllById(Iterable<ID> ids)",
            "count()",
            "deleteById(ID id)",
            "delete(T entity)",
            "deleteAllById(Iterable<? extends ID> ids)",
            "deleteAll(Iterable<? extends T> entities)",
            "deleteAll()",
        ],
    },
    "PagingAndSortingRepository": {
        "package": "org.springframework.data.repository",
        "extends": "CrudRepository",
        "methods": [
            "findAll(Sort sort)",
            "findAll(Pageable pageable)",
        ],
    },
    "Page": {
        "package": "org.springframework.data.domain",
        "methods": [
            "getContent()",
            "getTotalElements()",
            "getTotalPages()",
            "getNumber()",
            "getSize()",
            "getNumberOfElements()",
            "hasContent()",
            "hasNext()",
            "hasPrevious()",
            "isFirst()",
            "isLast()",
            "nextPageable()",
            "previousPageable()",
            "getSort()",
            "map(Function<? super T, ? extends U> converter)",
        ],
    },
    "Pageable": {
        "package": "org.springframework.data.domain",
        "methods": [
            "getPageNumber()",
            "getPageSize()",
            "getOffset()",
            "getSort()",
            "next()",
            "previousOrFirst()",
            "first()",
            "withPage(int pageNumber)",
            "hasPrevious()",
            "toOptional()",
            "ofSize(int pageSize)",
            "unpaged()",
        ],
    },
    "PageRequest": {
        "package": "org.springframework.data.domain",
        "methods": [
            "of(int page, int size)",
            "of(int page, int size, Sort sort)",
            "of(int page, int size, Sort.Direction direction, String... properties)",
            "ofSize(int pageSize)",
        ],
    },
    "Sort": {
        "package": "org.springframework.data.domain",
        "methods": [
            "by(String... properties)",
            "by(Sort.Direction direction, String... properties)",
            "by(Sort.Order... orders)",
            "ascending()",
            "descending()",
            "and(Sort sort)",
            "unsorted()",
            "isSorted()",
            "isUnsorted()",
            "iterator()",
        ],
    },
}

# Spring Framework stubs
SPRING_FRAMEWORK_STUBS: Dict[str, Dict[str, List[str]]] = {
    "Model": {
        "package": "org.springframework.ui",
        "methods": [
            "addAttribute(String attributeName, Object attributeValue)",
            "addAttribute(Object attributeValue)",
            "addAllAttributes(Collection<?> attributeValues)",
            "addAllAttributes(Map<String, ?> attributes)",
            "mergeAttributes(Map<String, ?> attributes)",
            "containsAttribute(String attributeName)",
            "getAttribute(String attributeName)",
            "asMap()",
        ],
    },
    "ModelAndView": {
        "package": "org.springframework.web.servlet",
        "methods": [
            "ModelAndView()",
            "ModelAndView(String viewName)",
            "ModelAndView(String viewName, Map<String, ?> model)",
            "setViewName(String viewName)",
            "getViewName()",
            "setView(View view)",
            "getView()",
            "addObject(String attributeName, Object attributeValue)",
            "addObject(Object attributeValue)",
            "addAllObjects(Map<String, ?> modelMap)",
            "getModel()",
            "getModelMap()",
            "getStatus()",
            "setStatus(HttpStatus status)",
        ],
    },
    "BindingResult": {
        "package": "org.springframework.validation",
        "methods": [
            "hasErrors()",
            "hasGlobalErrors()",
            "hasFieldErrors()",
            "hasFieldErrors(String field)",
            "getErrorCount()",
            "getGlobalErrorCount()",
            "getFieldErrorCount()",
            "getFieldErrorCount(String field)",
            "getAllErrors()",
            "getGlobalErrors()",
            "getFieldErrors()",
            "getFieldErrors(String field)",
            "getFieldError()",
            "getFieldError(String field)",
            "getFieldValue(String field)",
            "rejectValue(String field, String errorCode)",
            "rejectValue(String field, String errorCode, String defaultMessage)",
            "reject(String errorCode)",
            "reject(String errorCode, String defaultMessage)",
        ],
    },
    "Errors": {
        "package": "org.springframework.validation",
        "methods": [
            "hasErrors()",
            "hasGlobalErrors()",
            "hasFieldErrors()",
            "rejectValue(String field, String errorCode)",
            "rejectValue(String field, String errorCode, String defaultMessage)",
            "reject(String errorCode)",
            "reject(String errorCode, String defaultMessage)",
        ],
    },
    "RedirectAttributes": {
        "package": "org.springframework.web.servlet.mvc.support",
        "methods": [
            "addFlashAttribute(String attributeName, Object attributeValue)",
            "addFlashAttribute(Object attributeValue)",
            "addAttribute(String attributeName, Object attributeValue)",
            "addAttribute(Object attributeValue)",
            "addAllAttributes(Map<String, ?> attributes)",
            "mergeAttributes(Map<String, ?> attributes)",
            "getFlashAttributes()",
        ],
    },
    "Formatter": {
        "package": "org.springframework.format",
        "methods": [
            "print(T object, Locale locale)",
            "parse(String text, Locale locale)",
        ],
    },
    "Validator": {
        "package": "org.springframework.validation",
        "methods": [
            "supports(Class<?> clazz)",
            "validate(Object target, Errors errors)",
        ],
    },
}

# Testing framework stubs
TESTING_STUBS: Dict[str, Dict[str, List[str]]] = {
    "Assertions": {
        "package": "org.junit.jupiter.api",
        "methods": [
            "assertEquals(Object expected, Object actual)",
            "assertEquals(Object expected, Object actual, String message)",
            "assertNotEquals(Object unexpected, Object actual)",
            "assertTrue(boolean condition)",
            "assertTrue(boolean condition, String message)",
            "assertFalse(boolean condition)",
            "assertFalse(boolean condition, String message)",
            "assertNull(Object actual)",
            "assertNull(Object actual, String message)",
            "assertNotNull(Object actual)",
            "assertNotNull(Object actual, String message)",
            "assertSame(Object expected, Object actual)",
            "assertNotSame(Object unexpected, Object actual)",
            "assertThrows(Class<T> expectedType, Executable executable)",
            "assertDoesNotThrow(Executable executable)",
            "assertAll(Executable... executables)",
            "assertArrayEquals(Object[] expected, Object[] actual)",
            "assertIterableEquals(Iterable<?> expected, Iterable<?> actual)",
            "fail(String message)",
        ],
    },
    "Mockito": {
        "package": "org.mockito",
        "methods": [
            "mock(Class<T> classToMock)",
            "spy(T object)",
            "when(T methodCall)",
            "verify(T mock)",
            "verify(T mock, VerificationMode mode)",
            "any()",
            "any(Class<T> type)",
            "anyString()",
            "anyInt()",
            "anyLong()",
            "anyBoolean()",
            "anyList()",
            "anyMap()",
            "anySet()",
            "eq(T value)",
            "isNull()",
            "isNotNull()",
            "never()",
            "times(int wantedNumberOfInvocations)",
            "atLeast(int minNumberOfInvocations)",
            "atMost(int maxNumberOfInvocations)",
            "atLeastOnce()",
            "reset(T... mocks)",
            "doReturn(Object toBeReturned)",
            "doThrow(Throwable... toBeThrown)",
            "doNothing()",
            "doAnswer(Answer answer)",
        ],
    },
}

# All stubs combined
ALL_STUBS: Dict[str, Dict[str, List[str]]] = {
    **JAVA_STDLIB_STUBS,
    **SPRING_DATA_STUBS,
    **SPRING_FRAMEWORK_STUBS,
    **TESTING_STUBS,
}


def get_methods_for_class(class_name: str) -> List[str]:
    """Get all methods available for a class, including inherited ones."""
    methods = []

    # Extract base class name from generic (e.g., "JpaRepository<Owner, Integer>" -> "JpaRepository")
    base_name = class_name.split("<")[0].strip()

    if base_name in ALL_STUBS:
        stub = ALL_STUBS[base_name]
        methods.extend(stub.get("methods", []))

        # Also include inherited methods
        if "extends" in stub:
            for parent in stub["extends"].split(","):
                parent = parent.strip()
                methods.extend(get_methods_for_class(parent))

    return methods


def get_stub_context_for_classes(class_names: List[str]) -> str:
    """Generate API context string for a list of base classes."""
    lines = []

    for class_name in class_names:
        # Extract base class name from generic
        base_name = class_name.split("<")[0].strip()

        if base_name in ALL_STUBS:
            stub = ALL_STUBS[base_name]
            package = stub.get("package", "")
            extends = stub.get("extends", "")

            lines.append(f"\n// {base_name} ({package})")
            if extends:
                lines.append(f"// extends: {extends}")

            for method in stub.get("methods", []):
                lines.append(f"//   {method}")

    return "\n".join(lines) if lines else ""


def get_all_known_classes() -> List[str]:
    """Return list of all classes we have stubs for."""
    return list(ALL_STUBS.keys())
