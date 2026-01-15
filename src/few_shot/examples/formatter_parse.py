"""
Few-shot example: Spring Formatter with parse/print.

This example demonstrates testing a Spring Formatter:
- Testing parse() method with valid and invalid inputs
- Testing print() method
- Mocking repository dependencies
- Handling ParseException
"""

FORMATTER_EXAMPLE = '''
// =============================================================================
// PATTERN: Spring Formatter (parse/print)
// =============================================================================
// KEY CONCEPTS:
//   - Formatter has parse(String, Locale) and print(Object, Locale) methods
//   - parse() may throw ParseException for invalid input
//   - Often has repository dependency to look up entities
//   - Use @Mock for repository, @InjectMocks for formatter
// =============================================================================

package org.example.formatter;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.text.ParseException;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class PetTypeFormatterTest {

    @Mock
    private OwnerRepository ownerRepository;

    @InjectMocks
    private PetTypeFormatter formatter;

    // =========================================================================
    // parse() Tests - String to Object
    // =========================================================================

    @Test
    @DisplayName("parse should return PetType when name matches")
    void parse_withValidName_returnsPetType() throws ParseException {
        // PATTERN: Setup repository to return collection with matching item
        PetType dogType = new PetType();
        dogType.setName("dog");

        Collection<PetType> petTypes = List.of(dogType);
        when(ownerRepository.findPetTypes()).thenReturn(petTypes);

        // PATTERN: parse takes String and Locale
        PetType result = formatter.parse("dog", Locale.US);

        assertNotNull(result);
        assertEquals("dog", result.getName());
    }

    @Test
    @DisplayName("parse should throw ParseException when name not found")
    void parse_withInvalidName_throwsParseException() {
        // PATTERN: Empty or non-matching collection
        when(ownerRepository.findPetTypes()).thenReturn(List.of());

        // PATTERN: parse throws ParseException for invalid input
        assertThrows(
            ParseException.class,
            () -> formatter.parse("unknown", Locale.US)
        );
    }

    @Test
    @DisplayName("parse should be case-insensitive")
    void parse_withDifferentCase_findsPetType() throws ParseException {
        // PATTERN: Many formatters do case-insensitive matching
        PetType catType = new PetType();
        catType.setName("cat");

        when(ownerRepository.findPetTypes()).thenReturn(List.of(catType));

        // Try uppercase
        PetType result = formatter.parse("CAT", Locale.US);

        assertNotNull(result);
        assertEquals("cat", result.getName());
    }

    // =========================================================================
    // print() Tests - Object to String
    // =========================================================================

    @Test
    @DisplayName("print should return pet type name")
    void print_withPetType_returnsName() {
        PetType petType = new PetType();
        petType.setName("bird");

        // PATTERN: print takes Object and Locale, returns String
        String result = formatter.print(petType, Locale.US);

        assertEquals("bird", result);
    }

    @Test
    @DisplayName("print should handle null gracefully")
    void print_withNull_returnsEmptyOrNull() {
        // PATTERN: print may return empty string or handle null
        String result = formatter.print(null, Locale.US);

        // Depending on implementation, may be null or empty
        assertTrue(result == null || result.isEmpty());
    }
}
'''

FORMATTER_META = {
    "component_type": "formatter",
    "patterns": ["parse_method", "print_method", "parse_exception", "repository_dependency"],
    "annotations": ["@Component"],
    "key_imports": [
        "java.text.ParseException",
        "java.util.Locale",
        "java.util.Collection",
        "java.util.List",
    ],
    "critical_pattern": "parse() throws ParseException, print() returns String",
    "common_mistake": "Forgetting that parse takes Locale parameter",
}
