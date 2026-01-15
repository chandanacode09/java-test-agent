"""
Few-shot example: Basic Entity with getters/setters.

This example demonstrates testing a simple JPA entity with:
- Getter/setter methods
- Null handling
- Basic property storage
"""

ENTITY_BASIC_EXAMPLE = '''
// =============================================================================
// PATTERN: Basic JPA Entity with Getters/Setters
// =============================================================================
// KEY CONCEPTS:
//   - Test each getter returns what setter stored
//   - Test null handling for nullable fields
//   - Test default values if any
//   - Use @BeforeEach to create fresh instance
// =============================================================================

package org.example.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

class PersonTest {

    private Person person;

    @BeforeEach
    void setUp() {
        // PATTERN: Create fresh instance before each test
        person = new Person();
    }

    // =========================================================================
    // Getter/Setter Tests
    // =========================================================================

    @Test
    @DisplayName("setFirstName and getFirstName should store and retrieve value")
    void testFirstName_setAndGet() {
        // PATTERN: Set value, then verify getter returns it
        person.setFirstName("John");

        assertEquals("John", person.getFirstName());
    }

    @Test
    @DisplayName("setLastName and getLastName should store and retrieve value")
    void testLastName_setAndGet() {
        person.setLastName("Doe");

        assertEquals("Doe", person.getLastName());
    }

    @Test
    @DisplayName("getFirstName should return null when not set")
    void testFirstName_defaultNull() {
        // PATTERN: Test default/initial state
        assertNull(person.getFirstName());
    }

    @Test
    @DisplayName("setFirstName should accept null")
    void testFirstName_acceptsNull() {
        // PATTERN: Test null is accepted for nullable fields
        person.setFirstName("John");
        person.setFirstName(null);

        assertNull(person.getFirstName());
    }
}
'''

ENTITY_BASIC_META = {
    "component_type": "entity_basic",
    "patterns": ["getter_setter", "null_handling", "default_values"],
    "annotations": ["@Entity", "@MappedSuperclass"],
    "key_imports": [
        "org.junit.jupiter.api.BeforeEach",
        "org.junit.jupiter.api.Test",
        "org.junit.jupiter.api.DisplayName",
        "static org.junit.jupiter.api.Assertions.*",
    ],
}
