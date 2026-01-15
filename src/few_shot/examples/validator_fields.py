"""
Few-shot example: Spring Validator with Multiple Required Fields.

This example demonstrates testing a Spring Validator:
- Testing supports() method for type checking
- Testing each required field independently
- Using BeanPropertyBindingResult for Errors
- Setting ALL required fields for valid test case
"""

VALIDATOR_EXAMPLE = '''
// =============================================================================
// PATTERN: Spring Validator with Multiple Required Fields
// =============================================================================
// KEY CONCEPTS:
//   - Validator checks multiple fields - test each one independently
//   - Use BeanPropertyBindingResult as Errors implementation
//   - For valid test: set ALL required fields, not just one!
//   - Test supports() returns true for target class and subclasses
// =============================================================================

package org.example.validation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.springframework.validation.BeanPropertyBindingResult;
import org.springframework.validation.Errors;
import java.time.LocalDate;

import static org.junit.jupiter.api.Assertions.*;

class PetValidatorTest {

    private PetValidator validator;

    @BeforeEach
    void setUp() {
        validator = new PetValidator();
    }

    // =========================================================================
    // supports() Method Tests
    // =========================================================================

    @Test
    @DisplayName("supports should return true for Pet class")
    void supports_withPetClass_returnsTrue() {
        assertTrue(validator.supports(Pet.class));
    }

    @Test
    @DisplayName("supports should return false for other classes")
    void supports_withOtherClass_returnsFalse() {
        assertFalse(validator.supports(Owner.class));
        assertFalse(validator.supports(String.class));
    }

    // =========================================================================
    // validate() - Valid Object (ALL fields set)
    // =========================================================================

    @Test
    @DisplayName("validate should pass with all valid fields")
    void validate_withAllValidFields_hasNoErrors() {
        // CRITICAL PATTERN: Set ALL required fields for valid test!
        // PetValidator checks: name, type, birthDate
        Pet pet = new Pet();
        pet.setName("Buddy");           // Required field 1
        pet.setType(new PetType());     // Required field 2
        pet.setBirthDate(LocalDate.now());  // Required field 3

        // PATTERN: Use BeanPropertyBindingResult as Errors implementation
        Errors errors = new BeanPropertyBindingResult(pet, "pet");

        validator.validate(pet, errors);

        // Should have no errors when all fields are valid
        assertFalse(errors.hasErrors());
    }

    // =========================================================================
    // validate() - Individual Field Validation
    // =========================================================================

    @Test
    @DisplayName("validate should reject empty name")
    void validate_withEmptyName_rejectsName() {
        Pet pet = new Pet();
        pet.setName("");                    // Invalid: empty
        pet.setType(new PetType());         // Valid
        pet.setBirthDate(LocalDate.now());  // Valid

        Errors errors = new BeanPropertyBindingResult(pet, "pet");

        validator.validate(pet, errors);

        // PATTERN: Check specific field has error
        assertTrue(errors.hasFieldErrors("name"));
    }

    @Test
    @DisplayName("validate should reject null type for new pet")
    void validate_withNullType_rejectsType() {
        Pet pet = new Pet();
        pet.setName("Buddy");               // Valid
        pet.setType(null);                  // Invalid: null
        pet.setBirthDate(LocalDate.now());  // Valid
        // Note: pet.isNew() = true because no ID set

        Errors errors = new BeanPropertyBindingResult(pet, "pet");

        validator.validate(pet, errors);

        assertTrue(errors.hasFieldErrors("type"));
    }

    @Test
    @DisplayName("validate should NOT reject null type for existing pet")
    void validate_existingPetWithNullType_noTypeError() {
        // PATTERN: Some validators have conditional logic based on isNew()
        Pet pet = new Pet();
        pet.setId(1);                       // Has ID = existing pet, isNew() = false
        pet.setName("Buddy");               // Valid
        pet.setType(null);                  // Null type OK for existing pet
        pet.setBirthDate(LocalDate.now());  // Valid

        Errors errors = new BeanPropertyBindingResult(pet, "pet");

        validator.validate(pet, errors);

        // Type not required for existing pets
        assertFalse(errors.hasFieldErrors("type"));
    }

    @Test
    @DisplayName("validate should reject null birthDate")
    void validate_withNullBirthDate_rejectsBirthDate() {
        Pet pet = new Pet();
        pet.setName("Buddy");               // Valid
        pet.setType(new PetType());         // Valid
        pet.setBirthDate(null);             // Invalid: null

        Errors errors = new BeanPropertyBindingResult(pet, "pet");

        validator.validate(pet, errors);

        assertTrue(errors.hasFieldErrors("birthDate"));
    }
}
'''

VALIDATOR_META = {
    "component_type": "validator",
    "patterns": ["validate_method", "supports_method", "bean_property_binding_result", "field_errors"],
    "annotations": [],
    "key_imports": [
        "org.springframework.validation.BeanPropertyBindingResult",
        "org.springframework.validation.Errors",
        "java.time.LocalDate",
    ],
    "critical_pattern": "Set ALL required fields in the valid test case",
    "common_mistake": "Only setting one field and expecting no errors - validators check multiple fields!",
}
