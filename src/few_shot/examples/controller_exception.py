"""
Few-shot example: Controller with Exception Handling (orElseThrow).

This example demonstrates testing a Spring MVC Controller that uses:
- Optional.orElseThrow() for not-found cases
- Exception assertions with assertThrows
- @ModelAttribute methods that can throw
"""

CONTROLLER_EXCEPTION_EXAMPLE = '''
// =============================================================================
// PATTERN: Controller with orElseThrow Exception Handling
// =============================================================================
// KEY CONCEPTS:
//   - When repository returns Optional.empty(), controller throws exception
//   - Use assertThrows to verify exception is thrown
//   - Do NOT expect null return - expect the exception!
//   - Test both found (success) and not-found (exception) paths
// =============================================================================

package org.example.controller;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class OwnerControllerTest {

    @Mock
    private OwnerRepository ownerRepository;

    @InjectMocks
    private OwnerController controller;

    // =========================================================================
    // orElseThrow Pattern - Entity Found (Success Path)
    // =========================================================================

    @Test
    @DisplayName("findOwner should return owner when found")
    void findOwner_withValidId_returnsOwner() {
        // PATTERN: Mock Optional.of() for found case
        Owner owner = new Owner();
        owner.setId(1);
        owner.setFirstName("John");

        when(ownerRepository.findById(1)).thenReturn(Optional.of(owner));

        Owner result = controller.findOwner(1);

        assertNotNull(result);
        assertEquals("John", result.getFirstName());
    }

    // =========================================================================
    // orElseThrow Pattern - Entity Not Found (Exception Path)
    // =========================================================================

    @Test
    @DisplayName("findOwner should throw IllegalArgumentException when not found")
    void findOwner_withInvalidId_throwsException() {
        // PATTERN: Mock Optional.empty() for not-found case
        when(ownerRepository.findById(999)).thenReturn(Optional.empty());

        // PATTERN: Use assertThrows - do NOT expect null!
        // The controller uses orElseThrow, so it throws an exception
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> controller.findOwner(999)
        );

        // PATTERN: Optionally verify exception message
        assertTrue(exception.getMessage().contains("not found")
                || exception.getMessage().contains("999"));
    }

    // =========================================================================
    // @ModelAttribute with null ID (New Entity Path)
    // =========================================================================

    @Test
    @DisplayName("findOwner should return new Owner when id is null")
    void findOwner_withNullId_returnsNewOwner() {
        // PATTERN: @ModelAttribute methods often handle null ID = new entity
        Owner result = controller.findOwner(null);

        assertNotNull(result);
        // New owner has no ID set
        assertNull(result.getId());
    }

    // =========================================================================
    // Update/Delete Operations with orElseThrow
    // =========================================================================

    @Test
    @DisplayName("showOwner should throw when owner not found")
    void showOwner_withInvalidId_throwsException() {
        when(ownerRepository.findById(999)).thenReturn(Optional.empty());

        // Any method that fetches by ID and uses orElseThrow will throw
        assertThrows(
            IllegalArgumentException.class,
            () -> controller.showOwner(999)
        );
    }

    @Test
    @DisplayName("showOwner should return view when owner found")
    void showOwner_withValidId_returnsView() {
        Owner owner = new Owner();
        owner.setId(1);
        when(ownerRepository.findById(1)).thenReturn(Optional.of(owner));

        // Method returns ModelAndView or String view name
        var result = controller.showOwner(1);

        assertNotNull(result);
    }
}
'''

CONTROLLER_EXCEPTION_META = {
    "component_type": "controller_exception",
    "patterns": ["or_else_throw", "optional_handling", "assert_throws", "model_attribute_null_id"],
    "annotations": ["@Controller", "@ModelAttribute", "@PathVariable"],
    "key_imports": [
        "java.util.Optional",
        "static org.junit.jupiter.api.Assertions.*",
        "static org.mockito.Mockito.*",
    ],
    "critical_pattern": "assertThrows(IllegalArgumentException.class, () -> controller.method(invalidId))",
    "common_mistake": "Do NOT expect null return when orElseThrow is used - expect the exception!",
}
