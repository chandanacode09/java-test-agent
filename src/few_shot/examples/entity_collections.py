"""
Few-shot example: Entity with Collection Management.

This example demonstrates testing a JPA entity that manages collections:
- Adding items to a collection
- Removing items from a collection
- Finding items by ID or name
- Handling duplicates and not-found cases
"""

ENTITY_COLLECTIONS_EXAMPLE = '''
// =============================================================================
// PATTERN: Entity with Collection Management (add/remove/get)
// =============================================================================
// KEY CONCEPTS:
//   - Test add methods verify item is in collection after add
//   - Test conditional add logic (e.g., only add if isNew())
//   - Test lookup by ID and by name (often case-insensitive)
//   - Test not-found returns null (not exception)
// =============================================================================

package org.example.model;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;
import java.time.LocalDate;

class OwnerTest {

    private Owner owner;

    @BeforeEach
    void setUp() {
        owner = new Owner();
        owner.setFirstName("John");
        owner.setLastName("Doe");
    }

    // =========================================================================
    // Collection Add Tests
    // =========================================================================

    @Test
    @DisplayName("addPet should add new pet to the collection")
    void addPet_withNewPet_addsToCollection() {
        // PATTERN: Create related entity, add it, verify it's in collection
        Pet pet = new Pet();
        pet.setName("Buddy");
        pet.setBirthDate(LocalDate.now());
        pet.setType(new PetType());

        owner.addPet(pet);

        assertEquals(1, owner.getPets().size());
        assertTrue(owner.getPets().contains(pet));
        // PATTERN: Verify bidirectional relationship if applicable
        assertEquals(owner, pet.getOwner());
    }

    @Test
    @DisplayName("addPet should not add pet that already has an ID (not new)")
    void addPet_withExistingPet_doesNotAdd() {
        // PATTERN: Test conditional add logic - many add methods check isNew()
        Pet existingPet = new Pet();
        existingPet.setId(1);  // Has ID = not new
        existingPet.setName("Max");

        owner.addPet(existingPet);

        // Pet with ID should not be added to internal collection
        assertTrue(owner.getPets().isEmpty());
    }

    // =========================================================================
    // Collection Lookup by ID Tests
    // =========================================================================

    @Test
    @DisplayName("getPet by ID should return matching pet")
    void getPetById_withValidId_returnsPet() {
        // PATTERN: Setup collection directly for lookup tests
        Pet pet = new Pet();
        pet.setId(5);
        pet.setName("Buddy");
        owner.getPets().add(pet);

        Pet found = owner.getPet(5);

        assertNotNull(found);
        assertEquals("Buddy", found.getName());
    }

    @Test
    @DisplayName("getPet by ID should return null when not found")
    void getPetById_withInvalidId_returnsNull() {
        // PATTERN: Not-found typically returns null, not exception
        Pet result = owner.getPet(999);

        assertNull(result);
    }

    // =========================================================================
    // Collection Lookup by Name Tests
    // =========================================================================

    @Test
    @DisplayName("getPet by name should be case-insensitive")
    void getPetByName_caseInsensitive_findsPet() {
        // PATTERN: String lookups are often case-insensitive
        Pet pet = new Pet();
        pet.setId(1);
        pet.setName("Buddy");
        owner.getPets().add(pet);

        Pet found = owner.getPet("BUDDY");

        assertNotNull(found);
        assertEquals("Buddy", found.getName());
    }

    @Test
    @DisplayName("getPet by name should return null when not found")
    void getPetByName_notFound_returnsNull() {
        Pet result = owner.getPet("NonExistent");

        assertNull(result);
    }
}
'''

ENTITY_COLLECTIONS_META = {
    "component_type": "entity_collections",
    "patterns": ["add_to_collection", "remove_from_collection", "find_in_collection", "bidirectional_relationship"],
    "annotations": ["@Entity", "@OneToMany", "@ManyToMany"],
    "key_methods": ["addX", "removeX", "getX(id)", "getX(name)", "getPets", "getSpecialties"],
    "key_imports": [
        "org.junit.jupiter.api.BeforeEach",
        "org.junit.jupiter.api.Test",
        "org.junit.jupiter.api.DisplayName",
        "static org.junit.jupiter.api.Assertions.*",
        "java.time.LocalDate",
    ],
}
