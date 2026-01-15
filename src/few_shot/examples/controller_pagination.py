"""
Few-shot example: Controller with Pagination.

This example demonstrates testing a Spring MVC Controller that uses:
- Pageable parameters for list endpoints
- Page<Entity> return types
- Repository mocking with pagination
"""

CONTROLLER_PAGINATION_EXAMPLE = '''
// =============================================================================
// PATTERN: Controller with Pagination (Pageable/Page)
// =============================================================================
// KEY CONCEPTS:
//   - Mock repository methods that take Pageable parameter
//   - Use any(Pageable.class) for flexible matching
//   - Create Page objects with PageImpl for mock returns
//   - Test empty page, single result, multiple results
// =============================================================================

package org.example.controller;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.ui.Model;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class VetControllerTest {

    // PATTERN: Mock all repository/service dependencies
    @Mock
    private VetRepository vetRepository;

    @Mock
    private Model model;

    // PATTERN: @InjectMocks creates instance and injects mocks
    @InjectMocks
    private VetController controller;

    // =========================================================================
    // Pagination Tests - Repository returns Page<Entity>
    // =========================================================================

    @Test
    @DisplayName("showVetList should return vets view with paginated results")
    void showVetList_withValidPage_returnsVetsView() {
        // PATTERN: Create Page using PageImpl with list of entities
        Vet vet = new Vet();
        vet.setFirstName("James");
        vet.setLastName("Carter");
        Page<Vet> page = new PageImpl<>(List.of(vet));

        // PATTERN: Use any(Pageable.class) to match any pagination params
        when(vetRepository.findAll(any(Pageable.class))).thenReturn(page);

        String result = controller.showVetList(1, model);

        assertEquals("vets/vetList", result);
        // PATTERN: Verify model attributes were added
        verify(model).addAttribute(eq("listVets"), any());
    }

    @Test
    @DisplayName("showVetList should handle empty results")
    void showVetList_withNoVets_returnsEmptyPage() {
        // PATTERN: Test empty Page scenario
        Page<Vet> emptyPage = new PageImpl<>(Collections.emptyList());

        when(vetRepository.findAll(any(Pageable.class))).thenReturn(emptyPage);

        String result = controller.showVetList(1, model);

        assertEquals("vets/vetList", result);
    }

    @Test
    @DisplayName("showVetList should pass correct page number to repository")
    void showVetList_verifiesPageParameter() {
        Page<Vet> page = new PageImpl<>(List.of(new Vet()));
        when(vetRepository.findAll(any(Pageable.class))).thenReturn(page);

        controller.showVetList(2, model);

        // PATTERN: Verify the repository was called (interaction verification)
        verify(vetRepository).findAll(any(Pageable.class));
    }

    // =========================================================================
    // JSON Endpoint Tests (if applicable)
    // =========================================================================

    @Test
    @DisplayName("showResourcesVetList should return Vets collection")
    void showResourcesVetList_returnsVetsCollection() {
        Vet vet = new Vet();
        vet.setFirstName("Helen");
        Page<Vet> page = new PageImpl<>(List.of(vet));

        when(vetRepository.findAll(any(Pageable.class))).thenReturn(page);

        Vets result = controller.showResourcesVetList();

        assertNotNull(result);
        assertFalse(result.getVetList().isEmpty());
    }
}
'''

CONTROLLER_PAGINATION_META = {
    "component_type": "controller_pagination",
    "patterns": ["findAll_pageable", "page_response", "pageimpl_mock", "model_attributes"],
    "annotations": ["@Controller", "@RestController", "@GetMapping"],
    "key_imports": [
        "org.springframework.data.domain.Page",
        "org.springframework.data.domain.PageImpl",
        "org.springframework.data.domain.Pageable",
        "static org.mockito.ArgumentMatchers.*",
        "static org.mockito.Mockito.*",
    ],
    "critical_pattern": "when(repository.findAll(any(Pageable.class))).thenReturn(page)",
}
