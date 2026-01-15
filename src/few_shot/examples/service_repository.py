"""
Few-shot example: Service with Repository injection.

This example demonstrates testing a Spring service that:
- Has a repository injected via @Autowired
- Performs CRUD operations
- Uses Optional for findById
"""

SERVICE_REPOSITORY_EXAMPLE = '''
// =============================================================================
// PATTERN: Service with Repository Injection
// =============================================================================
// KEY CONCEPTS:
//   - Use @Mock for repository dependency
//   - Use @InjectMocks for service under test
//   - Use when/thenReturn to stub repository methods
//   - Use verify() to check repository interactions
//   - Use Optional.of() and Optional.empty() for findById
// =============================================================================

package org.example.services;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

@ExtendWith(MockitoExtension.class)
class ProductServiceTest {

    @Mock
    private ProductRepository productRepository;

    @InjectMocks
    private ProductService productService;

    // =========================================================================
    // Create/Save Tests
    // =========================================================================

    @Test
    @DisplayName("addProduct should call repository save")
    void addProduct_callsRepositorySave() {
        // PATTERN: Create entity, call service, verify repository interaction
        Product product = new Product();
        product.setName("Test Product");

        productService.addProduct(product);

        verify(productRepository).save(product);
    }

    // =========================================================================
    // Read/Find Tests
    // =========================================================================

    @Test
    @DisplayName("getAllProducts should return list from repository")
    void getAllProducts_returnsListFromRepository() {
        // PATTERN: Stub repository, call service, verify result
        Product p1 = new Product();
        p1.setName("Product 1");
        Product p2 = new Product();
        p2.setName("Product 2");
        List<Product> expected = Arrays.asList(p1, p2);

        when(productRepository.findAll()).thenReturn(expected);

        List<Product> result = productService.getAllProducts();

        assertEquals(expected, result);
        assertEquals(2, result.size());
        verify(productRepository).findAll();
    }

    @Test
    @DisplayName("getProduct should return entity when found")
    void getProduct_whenExists_returnsProduct() {
        // PATTERN: Stub findById with Optional.of(), verify result
        Product expected = new Product();
        expected.setId(1);
        expected.setName("Found Product");

        when(productRepository.findById(1)).thenReturn(Optional.of(expected));

        Product result = productService.getProduct(1);

        assertEquals(expected, result);
        assertEquals("Found Product", result.getName());
        verify(productRepository).findById(1);
    }

    @Test
    @DisplayName("getProduct should throw when not found")
    void getProduct_whenNotExists_throws() {
        // PATTERN: Stub findById with Optional.empty(), expect exception
        when(productRepository.findById(999)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> {
            productService.getProduct(999);
        });
    }

    // =========================================================================
    // Update Tests
    // =========================================================================

    @Test
    @DisplayName("updateProduct should save updated entity")
    void updateProduct_savesUpdatedEntity() {
        // PATTERN: Stub findById, call update, verify save
        Product existing = new Product();
        existing.setId(1);
        existing.setName("Old Name");

        Product updated = new Product();
        updated.setName("New Name");

        when(productRepository.findById(1)).thenReturn(Optional.of(existing));

        productService.updateProduct(updated, 1);

        verify(productRepository).save(any(Product.class));
    }

    // =========================================================================
    // Delete Tests
    // =========================================================================

    @Test
    @DisplayName("deleteProduct should call repository deleteById")
    void deleteProduct_callsRepositoryDelete() {
        // PATTERN: Call delete, verify repository interaction
        productService.deleteProduct(1);

        verify(productRepository).deleteById(1);
    }
}
'''

SERVICE_REPOSITORY_META = {
    "component_type": "service",
    "patterns": ["mock_repository", "inject_mocks", "when_thenReturn", "verify", "optional"],
    "annotations": ["@Service", "@Component"],
    "key_imports": [
        "org.junit.jupiter.api.extension.ExtendWith",
        "org.mockito.InjectMocks",
        "org.mockito.Mock",
        "org.mockito.junit.jupiter.MockitoExtension",
        "static org.mockito.Mockito.*",
        "static org.mockito.ArgumentMatchers.*",
        "java.util.Optional",
    ],
}
