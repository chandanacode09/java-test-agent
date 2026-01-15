package com.example.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import com.example.service.UserService;
import com.example.model.User;
import java.util.List;
import java.util.Optional;

/**
 * REST controller for user management.
 */
@RestController
@RequestMapping("/api/users")
public class SpringController {

    @Autowired
    private UserService userService;

    /**
     * Get all users.
     * @return list of users
     */
    @GetMapping
    public List<User> getAllUsers() {
        return userService.findAll();
    }

    /**
     * Get user by ID.
     * @param id the user ID
     * @return the user
     */
    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        Optional<User> user = userService.findById(id);
        return user.map(ResponseEntity::ok)
                   .orElse(ResponseEntity.notFound().build());
    }

    /**
     * Create a new user.
     * @param user the user to create
     * @return the created user
     */
    @PostMapping
    public User createUser(@RequestBody User user) {
        return userService.save(user);
    }

    /**
     * Update an existing user.
     * @param id the user ID
     * @param user the updated user data
     * @return the updated user
     */
    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User user) {
        user.setId(id);
        return userService.save(user);
    }

    /**
     * Delete a user.
     * @param id the user ID
     */
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        userService.delete(id);
    }
}
