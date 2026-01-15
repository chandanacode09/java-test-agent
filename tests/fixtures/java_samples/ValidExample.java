package com.example;

import java.util.List;
import java.util.Map;

/**
 * Example class for testing AST parser.
 * This class has various method types.
 */
public class ValidExample {
    private String name;
    private int value;
    private List<String> items;

    /**
     * Default constructor.
     */
    public ValidExample() {
        this.name = "";
        this.value = 0;
    }

    /**
     * Constructor with parameters.
     * @param name the name
     * @param value the value
     */
    public ValidExample(String name, int value) {
        this.name = name;
        this.value = value;
    }

    /**
     * Gets the name.
     * @return the name
     */
    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getValue() {
        return this.value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    /**
     * Process items with multiple parameters.
     * @param input the input string
     * @param count the count
     * @param flag a boolean flag
     */
    public void processWithParams(String input, int count, boolean flag) {
        if (flag) {
            for (int i = 0; i < count; i++) {
                System.out.println(input);
            }
        }
    }

    public List<String> getItems() {
        return List.of("a", "b", "c");
    }

    public Map<String, Integer> getMapping() {
        return Map.of("one", 1, "two", 2);
    }

    public static void staticMethod() {
        System.out.println("static");
    }
}
