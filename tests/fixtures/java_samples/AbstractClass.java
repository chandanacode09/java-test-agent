package com.example.base;

/**
 * Abstract class with abstract and concrete methods.
 */
public abstract class AbstractClass {

    protected String name;

    public AbstractClass(String name) {
        this.name = name;
    }

    /**
     * Abstract method to be implemented by subclasses.
     */
    public abstract void doSomething();

    /**
     * Another abstract method.
     * @return the result
     */
    public abstract String getResult();

    /**
     * Concrete method with implementation.
     */
    public void concreteMethod() {
        System.out.println("Concrete implementation: " + name);
    }

    /**
     * Protected method for subclasses.
     * @param value the value to process
     * @return processed value
     */
    protected String processValue(String value) {
        return value.toUpperCase();
    }
}
