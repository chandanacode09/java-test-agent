package com.example;

public class MalformedUnbalanced {
    public void method() {
        if (true) {
            System.out.println("inside if");
            // Missing closing brace for if

    public void anotherMethod() {
        // This method is malformed due to missing brace above
    }
}
