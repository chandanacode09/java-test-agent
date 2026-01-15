package com.example.generic;

import java.util.*;
import java.util.function.Function;

/**
 * Class with complex generics for testing parser.
 */
public class GenericClass<T, K extends Comparable<K>> {
    private Map<K, List<T>> items;
    private Set<Map.Entry<String, T>> entries;

    public GenericClass() {
        this.items = new HashMap<>();
        this.entries = new HashSet<>();
    }

    public Map<K, List<T>> getItems() {
        return items;
    }

    public void setItems(Map<K, List<T>> items) {
        this.items = items;
    }

    public Set<Map.Entry<String, T>> getEntries() {
        return entries;
    }

    public <R> R transform(T input, Function<T, R> transformer) {
        return transformer.apply(input);
    }

    public Map<String, List<Map<K, T>>> getNestedGenerics() {
        return new HashMap<>();
    }

    public void addItem(K key, T value) {
        items.computeIfAbsent(key, k -> new ArrayList<>()).add(value);
    }

    public Optional<T> findFirst(K key) {
        List<T> list = items.get(key);
        return list != null && !list.isEmpty() ? Optional.of(list.get(0)) : Optional.empty();
    }
}
