# CLAUDE.md - Project Guidelines

## Core Behavior Principles

### 1. Proactive Problem Anticipation
- **Think ahead**: Before implementing a fix, consider what other issues might exist in the same category
- **Don't be reactive**: When you find a bug, ask yourself "what other similar bugs might exist?"
- **Identify patterns**: If one component needs X, similar components probably need X too
- Example: If LLM hallucinates method names for one repository, anticipate it will happen for ALL repositories

### 2. Root Cause Analysis
- **Go deeper**: Don't just fix symptoms, understand WHY the problem occurred
- **Ask "why" 5 times**: Keep digging until you find the fundamental issue
- **Fix the system, not the instance**: If a bug reveals a pattern, fix all instances of the pattern

### 3. Information Flow Thinking
- **What does the LLM need?**: Always consider what information the LLM needs to succeed
- **Garbage in, garbage out**: If you give incomplete context, expect incomplete results
- **Be explicit**: Don't assume the LLM will figure things out - provide the exact data it needs

### 4. Speed vs Correctness
- **Measure first**: Before optimizing, measure actual performance
- **Don't break things for speed**: A fast wrong answer is worse than a slow right answer
- **Incremental changes**: Make one change at a time and verify it doesn't regress

### 5. Learning from Failures
- **Failures are data**: Every failure tells you something about the system
- **Document patterns**: When you discover why something fails, note the pattern
- **Build systematic fixes**: Don't just patch - improve the underlying system

## Project-Specific Guidelines

### Java Test Generation
- LLMs hallucinate method names - always provide actual API signatures
- Checked exceptions must be declared with `throws` or caught
- Import filtering should be conservative - include rather than exclude
- Repository interfaces need explicit method extraction

### Eval Framework
- Trust SpecAgent results directly when possible
- Redundant graders add latency without value
- Use FAST_FLAGS to skip formatting/style checks in tests

### Debugging Approach
1. Read the actual error message carefully
2. Check if the generated code matches expected patterns
3. Compare with working examples (e.g., OwnerControllerTest.java)
4. Identify what information was missing from LLM context

## Anti-Patterns to Avoid
- Making changes without understanding impact
- Reducing timeouts/iterations as a "fix" for slowness
- Filtering imports too aggressively
- Assuming LLM will infer information it wasn't given
- Being reactive instead of proactive
