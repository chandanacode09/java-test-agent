# Java Test Agent

Automated test generation and fixing for Java/Maven projects using LLM (Large Language Models).

## Overview

This agent uses a ReAct (Reason-Act-Observe) loop to:
1. **Parse** Java source code using AST parsing
2. **Generate** test specifications via LLM (OpenRouter API)
3. **Render** test code using Jinja2 templates
4. **Fix** failing tests iteratively until all pass

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     JavaTestAgent                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐   │
│  │  ASTParser  │ → │SpecGenerator│ → │TemplateRenderer │   │
│  │  (parsing)  │   │   (LLM)     │   │   (Jinja2)      │   │
│  └─────────────┘   └─────────────┘   └─────────────────┘   │
│         ↓                                    ↓              │
│  FunctionContext              TestSpec → Java Test Code     │
│  ClassContext                                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              JavaReActLoop                           │   │
│  │  (Reason → Act → Observe cycle to fix failing tests) │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
import os
from pathlib import Path
from src.java_agent import JavaTestAgent, JavaAgentConfig

config = JavaAgentConfig(
    project_path=Path('/path/to/your/maven-project'),
    api_key=os.environ.get('OPENROUTER_API_KEY'),
    model='x-ai/grok-code-fast-1',  # or other OpenRouter models
    max_iterations=5,
    verbose=True
)

agent = JavaTestAgent(config)
result = agent.run('OwnerController')

print(f"Tests passed: {result.tests_passed}/{result.tests_total}")
print(f"Success: {result.success}")
```

### CLI

```bash
export OPENROUTER_API_KEY="your-api-key"
python -m src.java_agent /path/to/maven-project ClassName1 ClassName2
```

## Supported Models

Any model available on OpenRouter:
- `x-ai/grok-code-fast-1` (recommended for speed/cost)
- `anthropic/claude-3.5-haiku`
- `anthropic/claude-3.5-sonnet`
- `google/gemini-pro`

## Features

### Smart Spring MVC Support
- Detects `@Controller`, `@Service`, `@Repository` annotations
- Uses `@Mock`/`@InjectMocks` patterns for dependency injection
- Generates proper Mockito-based tests

### Error Categorization
The ReAct loop categorizes errors for targeted fixes:
- `ASSERTION_FAILURE` - Wrong expected values
- `NULL_POINTER` - Missing initialization
- `ILLEGAL_ARGUMENT` - Validation failures
- `MISSING_METHOD` - Wrong method names
- `WRONG_ARGUMENTS` - Constructor/method signature mismatch

### Self-Reflection
Tracks previous fix attempts to avoid repeating failed strategies.

## Test Results

Tested on Spring PetClinic:

| Class | Type | Tests Passed | Iterations |
|-------|------|--------------|------------|
| BaseEntity | POJO | 8/8 (100%) | 1 |
| NamedEntity | POJO | 11/11 (100%) | 1 |
| Vet | Entity | 6/6 (100%) | 1 |
| Pet | Entity | 7/7 (100%) | 1 |
| PetType | Entity | 7/7 (100%) | 1 |
| Specialty | Entity | 11/11 (100%) | 3 |
| Owner | Complex | 15/16 (93.8%) | 8 |
| OwnerController | Controller | 17/17 (100%) | 1 |

## Project Structure

```
java-test-agent/
├── src/
│   ├── __init__.py
│   ├── java_agent.py           # Main agent orchestrator
│   ├── models.py               # Data models
│   ├── react_loop_java.py      # ReAct loop implementation
│   ├── context/
│   │   └── ast_parser.py       # Java/Python AST parsing
│   ├── generator/
│   │   ├── spec_generator.py   # LLM-based test spec generation
│   │   └── prompts.py          # Prompt templates
│   └── renderer/
│       └── template_renderer.py # Jinja2 template rendering
├── templates/
│   └── java/
│       ├── unit_method.java.j2
│       └── spring_controller.java.j2
├── requirements.txt
└── README.md
```

## License

MIT
