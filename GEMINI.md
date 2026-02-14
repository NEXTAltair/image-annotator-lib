# Gemini Project Context: image-annotator-lib

This document provides essential context about the `image-annotator-lib` project for the Gemini agent. It is based on the project structure, existing code, and the rules defined in `.cursor/rules/`.

## 1. Project Overview

`image-annotator-lib` is a Python library designed for integrated image tagging and scoring. It supports both local, on-device models (e.g., ONNX-based taggers) and various Web API-based models (e.g., from Google, OpenAI, Anthropic) for generating image annotations like tags and captions.

## 2. Core Principles & Rules (High Importance)

These rules, derived from `.cursor/rules/`, must be strictly followed at all times.

- **Reference Documentation:** Always reference documents in the `docs/` directory and rule files (`.mdc`) in `.cursor/rules/` before responding.
- **Follow Plans and Memory:** Adhere to the planning and memory management workflows defined in `.cursor/rules/plan.mdc` and `.cursor/rules/memory.mdc`. Use Serena memory for short-term task tracking and OpenClaw→Notion for long-term records.
- **YAGNI (You Ain't Gonna Need It):** Do not implement features that might be needed later. Only implement the minimum required functionality for the current task.
- **No Full-Width Characters:** Never use full-width alphanumeric characters or symbols in code, comments, or documentation.
- **Record Changes:** If you alter the existing design, you must check related documentation and record the change and its reasoning in the appropriate location.
- **Encapsulation:** Strictly adhere to the encapsulation rules. Never access internal variables (prefixed with `_`) of other classes directly. Follow the "Tell, Don't Ask" principle. Getters and setters are generally forbidden.

## 3. Core Technologies

- **Language:** Python 3.12
- **Build System:** `hatchling` (configured in `pyproject.toml`)
- **Testing:** `pytest` with plugins like `pytest-cov` and `pytest-xdist`.
- **Linting & Formatting:** `ruff`
- **Type Checking:** `mypy`
- **Key Libraries:**
    - `Pillow` for image manipulation.
    - `tensorflow`, `transformers`, `onnxruntime` for local model inference.
    - `pydantic-ai` for interacting with modern Web APIs (Google, OpenAI, Anthropic).
    - `loguru` for logging.

## 4. Coding Conventions

- **Type Hinting:**
    - Use modern type hints (e.g., `list`, `dict`) instead of `typing.List`, `typing.Dict`.
    - Use the `|` operator for unions (e.g., `str | None`) instead of `Optional`.
    - Use `TypedDict` for complex dictionary structures.
    - Use the `@override` decorator when overriding methods from a parent class.
- **List Comprehensions:** Limit list comprehensions to one `if` and one `for` clause to maintain readability.

## 5. Architecture

The library's architecture is modular, separating different model types and providers.

### Key Architectural Principles:

- **Encapsulation:** The design strictly follows the "Tell, Don't Ask" principle. Classes should expose behavior (methods) rather than state. Direct access to internal variables (`_`) is forbidden, and getters/setters are discouraged in favor of methods that perform actions.
- **Provider Pattern:** The `ProviderManager` abstracts away the specifics of each API provider (Google, OpenAI, etc.), allowing for shared connections and resources. This is a central piece of the modern Web API interaction model.
- **Factory Pattern:** The `PydanticAIProviderFactory` serves as a central factory for creating and caching `pydantic-ai` `Agent` instances, optimizing resource usage.
- **Registry Pattern:** Used for both class definitions (`get_cls_obj_registry`) and model instances (`_MODEL_INSTANCE_REGISTRY` in `api.py`).

### Key Components:

- **`src/image_annotator_lib/api.py`**: The main public-facing API of the library. It provides the `annotate()` function and manages the lifecycle of annotator instances.
- **`src/image_annotator_lib/core/`**: Contains the core logic.
    - **`provider_manager.py`**: Manages provider-level instances (e.g., one instance for all of Google's models) to share resources like API clients efficiently.
    - **`pydantic_ai_factory.py`**: Contains `PydanticAIProviderFactory` and `PydanticAIAnnotatorMixin`. This factory creates and caches `pydantic-ai` `Agent` instances.
    - **`config.py`**: A global `config_registry` handles loading and accessing model configurations from TOML files.
    - **`registry.py`**: A class registry (`get_cls_obj_registry`) maps model names to their corresponding annotator classes.
    - **`base/`**: Abstract base classes for different annotator types.
- **`src/image_annotator_lib/model_class/`**: Contains the concrete implementations of annotator classes.

## 6. Development Workflow

Commands should be run from the project root (`C:\LoRAIro\local_packages\image-annotator-lib`). The project uses `uv` as a virtual environment and runner.

- **Run all unit tests:**
  ```bash
  uv run pytest tests/unit
  ```
- **Run a specific test file:**
  ```bash
  uv run pytest path/to/test_file.py
  ```
- **Run linter/formatter:**
  ```bash
  uv run ruff check . --fix
  uv run ruff format .
  ```
- **Run type checker:**
  ```bash
  uv run mypy src
  ```

## 7. Key Files to Reference

- **Project Rules & Conventions:**
    - **`.cursor/rules/rules.mdc`**: **Primary source of truth for all rules.**
    - **`.cursor/rules/memory.mdc`**: Defines the structure and use of memory/documentation files.
    - **`.cursor/rules/plan.mdc`**: Defines the planning and task execution workflow.
    - **`.cursor/rules/directory-structure.mdc`**: For understanding the project's directory layout.
    - **`.cursor/rules/implement.mdc`**: For rules regarding implementation.
    - **`.cursor/rules/debug.mdc`**: For debugging procedures.
- **Task Management:**
    - **MCP Serena Memory**: All task tracking, project status, and development records
- **OpenClaw → Notion LTM**: Long-term design decisions and architectural knowledge
- **Project Configuration:**
    - **`pyproject.toml`**: Project metadata, dependencies, and tool configurations (`ruff`, `pytest`, `mypy`).
- **Core Logic:**
    - **`src/image_annotator_lib/api.py`**: Entry point for understanding user-facing functionality.
    - **`src/image_annotator_lib/core/provider_manager.py`**: Crucial for understanding the new Web API architecture.
    - **`src/image_annotator_lib/core/pydantic_ai_factory.py`**: Key to the `pydantic-ai` implementation.
- **Documentation:**
    - **`docs/`**: Contains architectural and design decision documents.
