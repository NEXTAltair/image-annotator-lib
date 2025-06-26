# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Setup development environment (recommended)
make setup

# Install dependencies manually
make install      # Production dependencies
make install-dev  # Development dependencies

# Add new dependencies (manual uv commands)
uv add package-name
uv add --dev package-name
```

### Testing
```bash
# Run all tests
make test

# Run specific test categories
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-webapi      # Web API tests only
make test-scorer      # Scorer model tests only
make test-tagger      # Tagger model tests only

# Run with coverage
make test-cov

# Run single test file (manual pytest)
pytest tests/unit/core/test_config.py
```

### Code Quality
```bash
# Run linting and formatting
make lint
make format

# Run type checking
make typecheck
```

### Example Usage
```bash
# Run example script
python example/example_lib.py
```

## Project Architecture

### Core Design Pattern: 3-Layer Hierarchy

The library follows a clean 3-layer architecture for annotators:

1. **BaseAnnotator** (`core/base/annotator.py`) - Common interface and shared functionality
2. **Framework-specific base classes** - Located in `core/base/` directory:
   - `core/base/webapi.py` - WebApiBaseAnnotator
   - `core/base/onnx.py` - ONNXBaseAnnotator
   - `core/base/transformers.py` - TransformersBaseAnnotator
   - `core/base/tensorflow.py` - TensorflowBaseAnnotator
   - `core/base/clip.py` - ClipBaseAnnotator
   - `core/base/pipeline.py` - PipelineBaseAnnotator
3. **Concrete model classes** - Specific implementations (WDTagger, AestheticScorer, etc.)

### Key Components

**Entry Point:**
- `annotate()` function in `api.py` - Main user interface for batch processing
- `list_available_annotators()` - Returns available models from config

**Core Services:**
- `ModelLoad` (`core/model_factory.py`) - Model loading, caching, and memory management with LRU strategy
- `ModelRegistry` (`core/registry.py`) - Maps model names to implementation classes
- `ModelConfigRegistry` (`core/config.py`) - Configuration management with system/user config separation
- `PydanticAIProviderFactory` (`core/pydantic_ai_factory.py`) - Provider-level Agent factory for PydanticAI WebAPI models
- `ProviderManager` (`core/provider_manager.py`) - Manages provider-level instances for efficient PydanticAI usage

**Model Categories:**
- **Local ML Models**: ONNX, Transformers, TensorFlow, CLIP-based models
- **Web API Models (PydanticAI-based)**: Located in `model_class/annotator_webapi/` directory:
  - `anthropic_api.py` - AnthropicApiAnnotator (Provider-level with Agent caching)
  - `google_api.py` - GoogleApiAnnotator (Provider-level with Agent caching)
  - `openai_api_chat.py` - OpenAI & OpenRouter implementations (Provider-level)
  - `openai_api_response.py` - Legacy OpenAI implementations
  - `webapi_shared.py` - Shared WebAPI utilities and prompts
- **Specialized Models**: DeepDanbooru taggers, aesthetic scorers, captioning models

**Key Design Principles:**
- Unified API via `annotate()` function for all model types
- pHash-based result mapping (image -> results) for robust batch processing
- Memory-aware model loading with automatic cache management
- Framework-agnostic base classes with shared functionality
- **Provider-level resource sharing for PydanticAI WebAPI models** - Efficient Agent reuse across multiple inferences

### Configuration System

**Config Files:**
- `config/annotator_config.toml` - Main model configuration (sample available)
- `config/available_api_models.toml` - API model definitions
- `config/user_config.toml` - User-specific overrides
- `src/image_annotator_lib/resources/system/annotator_config.toml` - System defaults

**Key Config Sections:**
- Model definitions with `model_path`, `class`, `device`, `estimated_size_gb`
- Web API models with timeout, retry settings, rate limiting
- Memory management parameters for cache control

### Memory Management

The `ModelLoad` class implements sophisticated memory management:
- **Pre-load size calculation** - Estimates model size before loading to prevent OOM
- **LRU cache strategy** - Automatically evicts least recently used models
- **CUDA/CPU management** - Moves models between devices as needed
- **Memory monitoring** - Uses psutil to check available system memory

### Web API Integration

**Provider-Level PydanticAI Architecture:**
- `PydanticAIProviderFactory` manages Provider instances and Agent caching for efficiency
- `ProviderManager` coordinates provider-level inference execution with model ID routing
- Automatic provider detection (OpenAI, Anthropic, Google, OpenRouter) from model IDs
- Shared Provider instances across multiple model requests for optimal resource usage
- Agent caching with LRU strategy and configuration change detection

**Implementation Pattern:**
- Base class `WebApiBaseAnnotator` with `PydanticAIAnnotatorMixin` for PydanticAI models
- Provider-specific authentication and custom headers (e.g., OpenRouter referer/app_name)
- Structured output via Pydantic models (`AnnotationSchema` in `core/types.py`)
- Robust error handling for rate limits, authentication, and response parsing
- **API Model Discovery** (`core/api_model_discovery.py`) - Automatic discovery of available external API models

### Test Architecture

**Test Categories (pytest markers):**
- `unit` - Fast unit tests with mocking
- `integration` - End-to-end tests with real models
- `webapi` - Web API integration tests
- `scorer`/`tagger` - Model-type specific tests

**Test Structure:**
- `tests/unit/` - Unit tests by module
- `tests/model_class/` - Model-specific integration tests
- `tests/resources/` - Test images and data
- `conftest.py` - Shared test fixtures
- `test_provider_level_integration.py` - Provider-level PydanticAI integration tests
- `test_*_api_pydanticai_integration.py` - Provider-specific PydanticAI tests

### Important File Patterns

**Result Format:**
```python
# annotate() returns: Dict[str, Dict[str, AnnotationResult]]
{
    "image_phash": {
        "model_name": {
            "tags": ["tag1", "tag2", ...],
            "formatted_output": {...},
            "error": None  # or error message
        }
    }
}
```

**Model Configuration:**
```toml
[model_name]
model_path = "huggingface/repo-name"  # or URL or local path
class = "ModelClassName"
device = "cuda"  # or "cpu"
estimated_size_gb = 1.5
```

### Development Guidelines

**Adding New Models:**
1. Create concrete class inheriting from appropriate framework base class
2. For PydanticAI WebAPI models: inherit from `WebApiBaseAnnotator` + `PydanticAIAnnotatorMixin`
3. Implement required abstract methods (`_generate_tags`, `_run_inference`, etc.)
4. For PydanticAI models: implement `run_with_model()` method for provider-level execution
5. Add model entry to `annotator_config.toml` with `api_model_id` for WebAPI models
6. Add corresponding test in appropriate test directory

**Code Style:**
- Uses Ruff for linting/formatting (line length: 108)
- Type hints required for all functions
- Modern Python types (list/dict over typing.List/Dict)
- Loguru for structured logging

**Error Handling:**
- Custom exceptions in `exceptions/errors.py`
- Graceful degradation - partial results returned on model failures
- Comprehensive error messages with context

**Memory Considerations:**
- All models should specify `estimated_size_gb` in config
- Use `ModelLoad` for all model loading operations
- For PydanticAI WebAPI models: Provider-level sharing reduces memory footprint
- Consider device placement (CUDA vs CPU) based on model requirements

**PydanticAI-Specific Guidelines:**
- Use `PydanticAIProviderFactory.get_cached_agent()` for efficient Agent reuse
- Implement `run_with_model()` for provider-level execution with model override support
- Use `PydanticAIAnnotatorMixin._preprocess_images_to_binary()` for PIL→BinaryContent conversion
- Provider instances are shared across models using the same provider and configuration

### Provider-Level Architecture (PydanticAI)

**Design Philosophy:**
The Provider-level architecture optimizes resource usage for PydanticAI WebAPI models by sharing Provider instances across multiple model requests, rather than creating separate instances for each model name.

**Key Components:**

1. **PydanticAIProviderFactory** (`core/pydantic_ai_factory.py`):
   - Manages Provider instances with object ID-based caching
   - Creates and caches PydanticAI Agents with LRU strategy
   - Handles provider-specific configurations (OpenRouter custom headers, etc.)
   - Uses PydanticAI's built-in `infer_model()` and `infer_provider()` functions

2. **ProviderManager** (`core/provider_manager.py`):
   - Coordinates provider-level inference execution
   - Determines appropriate provider from model IDs
   - Routes inference requests to correct provider instances
   - Manages model ID prefix handling (e.g., "openrouter:" prefix)

3. **PydanticAIAnnotatorMixin** (`core/pydantic_ai_factory.py`):
   - Provides shared functionality for PydanticAI-based annotators
   - Handles PIL Image → BinaryContent conversion
   - Implements async inference with sync wrapper support
   - Manages configuration loading and Agent setup

4. **PydanticAIWebAPIWrapper** (`api.py`):
   - Provides backward compatibility with existing `annotate()` API
   - Automatically detects PydanticAI annotators and routes to Provider Manager
   - Converts Provider Manager results to standard AnnotationResult format

**Benefits:**
- **Memory Efficiency**: Single Provider instance shared across multiple models
- **Performance**: Agent caching with LRU strategy reduces initialization overhead
- **Scalability**: Provider-level sharing supports large numbers of API models
- **Flexibility**: Model ID override support for dynamic model selection
- **Maintainability**: Centralized provider management and configuration

**Usage Pattern:**
```python
# Traditional (inefficient for PydanticAI)
model1 = AnthropicApiAnnotator("model1")  # Creates provider instance
model2 = AnthropicApiAnnotator("model2")  # Creates another provider instance

# Provider-level (efficient)
results = ProviderManager.run_inference_with_model(
    model_name="model1",
    images=images,
    api_model_id="claude-3-5-sonnet"
)  # Reuses shared Anthropic provider
```

### Tools and Utilities

**Debug Scripts in `tools/`:**
- `webapi_annotate.py` - Test Web API annotators
- `check_api_model_discovery.py` - Validate API model availability
- `check_openrouter_models.py` - OpenRouter model discovery
- `check_googleapi_payload.py` - Test Google API payload structure
- `check_openrouter_payload.py` - Test OpenRouter API payload structure

**Model Storage:**
- `models/` directory contains downloaded model files
- `models_data.json` tracks model metadata
- Automatic download and caching for remote models
