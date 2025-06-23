# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync --dev

# Add new dependencies
uv add package-name

# Add development dependencies
uv add --dev package-name
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories using markers
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only  
pytest -m webapi      # Web API tests only
pytest -m scorer      # Scorer model tests only
pytest -m tagger      # Tagger model tests only

# Run single test file
pytest tests/unit/core/test_config.py

# Run with coverage
pytest --cov=src --cov-report=xml
pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Run linting and formatting
ruff check
ruff format

# Run type checking
mypy src/
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

**Model Categories:**
- **Local ML Models**: ONNX, Transformers, TensorFlow, CLIP-based models
- **Web API Models**: Located in `model_class/annotator_webapi/` directory:
  - `anthropic_api.py` - AnthropicApiAnnotator
  - `google_api.py` - GoogleApiAnnotator
  - `openai_api_chat.py` & `openai_api_response.py` - OpenAI implementations
  - `webapi_shared.py` - Shared WebAPI utilities
- **Specialized Models**: DeepDanbooru taggers, aesthetic scorers, captioning models

**Key Design Principles:**
- Unified API via `annotate()` function for all model types
- pHash-based result mapping (image -> results) for robust batch processing
- Memory-aware model loading with automatic cache management
- Framework-agnostic base classes with shared functionality

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

Web API annotators follow a consistent pattern:
- Base class `WebApiBaseAnnotator` (`core/base/webapi.py`) provides common API handling
- Provider-specific classes in `model_class/annotator_webapi/` handle authentication and API specifics
- Pydantic models (`AnnotationSchema` in `core/types.py`) ensure type safety
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
2. Implement required abstract methods (`_generate_tags`, `_run_inference`, etc.)
3. Add model entry to `annotator_config.toml`
4. Add corresponding test in appropriate test directory

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
- Consider device placement (CUDA vs CPU) based on model requirements

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