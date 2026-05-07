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

**IMPORTANT: NEVER run `uv` commands from this local package directory.**
**ALWAYS execute from the project root (`/workspaces/LoRAIro/`).**

```bash
# From project root ONLY:
uv run pytest local_packages/image-annotator-lib/tests/

# Run specific test categories
uv run pytest -m unit local_packages/image-annotator-lib/tests/
uv run pytest -m integration local_packages/image-annotator-lib/tests/
uv run pytest -m webapi local_packages/image-annotator-lib/tests/

# Run single test file
uv run pytest local_packages/image-annotator-lib/tests/unit/core/test_config.py

# Run with coverage (use coverage run to avoid torch reload issues)
uv run coverage run -m pytest local_packages/image-annotator-lib/tests/
uv run coverage report -m
uv run coverage xml  # For CI integration
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

### WebAPI Inference Architecture (ADR 0023 Phase 1)

> **Design Authority:**
> WebAPI 推論経路の責務分離 (PydanticAI / LiteLLM / image-annotator-lib) は LoRAIro 側 ADR が SSoT。
> 詳細仕様は [LoRAIro ADR 0023 — PydanticAI / LiteLLM WebAPI Inference Boundary](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md) を参照。
> 関連 ISSUE: [#37 (ADR)](https://github.com/NEXTAltair/image-annotator-lib/issues/37) /
> [#36 (Phase 1 実装)](https://github.com/NEXTAltair/image-annotator-lib/issues/36) /
> [#35 (device 判定分離)](https://github.com/NEXTAltair/image-annotator-lib/issues/35)

**責務分離:**

| 機能 | 担当 |
|---|---|
| WebAPI モデル discovery / capability metadata | LiteLLM 同梱 DB (runtime call、TOML キャッシュなし) |
| 推論実行 / multimodal input / structured output / output retry | PydanticAI native provider/model |
| LiteLLM ID と PydanticAI 実行 descriptor の mapping | `core/model_id.py` |
| Schema → Result 変換 / 軽微正規化 | `core/result_adapter.py` |
| PIL Image → BinaryContent 変換 | `core/image_preprocess.py` |
| BaseAnnotator wrapping (direct LiteLLM ID + registry 登録 WebAPI) | `core/webapi_annotator.py` |
| Agent 構築・実行 (キャッシュなし) | `core/provider_manager.py` |

**主要コンポーネント:**

1. **`core/model_id.py`** — LiteLLM ID から `PydanticAIModelRef` への変換 + provider object 構築
   - `_BUILDER_DISPATCH` テーブルが `SUPPORTED_PROVIDERS` の真の source
   - Phase 1 対応: OpenAI / Anthropic / Google (Gemini alias) / OpenRouter
   - 未知 provider は `UnknownProviderError` で fail-fast
   - API key は provider object に明示注入 (`os.environ` mutate なし)

2. **`core/provider_manager.py`** — 推論実行の中核
   - `run_inference_with_model_async()`: async core 実装
   - `run_inference_with_model()`: sync wrapper (running event loop 内では明示エラー)
   - 推論呼び出しごとに Agent / Provider / Model を新規作成 (キャッシュなし)
   - `litellm.supports_vision()` で実行直前 fail-fast
   - PydanticAI: `await agent.run([prompt_text, binary_content])` の sequence 形式
   - `output_retries=1` で structured output validation failure を 1 回再生成

3. **`core/webapi_annotator.py`** — `BaseAnnotator` 継承の汎用 wrapper
   - direct LiteLLM ID (`google/gemini-...`) と registry 登録 WebAPI モデルの双方を扱う
   - `BaseAnnotator.__init__` の `config_registry` 依存を回避するため最小限の field 設定
   - `__enter__` / `__exit__` は no-op

4. **`core/api_model_discovery.py`** — LiteLLM 同梱 DB の runtime query
   - `discover_available_vision_models()` → `{"models": [...], "metadata": {...}}`
   - `get_available_models()` / `list_all_models()` / `is_model_deprecated()` を helper として公開
   - 旧 `available_api_models.toml` キャッシュ / TTL refresh / OpenRouter fallback は廃止

**Exception 階層 (`exceptions/errors.py`):**
- `WebApiError` (root)
  - `IdMappingError` — litellm_model_id 解析失敗
  - `UnknownProviderError` — `SUPPORTED_PROVIDERS` 外
  - `MissingApiKeyError` — api_keys に該当 provider キーなし
  - `VisionUnsupportedError` — `litellm.supports_vision()` False
  - `InferenceError` — PydanticAI 実行時エラーの wrap

**Usage Pattern (ADR 0023 Phase 1):**
```python
from image_annotator_lib import annotate

# direct LiteLLM ID 経由 (registry 登録不要)
results = annotate(
    images_list=[...],
    model_name_list=["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"],
    api_keys={"openai": "sk-...", "anthropic": "sk-ant-..."},
)
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
