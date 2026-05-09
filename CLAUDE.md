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

1. **BaseAnnotator** (`core/base/annotator.py`) - Common interface and shared functionality. ADR 0023 Phase 1 (Issue #35) で device 判定はサブクラス責務に移譲済み。
2. **Framework-specific base classes** - Located in `core/base/` directory:
   - `core/base/onnx.py` - ONNXBaseAnnotator
   - `core/base/transformers.py` - TransformersBaseAnnotator
   - `core/base/tensorflow.py` - TensorflowBaseAnnotator
   - `core/base/clip.py` - ClipBaseAnnotator
   - `core/base/pipeline.py` - PipelineBaseAnnotator
   - WebAPI base class (`WebApiBaseAnnotator`) は ADR 0023 Phase 1 で廃止された。WebAPI 系は `core/webapi_annotator.py:WebApiAnnotator` 単独に統合されている。
3. **Concrete model classes** - Specific implementations (WDTagger, AestheticScorer, etc.)

### Key Components

**Entry Point:**
- `annotate()` function in `api.py` - Main user interface for batch processing
- `list_available_annotators()` - Returns available models from config

**Core Services:**
- `ModelLoad` (`core/model_factory.py`) - Model loading, caching, and memory management with LRU strategy
- `ModelRegistry` (`core/registry.py`) - Maps model names to implementation classes (WebAPI モデルは `WebApiAnnotator` 直接登録)
- `ModelConfigRegistry` (`core/config.py`) - Configuration management with system/user config separation
- `WebApiAnnotator` (`core/webapi_annotator.py`) - 全 WebAPI プロバイダー (OpenAI / Anthropic / Google / OpenRouter) を統一処理する `BaseAnnotator` サブクラス。Agent / Provider / Model はキャッシュなし
- `ProviderManager` (`core/provider_manager.py`) - LiteLLM ID から PydanticAI native provider/model を構築し推論実行 (ADR 0023 Phase 1 / async-first)

**Model Categories:**
- **Local ML Models**: ONNX, Transformers, TensorFlow, CLIP-based models
- **Web API Models**: ADR 0023 Phase 1 で `WebApiAnnotator` (`core/webapi_annotator.py`) 1 種に統合。LiteLLM 同梱 DB から discovery され、provider 別ロジックは `provider_manager.py` 内に閉じる。
  - 旧 `model_class/annotator_webapi/{anthropic_api,google_api,openai_api_chat,openai_api_response,pydantic_ai_unified}.py` および `model_class/pydantic_ai_webapi_annotator.py` は Phase 1.x (Issue #35) で削除済み。
  - WebAPI 推論で使う prompt は `model_class/annotator_webapi/webapi_shared.py:BASE_PROMPT` (`provider_manager.py` から参照)。
- **Specialized Models**: DeepDanbooru taggers, aesthetic scorers, captioning models

**Key Design Principles:**
- Unified API via `annotate()` function for all model types
- pHash-based result mapping (image -> results) for robust batch processing
- Memory-aware model loading with automatic cache management
- Framework-agnostic base classes with shared functionality
- **WebAPI 系は `WebApiAnnotator` 1 種に統一** - Agent / Provider / Model はキャッシュせず推論呼び出しごとに新規作成 (ADR 0023 Phase 1)

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

### Web API Integration (ADR 0023 Phase 1 / Phase 1.x)

**統一 WebAPI Architecture:**
- `WebApiAnnotator` (`core/webapi_annotator.py`) — direct LiteLLM ID (`google/gemini-...`) と registry 登録 WebAPI モデル双方を扱う唯一の `BaseAnnotator` サブクラス
- `ProviderManager` (`core/provider_manager.py`) — LiteLLM ID 解析 + PydanticAI native provider/model 構築 + 推論実行。`run_inference_with_model_async()` が中核実装、sync wrapper は running event loop 内で `InferenceError`
- Agent / Provider / Model はキャッシュしない (推論呼び出しごとに毎回新規作成)
- `os.environ` mutate 禁止: API key は `api_keys: dict[str, str]` 経由で provider object に明示注入のみ
- Capability 判定 (`supports_vision` + `supports_function_calling`) は discovery / registry 段階で完結 (Issue #45)。推論層は capability 前提で動作する

**Implementation Pattern:**
- 単一 base class `WebApiAnnotator` で OpenAI / Anthropic / Google / OpenRouter を統一処理
- Structured output は PydanticAI default Tool Output で得る (Pydantic schema = `AnnotationSchema` in `core/types.py`)
- Provider 別差異 (`openrouter:` prefix 等) は `core/model_id.py` の `_BUILDER_DISPATCH` テーブルに集約
- **API Model Discovery** (`core/api_model_discovery.py`) — LiteLLM 同梱 DB から runtime に WebAPI モデル一覧と capability metadata を取得 (TOML キャッシュなし)。絞り込み主条件は `supports_vision` + `supports_function_calling` (Issue #45)。`supports_response_schema` は判定に使わない
- WebAPI モデル登録は `core/registry.py:_register_webapi_models_from_discovery()` が `WebApiAnnotator` を直接 registry に entry する (旧 `PydanticAIWebAPIAnnotator` 経由は Phase 1.x で廃止)

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
1. **ローカル ML モデル**: 適切な framework base class (`TransformersBaseAnnotator` / `ONNXBaseAnnotator` / 等) を継承して実装
2. **WebAPI モデル**: 通常は `WebApiAnnotator` (`core/webapi_annotator.py`) で自動対応される。LiteLLM 同梱 DB に登録された vision-capable モデルは `_register_webapi_models_from_discovery()` が起動時に自動 registry 登録する。LoRAIro 側からは `model_name_list=["openai/gpt-4o", ...]` のように直接 LiteLLM ID を渡せる。
3. **新 provider 対応**: `core/model_id.py:_BUILDER_DISPATCH` テーブルに provider 別 builder を追加するだけで完結 (allowlist 編集不要)
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
- All models should specify `estimated_size_gb` in config (ローカル ML モデルのみ)
- Use `ModelLoad` for all model loading operations (ローカル ML モデルのみ)
- WebAPI モデルは cloud 推論のため device / memory 管理は不要 (`WebApiAnnotator.device = "api"`)
- ローカル ML モデルは ML 系 base class (`TransformersBaseAnnotator` 等) が `__init__` で `determine_effective_device` を呼び CPU フォールバックを行う (Issue #35 で `BaseAnnotator` から責務移譲)

**PydanticAI-Specific Guidelines:**
- WebAPI 推論は `provider_manager.run_inference_with_model_async()` 経由で実行 (Agent / Provider / Model はキャッシュなし、毎回新規作成)
- API key は `api_keys: dict[str, str]` 経由で provider object に明示注入 (`os.environ` mutate 禁止)
- 画像入力は `core/image_preprocess.preprocess_images_to_binary()` で `BinaryContent` 化
- Structured output は `await agent.run([prompt_text, binary_content])` の sequence 形式

### WebAPI Inference Architecture (ADR 0023 Phase 1)

> **Design Authority:**
> WebAPI 推論経路の責務分離 (PydanticAI / LiteLLM / image-annotator-lib) は LoRAIro 側 ADR が SSoT。
> 詳細仕様は [LoRAIro ADR 0023 — PydanticAI / LiteLLM WebAPI Inference Boundary](https://github.com/NEXTAltair/LoRAIro/blob/main/docs/decisions/0023-pydanticai-litellm-webapi-inference-boundary.md) を参照。
> 関連 ISSUE: [#37 (ADR)](https://github.com/NEXTAltair/image-annotator-lib/issues/37) /
> [#36 (Phase 1 実装)](https://github.com/NEXTAltair/image-annotator-lib/issues/36) /
> [#35 (device 判定分離)](https://github.com/NEXTAltair/image-annotator-lib/issues/35) /
> [#42 (refusal 例外階層)](https://github.com/NEXTAltair/image-annotator-lib/issues/42) /
> [#41 (litellm_model_id rename)](https://github.com/NEXTAltair/image-annotator-lib/issues/41) /
> [#45 (function_calling 主条件)](https://github.com/NEXTAltair/image-annotator-lib/issues/45)

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
   - Capability check は不要 (Issue #45 で discovery 段階に集約済み)
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
  - `InferenceError` — PydanticAI 実行時エラーの wrap
  - `SafetyRefusalError` / `ContentPolicyRefusalError` — provider safety/content refusal (Phase 1.5 Issue #42)

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
