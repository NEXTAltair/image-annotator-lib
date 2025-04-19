# Developer Guide for image-annotator-lib

This guide provides information for developers working with or extending the `image-annotator-lib`, covering topics such as adding new models, running tests, and configuring logging.

## How to Add a New Model

This section provides a comprehensive guide for developers on how to add a new image annotation model (including traditional ML models and Web API based models) to the `image-annotator-lib`.

### 1. Understand the Architecture and Class Hierarchy

The library employs a 3-layer class hierarchy to manage different types of annotators while minimizing code duplication. Understanding this structure is crucial before implementing a new model.

**Class Hierarchy:**

1.  **`BaseAnnotator` (`core/base.py`)**:
    *   Abstract base class for all annotators (Tagger and Scorer).
    *   Provides common functionality like logger initialization, configuration loading (via `config_registry`), context management (`__enter__`, `__exit__`), **chunk processing for batch inference**, pHash calculation, basic error handling, and **standardized result generation (`_generate_result`)**.
    *   Defines abstract methods that must be implemented by subclasses: `_preprocess_images`, `_run_inference`, `_format_predictions`, and `_generate_tags`.

2.  **Framework/Type-Specific Base Classes (`core/base.py`, `model_class/annotator_webapi.py`)**:
    *   Inherit from `BaseAnnotator`.
    *   Implement common logic for specific ML frameworks (ONNX, Transformers, TensorFlow, CLIP, etc.) or types of annotators (e.g., Web API based).
    *   Handle framework-specific model loading/unloading and implement some of the abstract methods from `BaseAnnotator` that are common within that framework/type.
    *   Examples: `ONNXBaseAnnotator`, `TransformersBaseAnnotator`, `TensorflowBaseAnnotator`, `ClipBaseAnnotator`, `PipelineBaseAnnotator`, **`BaseWebApiAnnotator`**.

3.  **Concrete Model Classes (`models/`, `model_class/annotator_webapi.py`)**:
    *   Inherit from a framework/type-specific base class.
    *   Implement only the logic specific to the individual model. This typically involves implementing the remaining abstract methods (most commonly `_generate_tags`) and handling model-specific initialization and file loading.
    *   Examples: `WDTagger`, `AestheticShadowV2`, `GoogleApiAnnotator`, `OpenAIApiAnnotator`.

### 2. Implement the Concrete Model Class

Create a Python class for your new model in the appropriate directory (`src/image_annotator_lib/models/` for traditional models, `src/image_annotator_lib/model_class/annotator_webapi.py` for Web API models, or a new file if creating a new type).

**Implementation Steps:**

1.  **Choose the appropriate base class**: Select the base class that matches your model's framework or type (e.g., `ONNXBaseAnnotator`, `TransformersBaseAnnotator`, `BaseWebApiAnnotator`).

2.  **Define the class**:
    *   Inherit from the chosen base class.
    *   Implement the `__init__` method:
        *   Call `super().__init__(model_name)` to initialize the parent class. The `model_name` is automatically passed by the factory.
        *   Access model-specific configuration using `config_registry.get(self.model_name, "your_key", default_value)` from the shared `config_registry` instance (`src/image_annotator_lib/core/config.py`). **Do not access `self.config` directly.**
        *   Add any model-specific initialization (e.g., loading tag lists, setting thresholds, initializing API clients for Web API models). For Web API models, API keys should be loaded from environment variables within the base class's `_load_api_key` method, not directly in the concrete class `__init__`.

    *   **Override necessary abstract methods**: The base classes handle most of the workflow (loading, prediction loop, error handling, result structuring). You typically only need to implement or override methods specific to how your model processes data and generates tags/scores.

        *   `_preprocess_images(self, images: list[Image.Image]) -> Any`: Preprocess a list of PIL Images into the format the model expects. The return type depends on the model/framework (e.g., tensors, byte lists for some Web APIs, Base64 strings for others).
        *   `_run_inference(self, processed: Any) -> Any`: Run inference using the preprocessed data. This method should return the raw model outputs. For Web API models, this involves making the API call.
        *   `_format_predictions(self, raw_outputs: Any) -> list[Any]`: Format the raw model outputs into a standardized list or structure that is easier for `_generate_tags` to consume. This step is optional if `_generate_tags` can directly process the raw output.
        *   `_generate_tags(self, formatted_output: Any) -> list[str]`: Generate the final list of tags (or score strings like `["[SCORE]0.95"]`) from the formatted output. **This is often the main method you need to implement.** The base class's `predict` method will use the output of this method to construct the final `AnnotationResult`.

    *   **Do NOT typically override `_generate_result`**: The `_generate_result` method is implemented in `BaseAnnotator` to create the standardized `AnnotationResult` TypedDict. It takes `phash`, `tags` (from `_generate_tags`), `formatted_output` (from `_format_predictions`), and `error` as input. Overriding this method is generally not necessary and is discouraged to maintain result structure consistency.

    *   **Handle Batching**: The `BaseAnnotator.predict` method handles splitting the input images into chunks (batches) and iterating through them. Your `_preprocess_images` and `_run_inference` methods should be designed to accept and process data in batches as provided by the base class.

    *   **Handle Results and Errors**: The `BaseAnnotator.predict` method collects results and errors for each image and model. Errors occurring during `_preprocess_images`, `_run_inference`, `_format_predictions`, or `_generate_tags` should ideally be caught within those methods and returned as part of the result structure if possible, or allowed to propagate as exceptions which the base `predict` method will catch and record in the `error` field of the `AnnotationResult`.

**Example Implementation (Conceptual - ONNX Tagger):**

```python
# src/image_annotator_lib/models/tagger_onnx.py

# Import the shared config registry
from ..core.config import config_registry
from ..core.base import ONNXBaseAnnotator, AnnotationResult # Import AnnotationResult
from PIL import Image
from typing import Any, List, Dict # Import Dict

class MyNewONNXTagger(ONNXBaseAnnotator):
    """My new ONNX Tagger model.""" # Add a docstring

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Access model-specific config using config_registry
        self.my_threshold = config_registry.get(self.model_name, "threshold", 0.5)
        self.tag_names = self._load_tag_names() # Example: load tag names

    def _load_tag_names(self) -> List[str]:
        """Load tag names from a file specified in config."""
        # Access config using config_registry
        tag_file_path = config_registry.get(self.model_name, "tag_file_path")
        if not tag_file_path:
            self.logger.warning("tag_file_path not specified in config.")
            return []
        try:
            with open(tag_file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            self.logger.error(f"Tag file not found at {tag_file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading tag file {tag_file_path}: {e}", exc_info=True)
            return []

    # Implement abstract methods as needed.
    # _preprocess_images and _run_inference might be handled by ONNXBaseAnnotator.
    # You might only need to implement _generate_tags.

    def _generate_tags(self, formatted_output: Dict[str, float]) -> List[str]:
        """Generate tags from formatted model output."""
        final_tags = []
        # formatted_output is assumed to be a dict like {"tag_name": score}
        for tag_name, score in formatted_output.items():
            if score > self.my_threshold:
                final_tags.append(tag_name)
        return final_tags

    # If ONNXBaseAnnotator's _format_predictions is not suitable, override it:
    # def _format_predictions(self, raw_outputs: Any) -> Dict[str, float]:
    #     """Format raw model outputs into a dictionary of tag names and scores."""
    #     # Your formatting logic here
    #     pass

# Example Implementation (Conceptual - Web API Annotator):

# from ..core.base import BaseWebApiAnnotator, AnnotationResult
# from ..core.config import config_registry # Import config_registry
# from ..exceptions.errors import WebApiError # Import exceptions
# from PIL import Image
# from typing import Any, List, Dict

# class MyNewWebApiAnnotator(BaseWebApiAnnotator):
#     """My new Web API Annotator model."""

#     def __init__(self, model_name: str):
#         super().__init__(model_name)
#         # BaseWebApiAnnotator handles API key loading, rate limiting, retries.
#         # Access model-specific config using config_registry
#         self.model_specific_prompt = config_registry.get(self.model_name, "model_specific_prompt")
#         # API client is initialized in __enter__ by BaseWebApiAnnotator

#     # _load_api_key is implemented in BaseWebApiAnnotator

#     # _preprocess_images might be implemented in BaseWebApiAnnotator (e.g., Base64 encoding)
#     # If your API requires a different format (e.g., raw bytes), override it:
#     # def _preprocess_images(self, images: list[Image.Image]) -> list[bytes]:
#     #     # Your image preprocessing logic for the API
#     #     pass

#     def _run_inference(self, processed_images: Any) -> list[Any]:
#         """Run inference using the Web API."""
#         if not self.client:
#              raise WebApiError("API client not initialized.") # Use appropriate exception

#         results = []
#         # Get necessary config values
#         provider_model_name = config_registry.get(self.model_name, "model_name_on_provider")
#         prompt_to_use = self.model_specific_prompt or self.prompt_template # Use specific prompt if available

#         for processed_image_data in processed_images: # Processed data from _preprocess_images
#             try:
#                 self._wait_for_rate_limit() # Handled by BaseWebApiAnnotator
#                 # Make API call using self.client and processed_image_data
#                 # Use provider_model_name and prompt_to_use
#                 # Handle API-specific errors and retry logic (BaseWebApiAnnotator might help)
#                 api_response = self.client.call_api(processed_image_data, provider_model_name, prompt_to_use) # Conceptual call
#                 results.append(api_response)
#             except Exception as e:
#                 self.logger.error(f"API call failed: {e}", exc_info=True)
#                 # Return an error structure or raise an exception
#                 results.append({"error": str(e)}) # Example error structure
#         return results

#     # _format_predictions might be implemented in BaseWebApiAnnotator (e.g., JSON parsing)
#     # If your API response format is unique, override it:
#     # def _format_predictions(self, raw_outputs: list[Any]) -> list[Any]:
#     #     # Your API response parsing logic
#     #     pass

#     def _generate_tags(self, formatted_output: Any) -> List[str]:
#         """Generate tags from formatted API output."""
#         # Your logic to extract tags/scores from the formatted API response
#         pass

#     # __enter__ and __exit__ are typically handled by BaseWebApiAnnotator for client management.
```

### 3. Add Entry to Configuration File

To make the new model available, add an entry for it in the configuration file (`src/image_annotator_lib/resources/system/annotator_config.toml` or the user config file `src/image_annotator_lib/resources/user/annotator_config.toml`). User configuration entries will override system entries if they share the same model name.

**Configuration Fields:**

- **Section Name (`[model_unique_name]`)**: A unique name to identify the model within the library.
- `class` (Required): The name of your implemented concrete model class (as a string).
- `model_path` (Required for local/downloadable models): The path or URL to the model file(s) or repository. For API models, this might be the model identifier on the provider's platform.
- `estimated_size_gb` (Recommended for local/downloadable models): Approximate size of the model in GB for memory management. API models typically have this set to 0.
- `device` (Optional): Specify the device (`"cuda"`, `"cpu"`, etc.) for the model. Defaults based on system availability if not set.
- Other model-specific configuration keys can be added and accessed via `config_registry.get(self.model_name, "your_key", default_value)` within your class.

**Example Entry:**

```toml
# In system/annotator_config.toml or user/annotator_config.toml

[my-new-onnx-tagger-v1]
class = "MyNewONNXTagger"
model_path = "path/to/your/model.onnx"
estimated_size_gb = 0.5
# Optional device override
# device = "cuda"
# Model-specific config
tags_file_path = "path/to/tags.txt"
threshold = 0.45 # Override default threshold
```

### 4. Verify Functionality

After completing the steps above, test if the new model works correctly using the library.

```python
from image_annotator_lib import annotate, list_available_annotators
from PIL import Image

# Check if the new model appears in the list
available = list_available_annotators()
print(available)
assert "my-new-onnx-tagger-v1" in available

# Run a test annotation
img = Image.open("path/to/test/image.jpg")
results = annotate([img], ["my-new-onnx-tagger-v1"]) # Use the unique name from config
print(results)
```

This completes the process of adding a new model. Consider adding unit tests for your new model class as well (see the Testing section below).

---

## How to Run Tests

This section describes how to run the tests for the `image-annotator-lib` project, which uses `pytest` and `pytest-bdd`.

### 1. Preparation

Ensure you have installed the development dependencies. From the project root directory (containing `pyproject.toml`), run:

```bash
# Make sure your virtual environment is activated
# (e.g., source .venv/bin/activate or .venv\Scripts\activate)

# Install the library with development dependencies
uv pip install -e .[dev]
# or: pip install -e .[dev]
```

### 2. Running Tests

Execute tests using the `pytest` command from the project root directory.

#### 2.1. Run All Tests

```bash
pytest
```

This command discovers and runs all tests under the `tests/` directory (and other locations pytest checks by default).

#### 2.2. Run with Verbose Output

Use the `-v` flag for more detailed output, showing individual test function names and results.

```bash
pytest -v
```

#### 2.3. BDD (Gherkin) Formatted Results

For BDD tests using `pytest-bdd`, display results in a Gherkin-like format:

```bash
pytest --gherkin-terminal-reporter
```

This improves readability by showing which scenarios and steps passed or failed.

#### 2.4. Run Tests in Specific Files or Directories

Target specific tests by providing paths:

```bash
# Run tests in a specific file
pytest tests/unit/test_api.py

# Run all tests in a directory
pytest tests/integration/
```

#### 2.5. Run Specific Tests or Scenarios by Name

Use the `-k` option to run tests matching a name expression (substring matching).

```bash
# Run test functions containing 'annotate'
pytest -k annotate

# Run BDD scenarios containing 'model loading'
pytest -k "model loading"
```

(Note: Use quotes if the name expression contains spaces.)

#### 2.6. Measure Test Coverage

If `pytest-cov` is installed (included in `[dev]` dependencies), measure test coverage:

```bash
pytest --cov=src/image_annotator_lib tests/
```

This generates a report showing how much of the code in `src/image_annotator_lib` is covered by tests.

### 3. Running Tests in VSCode (pytest-bdd extension)

If you use VSCode with the "BDD - Cucumber/Gherkin Full Support" extension (`vtenentes.bdd`), you can run BDD scenarios directly from `.feature` files.

- **Run Scenario**: Executes the scenario under the cursor (Command Palette: `BDD: Run Scenario`).
- **Debug Scenario**: Debugs the scenario under the cursor (Command Palette: `BDD: Debug Scenario`).

Refer to the extension's documentation for more details.

---

## How to Configure Logging

This section explains how to configure and use the logger in `image-annotator-lib`. Proper logging helps in debugging and monitoring the library's behavior.

### 1. Basic Logger Usage

The library uses Python's standard `logging` module. It's recommended to get a logger at the module level.

```python
import logging

# Get logger at module level
# Using __name__ makes the logger name the full module path
# (e.g., 'image_annotator_lib.models.tagger_onnx')
logger = logging.getLogger(__name__)

# Example log messages
logger.info("This is an info message.")
logger.debug("This is a debug message.")
logger.warning("This is a warning message.")
```

### 2. Logger Setup (`setup_logger`)

The basic configuration (level, format, handlers) for the library's loggers is handled by the `setup_logger` function in `core/utils.py`. This function is typically called internally during library initialization.

**Key Features of `setup_logger`:**

- Gets or creates a logger with the specified name.
- Sets the logging level (default is `logging.INFO`).
- Sets a standard log format (`%(asctime)s - %(name)s - %(levelname)s - %(message)s`).
- Configures handlers to output logs to both the console (standard output) and a log file (`logs/image_annotator_lib.log`).
  - The `logs/` directory is created automatically if it doesn't exist.
- Prevents duplicate handler setup.

**Excerpt from `core/utils.py`:**

```python
# src/image_annotator_lib/core/utils.py

import logging
from pathlib import Path

LOG_FILE = Path("logs/image_annotator_lib.log") # Log file path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    # Ensures logger level is set appropriately, even if already configured
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)

    # Check if handlers are already configured for this logger to avoid duplication
    if not logger.handlers:
        # Also check the root logger if propagating
        # This basic setup assumes direct handling, not relying on root logger config
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # Set level for the handler itself if needed, otherwise inherits logger level
        # stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        # File handler
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8") # Use UTF-8
            file_handler.setFormatter(formatter)
            # file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except OSError as e:
            # Handle potential errors during log file setup (e.g., permission denied)
            logger.error(f"Could not set up log file handler at {LOG_FILE}: {e}", exc_info=False)
            # Continue without file logging if setup fails

    # Ensure propagation is handled as intended (usually True by default)
    # logger.propagate = True

    return logger
```

### 3. Controlling Log Levels

You can adjust the verbosity of the logs by changing the log level. Common levels include:

- **`logging.DEBUG`**: Detailed information, useful for diagnosing problems.
- **`logging.INFO`**: Confirmation that things are working as expected.
- **`logging.WARNING`**: Indication of something unexpected or problematic in the near future (e.g., 'disk space low').
- **`logging.ERROR`**: Due to a more serious problem, the software has not been able to perform some function.
- **`logging.CRITICAL`**: A serious error, indicating that the program itself may be unable to continue running.

To change the log level for the entire library or specific modules when using the library, use the standard `logging` module:

```python
import logging

# Example: Set the root logger for 'image_annotator_lib' to DEBUG
logging.getLogger('image_annotator_lib').setLevel(logging.DEBUG)

# Example: Set a specific model module's logger to DEBUG
logging.getLogger('image_annotator_lib.models.tagger_onnx').setLevel(logging.DEBUG)

# You might also need to adjust handler levels if they were set explicitly
# Example: Find and adjust the console handler level for the library's root logger
lib_logger = logging.getLogger('image_annotator_lib')
for handler in lib_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.DEBUG)
```

### 4. Logging Within Classes

When logging within classes, you can use the module-level logger or get a logger specific to the class name for finer granularity.

```python
import logging

logger = logging.getLogger(__name__) # Module-level logger

class MyAnnotator:
    def __init__(self, model_name: str):
        # Option 1: Use module logger
        logger.info(f"Initializing MyAnnotator for {model_name}")

        # Option 2: Use a class-specific logger (optional)
        # self.class_logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # self.class_logger.info(f"Initializing MyAnnotator instance for {model_name}")

        self.model_name = model_name

    def some_method(self):
        logger.debug(f"Executing some_method in MyAnnotator ({self.model_name})")
        # self.class_logger.debug("Executing some_method")
        try:
            # Perform some operation
            pass
        except Exception as e:
            # Log errors with traceback information
            logger.error(f"Error in some_method: {e}", exc_info=True)
```

### Summary

- Logs are output using the standard `logging` module.
- Basic setup via `core/utils.py:setup_logger` outputs to console and `logs/image_annotator_lib.log`.
- Log levels can be controlled externally using standard `logging` functions.
- Use `logging.getLogger(__name__)` for effective log source tracking.
