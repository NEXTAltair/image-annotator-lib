"""Unit tests for SimplifiedAgentWrapper (simplified_agent_wrapper.py)

Phase C Task: Achieve 26% → 75%+ coverage for simplified_agent_wrapper.py

Test Strategy:
- REAL components: Image preprocessing, tag extraction, dict formatting
- MOCKED: PydanticAI Agent, async event loops (avoid real async contexts)
- Safety: Mock asyncio.new_event_loop() to prevent OS-level threading issues
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic_ai.messages import BinaryContent

from image_annotator_lib.core.simplified_agent_wrapper import SimplifiedAgentWrapper

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_pydantic_ai_agent():
    """Mock PydanticAI Agent for wrapper tests."""
    mock_agent = MagicMock()
    mock_agent.run_sync = MagicMock()
    mock_agent.run = AsyncMock()
    return mock_agent


@pytest.fixture
def mock_agent_result_with_tags():
    """Mock Agent result with tags attribute."""
    mock_result = MagicMock()
    mock_result.tags = ["mock_tag_1", "mock_tag_2", "mock_tag_3"]
    return mock_result


# ==============================================================================
# Priority 1C: Simplified Agent Wrapper - Test 10-15
# ==============================================================================


class TestSimplifiedWrapperInitialization:
    """Initialization and setup tests for SimplifiedAgentWrapper."""

    @pytest.mark.unit
    def test_simplified_wrapper_initialization_and_setup(
        self, mock_pydantic_ai_agent, managed_config_registry
    ):
        """Test __init__ and _setup_agent() initialization.

        Coverage: Lines 20-41 (__init__, _setup_agent)

        REAL components:
        - Real model_id assignment
        - Real BaseAnnotator init

        MOCKED:
        - get_agent_factory().get_cached_agent()

        Scenario:
        1. Mock agent factory to return test agent
        2. Initialize SimplifiedAgentWrapper
        3. Verify _agent set correctly
        4. Verify model_id stored

        Assertions:
        - _agent assigned to mock agent
        - model_id correctly set
        - BaseAnnotator initialized (has model_name attribute)
        """
        # Setup: Register minimal config for test model (BaseAnnotator needs this)
        managed_config_registry.set(
            "test/model-id",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model-id",
                "api_model_id": "test/model-id",
                "api_key": "test_key",
            },
        )

        # Mock agent factory
        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            # Act: Initialize wrapper
            wrapper = SimplifiedAgentWrapper(model_id="test/model-id")

            # Assert: Agent set correctly
            assert wrapper._agent is mock_pydantic_ai_agent, "_agent正しく割り当て"
            assert wrapper.model_id == "test/model-id", "model_id正しく保存"

            # Assert: BaseAnnotator initialized
            assert wrapper.model_name == "test/model-id", "BaseAnnotator初期化（model_name設定）"

            # Assert: get_cached_agent called with correct model_id
            mock_factory_instance.get_cached_agent.assert_called_once_with("test/model-id")


class TestSimplifiedWrapperContextManager:
    """Context manager tests for SimplifiedAgentWrapper."""

    @pytest.mark.unit
    def test_simplified_wrapper_context_manager(self, mock_pydantic_ai_agent, managed_config_registry):
        """Test context manager __enter__/__exit__ flow.

        Coverage: Lines 43-52 (__enter__/__exit__)

        REAL components:
        - Real context manager flow

        MOCKED:
        - Agent instance

        Scenario:
        1. Initialize wrapper
        2. Enter context manager
        3. Exit context manager
        4. Verify no exceptions

        Assertions:
        - __enter__ returns self
        - __exit__ completes without exceptions
        """
        # Setup: Register minimal config for test model
        managed_config_registry.set(
            "test/model",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model",
                "api_model_id": "test/model",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            # Act: Use context manager
            wrapper = SimplifiedAgentWrapper(model_id="test/model")

            with wrapper as ctx:
                # Assert: __enter__ returns self
                assert ctx is wrapper, "__enter__はselfを返す"

            # Assert: __exit__ completed (no exceptions)
            # If we reach here, __exit__ succeeded without raising


class TestSimplifiedWrapperImagePreprocessing:
    """Image preprocessing tests for SimplifiedAgentWrapper."""

    @pytest.mark.unit
    def test_simplified_wrapper_preprocess_images_to_binary(
        self, mock_pydantic_ai_agent, lightweight_test_images, managed_config_registry
    ):
        """Test PIL.Image → BinaryContent conversion.

        Coverage: Lines 54-64, 128-136 (_preprocess_images, _pil_to_binary_content)

        REAL components:
        - Real PIL operations (BytesIO, Image.save)
        - Real BinaryContent creation

        MOCKED:
        - None (real PIL operations)

        Scenario:
        1. Initialize wrapper
        2. Call _preprocess_images with PIL images
        3. Verify BinaryContent objects created
        4. Verify correct media_type and data

        Assertions:
        - BytesIO used for conversion
        - PNG format used
        - BinaryContent created with correct media_type
        - Data is bytes
        """
        # Setup: Register minimal config for test model
        managed_config_registry.set(
            "test/model",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model",
                "api_model_id": "test/model",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            wrapper = SimplifiedAgentWrapper(model_id="test/model")

            # Act: Preprocess images
            binary_contents = wrapper._preprocess_images(lightweight_test_images[:2])

            # Assert: BinaryContent objects created
            assert len(binary_contents) == 2, "2つのBinaryContent作成"

            for bc in binary_contents:
                assert isinstance(bc, BinaryContent), "BinaryContent型"
                assert bc.media_type == "image/png", "media_type: image/png"
                assert isinstance(bc.data, bytes), "dataはbytes型"
                assert len(bc.data) > 0, "dataに内容あり"


class TestSimplifiedWrapperInference:
    """Inference execution tests for SimplifiedAgentWrapper."""

    @pytest.mark.unit
    def test_simplified_wrapper_run_inference_sync(
        self,
        mock_pydantic_ai_agent,
        mock_agent_result_with_tags,
        lightweight_test_images,
        managed_config_registry,
    ):
        """Test successful sync inference execution.

        Coverage: Lines 66-147 (_run_inference, _run_agent_inference sync path)

        REAL components:
        - Real sync execution path

        MOCKED:
        - agent.run_sync() returns mock result

        Scenario:
        1. Initialize wrapper
        2. Mock agent.run_sync() success
        3. Call _run_inference
        4. Verify run_sync called
        5. Verify result returned

        Assertions:
        - run_sync called with BinaryContent
        - Result returned from agent
        """
        # Setup: Register minimal config for test model
        managed_config_registry.set(
            "test/model",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model",
                "api_model_id": "test/model",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            # Mock run_sync success
            mock_pydantic_ai_agent.run_sync.return_value = mock_agent_result_with_tags

            wrapper = SimplifiedAgentWrapper(model_id="test/model")

            # Preprocess images
            binary_contents = wrapper._preprocess_images(lightweight_test_images[:1])

            # Act: Run inference
            results = wrapper._run_inference(binary_contents)

            # Assert: run_sync called
            assert mock_pydantic_ai_agent.run_sync.called, "agent.run_sync()呼び出し"

            # Assert: Results returned
            assert len(results) == 1, "1つの結果返却"
            assert results[0] is mock_agent_result_with_tags, "run_syncの結果返却"

    @pytest.mark.unit
    def test_simplified_wrapper_run_inference_async_fallback(
        self, mock_pydantic_ai_agent, mock_agent_result_with_tags, managed_config_registry
    ):
        """Test async fallback when sync fails with event loop error.

        Coverage: Lines 148-175 (_run_async_with_new_loop)

        REAL components:
        - Real fallback logic flow

        MOCKED:
        - RuntimeError on run_sync ("Event loop")
        - asyncio.new_event_loop() returns mock loop
        - concurrent.futures.ThreadPoolExecutor patched
        - Success on async path

        Scenario:
        1. Initialize wrapper
        2. Mock run_sync to raise RuntimeError with "Event loop" message
        3. Mock new_event_loop to return mock loop
        4. Call _run_agent_inference
        5. Verify fallback to async path
        6. Verify loop.run_until_complete called
        7. Verify loop.close called

        Assertions:
        - new_event_loop() called when sync fails
        - set_event_loop() called with new loop
        - loop.run_until_complete() called
        - loop.close() called in finally block
        - ThreadPoolExecutor used correctly

        Safety:
        - Mock event loop to avoid real async context issues
        """
        # Setup: Register minimal config for test model
        managed_config_registry.set(
            "test/model",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model",
                "api_model_id": "test/model",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            # Mock run_sync to fail with event loop error
            mock_pydantic_ai_agent.run_sync.side_effect = RuntimeError("Event loop is already running")

            # Mock async path to succeed
            mock_pydantic_ai_agent.run.return_value = mock_agent_result_with_tags

            wrapper = SimplifiedAgentWrapper(model_id="test/model")

            # Mock asyncio.new_event_loop
            mock_loop = MagicMock()
            mock_loop.run_until_complete.return_value = mock_agent_result_with_tags

            # Mock ThreadPoolExecutor to execute function synchronously
            with patch(
                "image_annotator_lib.core.simplified_agent_wrapper.asyncio.new_event_loop",
                return_value=mock_loop,
            ) as mock_new_loop:
                with patch(
                    "image_annotator_lib.core.simplified_agent_wrapper.asyncio.set_event_loop"
                ) as mock_set_loop:
                    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_class:
                        # Make executor.submit() execute function immediately
                        def submit_side_effect(func):
                            """Execute the submitted function and return a mock future."""
                            result = func()  # Actually execute the function
                            future = MagicMock()
                            future.result.return_value = result
                            return future

                        mock_executor = MagicMock()
                        mock_executor.submit.side_effect = submit_side_effect
                        mock_executor.__enter__.return_value = mock_executor
                        mock_executor.__exit__.return_value = None
                        mock_executor_class.return_value = mock_executor

                        # Create BinaryContent for testing
                        test_image = Image.new("RGB", (64, 64), "red")
                        binary_content = wrapper._pil_to_binary_content(test_image)

                        # Act: Run inference (should trigger async fallback)
                        result = wrapper._run_agent_inference(binary_content)

                        # Assert: new_event_loop called
                        mock_new_loop.assert_called_once_with()

                        # Assert: set_event_loop called with new loop
                        mock_set_loop.assert_called_once_with(mock_loop)

                        # Assert: run_until_complete called
                        mock_loop.run_until_complete.assert_called_once()

                        # Assert: loop.close called
                        mock_loop.close.assert_called_once()

                        # Assert: ThreadPoolExecutor used
                        mock_executor_class.assert_called_once()
                        mock_executor.submit.assert_called_once()

                        # Assert: Result returned
                        assert result is mock_agent_result_with_tags, "非同期パスの結果返却"


class TestSimplifiedWrapperFormatting:
    """Output formatting tests for SimplifiedAgentWrapper."""

    @pytest.mark.unit
    def test_simplified_wrapper_format_output_and_tags(
        self, mock_pydantic_ai_agent, mock_agent_result_with_tags, managed_config_registry
    ):
        """Test tag extraction and output formatting.

        Coverage: Lines 85-111, 113-126, 177-193 (_format_predictions, _generate_tags, _format_output)

        REAL components:
        - Real tag extraction logic
        - Real dict formatting

        MOCKED:
        - Agent result with tags attribute

        Scenario:
        1. Initialize wrapper
        2. Call _format_predictions with mock agent results
        3. Verify tags extracted
        4. Verify dict formatted correctly

        Assertions:
        - Tags list extracted from result.tags
        - method field set to "simplified_pydantic_ai"
        - tag_count matches tags length
        - model_id included in output
        """
        # Setup: Register minimal config for test model
        managed_config_registry.set(
            "test/model-formatting",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model-formatting",
                "api_model_id": "test/model-formatting",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            wrapper = SimplifiedAgentWrapper(model_id="test/model-formatting")

            # Act: Format predictions
            formatted = wrapper._format_predictions([mock_agent_result_with_tags])

            # Assert: Formatted output structure
            assert len(formatted) == 1, "1つのフォーマット済み結果"
            output = formatted[0]

            # Assert: Tags extracted
            assert output["tags"] == ["mock_tag_1", "mock_tag_2", "mock_tag_3"], "タグ正しく抽出"
            assert output["tag_count"] == 3, "tag_count正しい"

            # Assert: Method field set
            assert output["method"] == "simplified_pydantic_ai", "method: simplified_pydantic_ai"

            # Assert: model_id included
            assert output["model_id"] == "test/model-formatting", "model_id含まれる"

            # Act: Generate tags from formatted output
            tags = wrapper._generate_tags(output)

            # Assert: Tags list returned
            assert tags == ["mock_tag_1", "mock_tag_2", "mock_tag_3"], "_generate_tagsでタグ抽出"

    @pytest.mark.unit
    def test_simplified_wrapper_format_output_no_tags(
        self, mock_pydantic_ai_agent, managed_config_registry
    ):
        """Test formatting when result has no tags.

        Coverage: Lines 98-111 (_format_predictions edge case)

        REAL components:
        - Real empty tags handling

        MOCKED:
        - Agent result without tags attribute

        Scenario:
        1. Initialize wrapper
        2. Pass result without tags attribute
        3. Verify empty tags list used

        Assertions:
        - tags field is empty list
        - tag_count is 0
        - No exception raised
        """
        # Register config BEFORE wrapper initialization
        managed_config_registry.set(
            "test/model-no-tags",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model-no-tags",
                "api_model_id": "test/model-no-tags",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            wrapper = SimplifiedAgentWrapper(model_id="test/model-no-tags")

            # Mock result without tags
            mock_result_no_tags = MagicMock()
            del mock_result_no_tags.tags  # Remove tags attribute

            # Act: Format predictions
            formatted = wrapper._format_predictions([mock_result_no_tags])

            # Assert: Empty tags list
            assert formatted[0]["tags"] == [], "タグなし時は空リスト"
            assert formatted[0]["tag_count"] == 0, "tag_count: 0"


class TestSimplifiedWrapperRunInference:
    """run_inference public method tests."""

    @pytest.mark.unit
    def test_simplified_wrapper_run_inference_success(
        self, mock_pydantic_ai_agent, mock_agent_result_with_tags, managed_config_registry
    ):
        """Test run_inference public method with successful inference.

        Coverage: Lines 195-238 (run_inference method - UPDATED for complete pipeline)

        REAL components:
        - Real AnnotationResult creation
        - Real error handling
        - Real complete 4-step pipeline (preprocess → inference → format → extract)

        MOCKED:
        - Agent operations

        Scenario:
        1. Initialize wrapper with config
        2. Mock successful inference
        3. Call run_inference (tests REAL pipeline)
        4. Verify AnnotationResult returned

        Assertions:
        - AnnotationResult with tags from mock agent
        - formatted_output contains expected fields
        - error is None
        """
        # Register config BEFORE wrapper initialization
        managed_config_registry.set(
            "test/model-public",
            {
                "class": "SimplifiedAgentWrapper",
                "model_name_on_provider": "test/model-public",
                "api_model_id": "test/model-public",
                "api_key": "test_key",
            },
        )

        with patch("image_annotator_lib.core.simplified_agent_wrapper.get_agent_factory") as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_cached_agent.return_value = mock_pydantic_ai_agent
            mock_factory.return_value = mock_factory_instance

            # Mock successful run_sync
            mock_pydantic_ai_agent.run_sync.return_value = mock_agent_result_with_tags

            wrapper = SimplifiedAgentWrapper(model_id="test/model-public")

            # Create test image
            test_image = Image.new("RGB", (64, 64), "blue")

            # Act: Run inference (tests REAL complete pipeline)
            result = wrapper.run_inference(test_image)

            # Assert: AnnotationResult structure (TypedDict doesn't support isinstance)
            assert isinstance(result, dict), "AnnotationResult型はdict"
            assert "tags" in result, "tagsキー存在"
            assert "formatted_output" in result, "formatted_outputキー存在"
            assert "error" in result, "errorキー存在"
            assert result["error"] is None, "エラーなし"

            # Assert: Real tags from mock_agent_result_with_tags
            assert result["tags"] == ["mock_tag_1", "mock_tag_2", "mock_tag_3"], "タグ正しく抽出"

            # Assert: formatted_output structure
            assert "tags" in result["formatted_output"], "formatted_outputにtags存在"
            assert "model_id" in result["formatted_output"], "formatted_outputにmodel_id存在"
            assert "method" in result["formatted_output"], "formatted_outputにmethod存在"
            assert result["formatted_output"]["method"] == "simplified_pydantic_ai"
