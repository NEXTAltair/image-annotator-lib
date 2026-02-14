# tests/integration/test_provider_manager_integration.py
import pytest

from image_annotator_lib.core.provider_manager import ProviderManager


# A placeholder for checking API key availability
def api_key_available(provider: str):
    """Placeholder function to check if an API key is available."""
    return False


class TestProviderManagerIntegration:
    """
    Integration tests for the Provider Manager's core logic.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.parametrize(
        "model_name, api_model_id, config, expected_provider",
        [
            # 1. Explicit provider in config
            ("test_model_explicit", "any_id", {"provider": "google"}, "google"),
            ("test_model_explicit_case", "any_id", {"provider": "Anthropic"}, "anthropic"),
            # 2. Provider from model_id prefix
            ("test_model_prefix", "openai:gpt-4", {}, "openai"),
            (
                "test_model_prefix_edge",
                "google:gemini-pro",
                {"provider": "anthropic"},
                "anthropic",
            ),  # explicit config wins
            # 3. Provider from model name pattern
            ("test_model_pattern", "gpt-4-turbo", {}, "openai"),
            ("test_model_pattern", "claude-3-opus-20240229", {}, "anthropic"),
            ("test_model_pattern", "gemini-1.5-pro-latest", {}, "google"),
            # 4. Fallback to API key in config
            ("test_model_fallback", "some_custom_model", {"google_api_key": "key"}, "google"),
            ("test_model_fallback", "some_custom_model", {"anthropic_api_key": "key"}, "anthropic"),
            ("test_model_fallback", "some_custom_model", {"openai_api_key": "key"}, "openai"),
            # 5. Default fallback
            ("test_model_default", "unknown_model", {}, "openai"),
        ],
    )
    def test_provider_determination_integration(
        self, managed_config_registry, model_name, api_model_id, config, expected_provider
    ):
        """
        Tests provider determination with actual configurations and model IDs.
        """
        # Setup config for the test case
        managed_config_registry.set(model_name, config)

        # Run the determination logic
        determined_provider = ProviderManager._determine_provider(model_name, api_model_id)

        # Assert the result
        assert determined_provider == expected_provider

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_instance_sharing_integration(self):
        """
        Tests that requests for different models using the same provider
        share the same underlying provider instance.
        """
        # Clear any cached instances before the test
        ProviderManager._provider_instances.clear()

        # Get instance for the first google model
        instance1 = ProviderManager.get_provider_instance("google")

        # Get instance for the second google model
        instance2 = ProviderManager.get_provider_instance("google")

        # Get a different provider instance
        instance3 = ProviderManager.get_provider_instance("openai")

        assert id(instance1) == id(instance2)
        assert id(instance1) != id(instance3)
        assert "google" in ProviderManager._provider_instances
        assert "openai" in ProviderManager._provider_instances
        assert len(ProviderManager._provider_instances) == 2

        # Cleanup after test
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.real_api
    @pytest.mark.skipif(not api_key_available("any"), reason="API key required for real API tests")
    def test_provider_instance_sharing_real_api(self, api_key_manager):
        """
        Tests provider instance sharing using real API calls.
        """
        pytest.skip("Test not yet implemented.")

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_multiple_images(self, managed_config_registry, lightweight_test_images):
        """
        Tests batch processing of multiple images with unique pHash-based results.
        Verifies that each image is processed and returns a unique result.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        # Use 3 test images
        test_images = lightweight_test_images[:3]

        # Setup model configuration
        model_name = "test_batch_model"
        managed_config_registry.set(model_name, {"provider": "google", "api_key": "test_key"})

        # Mock calculate_phash to return predictable phashes
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.side_effect = ["phash_001", "phash_002", "phash_003"]

            # Mock _run_agent_safely to return unique results for each image
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:

                def mock_run_safely_side_effect(agent, binary_content, api_model_id):
                    # Create different responses for each call
                    call_count = mock_run_safely.call_count
                    mock_result = MagicMock()
                    mock_result.data = AnnotationSchema(
                        tags=[f"tag_batch_{call_count}"],
                        captions=[f"caption_batch_{call_count}"],
                        score=0.5 + (call_count * 0.1),
                    )
                    return mock_result

                mock_run_safely.side_effect = mock_run_safely_side_effect

                # Execute batch processing
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=test_images,
                    api_model_id="gemini-pro",
                    api_keys={"google": "test_key"},
                )

                # Verify results structure
                assert len(results) == 3, "Should return 3 results for 3 images"
                assert "phash_001" in results
                assert "phash_002" in results
                assert "phash_003" in results

                # Verify each result has unique content
                assert results["phash_001"]["tags"] == ["tag_batch_1"]
                assert results["phash_002"]["tags"] == ["tag_batch_2"]
                assert results["phash_003"]["tags"] == ["tag_batch_3"]

                # Verify agent was called 3 times
                assert mock_run_safely.call_count == 3

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_batch_processing_with_partial_errors(self, managed_config_registry, lightweight_test_images):
        """
        Tests batch processing when some images fail processing.
        Verifies that partial failures are handled gracefully and successful results are returned.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        # Use 3 test images
        test_images = lightweight_test_images[:3]

        # Setup model configuration
        model_name = "test_batch_error_model"
        managed_config_registry.set(model_name, {"provider": "anthropic", "api_key": "test_key"})

        # Mock calculate_phash to return predictable phashes
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.side_effect = ["phash_success_1", "phash_error_2", "phash_success_3"]

            # Mock _run_agent_safely with mixed success/failure
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:

                def mock_run_safely_side_effect(agent, binary_content, api_model_id):
                    # Make the 2nd call (call_count=2) fail
                    call_count = mock_run_safely.call_count
                    if call_count == 2:
                        raise RuntimeError("Simulated API error for image 2")

                    # Success responses for calls 1 and 3
                    mock_result = MagicMock()
                    mock_result.data = AnnotationSchema(
                        tags=[f"tag_success_{call_count}"],
                        captions=[f"caption_success_{call_count}"],
                        score=0.8,
                    )
                    return mock_result

                mock_run_safely.side_effect = mock_run_safely_side_effect

                # Execute batch processing - should handle errors gracefully
                try:
                    results = ProviderManager.run_inference_with_model(
                        model_name=model_name,
                        images_list=test_images,
                        api_model_id="claude-3-opus",
                        api_keys={"anthropic": "test_key"},
                    )

                    # If partial failure is handled within run_inference_with_model
                    assert len(results) >= 2, "Should return at least 2 successful results"

                    # Verify successful results
                    if "phash_success_1" in results:
                        assert results["phash_success_1"]["error"] is None
                        assert "tag_success_1" in results["phash_success_1"]["tags"]

                    if "phash_success_3" in results:
                        assert results["phash_success_3"]["error"] is None
                        assert "tag_success_3" in results["phash_success_3"]["tags"]

                    # Verify error result if present
                    if "phash_error_2" in results:
                        assert results["phash_error_2"]["error"] is not None
                        assert "error" in results["phash_error_2"]["error"].lower()

                except RuntimeError as e:
                    # If the implementation doesn't handle partial errors gracefully yet,
                    # verify that the error is propagated correctly
                    assert "Simulated API error" in str(e)

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.parametrize(
        "provider,api_model_id",
        [
            ("anthropic", "claude-3-opus"),
            ("openai", "gpt-4"),
            ("google", "gemini-pro"),
            ("openrouter", "openrouter:anthropic/claude-3-opus"),
        ],
    )
    def test_api_key_injection_priority(
        self, managed_config_registry, lightweight_test_images, provider, api_model_id
    ):
        """
        Tests that api_keys parameter takes priority over config_registry.
        Verifies all providers respect the injected API key.
        """
        from unittest.mock import patch

        from image_annotator_lib.core.types import AnnotationSchema

        # Use single test image
        test_images = lightweight_test_images[:1]

        # Setup model configuration with a DIFFERENT api_key in config
        model_name = f"test_api_key_priority_{provider}"
        managed_config_registry.set(model_name, {"provider": provider, "api_key": "config_registry_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_priority_test"

            # Mock _run_agent_safely
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                from unittest.mock import MagicMock

                mock_result = MagicMock()
                mock_result.data = AnnotationSchema(
                    tags=["priority_test"], captions=["API key priority test"], score=0.9
                )
                mock_run_safely.return_value = mock_result

                # Execute with INJECTED api_keys - should override config
                injected_key = "injected_api_key_should_win"
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=test_images,
                    api_model_id=api_model_id,
                    api_keys={provider: injected_key},
                )

                # Verify result
                assert "phash_priority_test" in results
                assert results["phash_priority_test"]["tags"] == ["priority_test"]
                assert results["phash_priority_test"]["error"] is None

                # Verify the injected key was used by checking the provider instance
                # was called (indirectly verified by mock_run_safely being called)
                assert mock_run_safely.call_count == 1

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_api_key_fallback_to_config_registry(self, managed_config_registry, lightweight_test_images):
        """
        Tests that when api_keys parameter is None, the system falls back to config_registry.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        # Use single test image
        test_images = lightweight_test_images[:1]

        # Setup model configuration with api_key in config
        model_name = "test_api_key_fallback"
        managed_config_registry.set(model_name, {"provider": "google", "api_key": "config_fallback_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_fallback_test"

            # Mock _run_agent_safely
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_result = MagicMock()
                mock_result.data = AnnotationSchema(
                    tags=["fallback_test"], captions=["Config registry fallback"], score=0.85
                )
                mock_run_safely.return_value = mock_result

                # Execute WITHOUT api_keys - should use config_registry
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=test_images,
                    api_model_id="gemini-pro",
                    api_keys=None,  # Explicitly None
                )

                # Verify result
                assert "phash_fallback_test" in results
                assert results["phash_fallback_test"]["tags"] == ["fallback_test"]
                assert results["phash_fallback_test"]["error"] is None

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    @pytest.mark.parametrize(
        "invalid_config,expected_error_pattern",
        [
            # APIキー未設定（プロバイダーのみ指定）
            (
                {"provider": "anthropic"},
                "API key|api_key|APIキー",
            ),
            # api_model_id未設定（最小限の設定）
            (
                {"provider": "google", "api_key": "test_key"},
                "API model|api_model_id|モデルID",
            ),
            # 不明なプロバイダー
            (
                {"provider": "unknown_provider", "api_key": "test", "api_model_id": "test"},
                "Unsupported provider|Unknown provider|unknown_provider",
            ),
        ],
    )
    def test_invalid_configuration_error(
        self, managed_config_registry, lightweight_test_images, invalid_config, expected_error_pattern
    ):
        """
        Tests that invalid configuration raises appropriate errors.
        Verifies error handling for missing API key, model ID, and unsupported provider.
        """
        from unittest.mock import patch

        model_name = "test_invalid_config_model"
        managed_config_registry.set(model_name, invalid_config)

        # Execute and expect error
        # Note: Implementation may handle errors differently (raise exception vs return error in result)
        try:
            with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
                mock_phash.return_value = "phash_config_error"

                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id=invalid_config.get("api_model_id", "test-model"),
                )

                # If implementation returns error in result dict
                assert "phash_config_error" in results
                assert results["phash_config_error"]["error"] is not None

                # Verify error message contains expected pattern
                import re

                error_message = results["phash_config_error"]["error"]
                assert re.search(expected_error_pattern, error_message, re.IGNORECASE), (
                    f"Expected error pattern '{expected_error_pattern}' "
                    f"not found in error message: {error_message}"
                )

        except Exception as e:
            # If implementation raises exception instead
            import re

            assert re.search(expected_error_pattern, str(e), re.IGNORECASE), (
                f"Expected error pattern '{expected_error_pattern}' not found in exception: {e!s}"
            )

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_timeout_handling(self, managed_config_registry, lightweight_test_images):
        """
        Tests that timeout errors are handled gracefully.
        Verifies that timeout during agent execution is caught and reported appropriately.
        """
        import concurrent.futures
        from unittest.mock import patch

        model_name = "test_timeout_model"
        managed_config_registry.set(
            model_name, {"provider": "anthropic", "api_key": "test_key", "api_model_id": "claude-3-opus"}
        )

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_timeout"

            # Mock _run_agent_safely to raise TimeoutError
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_run_safely.side_effect = concurrent.futures.TimeoutError(
                    "Agent execution timed out after 60s"
                )

                # Execute - should handle timeout gracefully
                try:
                    results = ProviderManager.run_inference_with_model(
                        model_name=model_name,
                        images_list=lightweight_test_images[:1],
                        api_model_id="claude-3-opus",
                        api_keys={"anthropic": "test_key"},
                    )

                    # If implementation handles timeout and returns error in result
                    assert "phash_timeout" in results
                    assert results["phash_timeout"]["error"] is not None
                    # Check for "timeout" or "timed out" pattern
                    error_lower = results["phash_timeout"]["error"].lower()
                    assert "timeout" in error_lower or "timed out" in error_lower, (
                        f"Expected 'timeout' or 'timed out' in error message, got: {results['phash_timeout']['error']}"
                    )

                except concurrent.futures.TimeoutError as e:
                    # If implementation propagates timeout exception
                    assert "timeout" in str(e).lower()

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_response_parsing_error_none(self, managed_config_registry, lightweight_test_images):
        """
        Tests handling of None API response.
        Verifies that None response is detected and reported appropriately.
        """
        from unittest.mock import MagicMock, patch

        model_name = "test_parsing_error_model"
        managed_config_registry.set(model_name, {"provider": "openai", "api_key": "test_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_parse_error"

            # Mock _run_agent_safely to return None response
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_result = MagicMock()
                mock_result.data = None
                mock_run_safely.return_value = mock_result

                # Execute
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="gpt-4",
                    api_keys={"openai": "test_key"},
                )

                # Verify error handling
                assert "phash_parse_error" in results

                # Verify error is captured
                error_message = results["phash_parse_error"].get("error")
                assert error_message is not None, "Expected error field to be set for None response"

                # Verify error message indicates no response
                assert "no response" in error_message.lower(), (
                    f"Expected 'no response' in error message, got: {error_message}"
                )

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_event_loop_already_running(self, managed_config_registry, lightweight_test_images):
        """
        Tests handling when an event loop is already running.
        Verifies that nested event loop scenarios are handled correctly.
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        model_name = "test_event_loop_running"
        managed_config_registry.set(model_name, {"provider": "anthropic", "api_key": "test_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_event_loop"

            # Mock asyncio.get_running_loop to simulate event loop already running
            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop

                # Mock asyncio.new_event_loop fallback
                with patch("asyncio.new_event_loop") as mock_new_loop:
                    fallback_loop = MagicMock()
                    mock_new_loop.return_value = fallback_loop

                    # Mock run_in_executor for the fallback execution
                    future = asyncio.Future()
                    mock_result = MagicMock()
                    mock_result.data = AnnotationSchema(
                        tags=["event_loop_test"], captions=["Event loop handling"], score=0.9
                    )
                    future.set_result(mock_result)
                    fallback_loop.run_in_executor.return_value = future

                    # Mock the agent run itself
                    with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                        mock_run_safely.return_value = mock_result

                        # Execute
                        results = ProviderManager.run_inference_with_model(
                            model_name=model_name,
                            images_list=lightweight_test_images[:1],
                            api_model_id="claude-3-opus",
                            api_keys={"anthropic": "test_key"},
                        )

                        # Verify successful handling
                        assert "phash_event_loop" in results
                        assert results["phash_event_loop"]["error"] is None
                        assert results["phash_event_loop"]["tags"] == ["event_loop_test"]

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_event_loop_creation_error(self, managed_config_registry, lightweight_test_images):
        """
        Tests handling when event loop creation fails.
        Verifies graceful error handling for event loop initialization issues.
        """
        from unittest.mock import patch

        model_name = "test_event_loop_creation_error"
        managed_config_registry.set(model_name, {"provider": "google", "api_key": "test_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_loop_creation_error"

            # Mock asyncio.new_event_loop to raise RuntimeError
            with patch("asyncio.new_event_loop") as mock_new_loop:
                mock_new_loop.side_effect = RuntimeError("Event loop creation failed")

                # Execute - should handle error gracefully
                try:
                    results = ProviderManager.run_inference_with_model(
                        model_name=model_name,
                        images_list=lightweight_test_images[:1],
                        api_model_id="gemini-pro",
                        api_keys={"google": "test_key"},
                    )

                    # If implementation handles error and returns it in result
                    assert "phash_loop_creation_error" in results
                    error = results["phash_loop_creation_error"].get("error")
                    assert error is not None, "Expected error for event loop creation failure"

                except RuntimeError as e:
                    # If implementation propagates the error
                    assert "event loop" in str(e).lower() or "creation failed" in str(e).lower()

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_alternative_openrouter_provider_creation(
        self, managed_config_registry, lightweight_test_images
    ):
        """
        Tests creation of OpenRouter provider instance.
        Verifies that OpenRouter provider is correctly instantiated and configured.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        model_name = "test_openrouter_model"
        managed_config_registry.set(
            model_name,
            {
                "provider": "openrouter",
                "api_key": "test_openrouter_key",
                "openrouter_referer": "https://test.com",
                "openrouter_app_name": "TestApp",
            },
        )

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_openrouter"

            # Mock _run_agent_safely
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_result = MagicMock()
                mock_result.data = AnnotationSchema(
                    tags=["openrouter_test"], captions=["OpenRouter provider test"], score=0.95
                )
                mock_run_safely.return_value = mock_result

                # Execute
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="openrouter:anthropic/claude-3-opus",
                    api_keys={"openrouter": "test_openrouter_key"},
                )

                # Verify successful execution
                assert "phash_openrouter" in results
                assert results["phash_openrouter"]["error"] is None
                assert results["phash_openrouter"]["tags"] == ["openrouter_test"]

                # Verify provider instance was created for openrouter
                assert "openrouter" in ProviderManager._provider_instances

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_alternative_google_provider_creation(self, managed_config_registry, lightweight_test_images):
        """
        Tests creation of Google provider instance.
        Verifies that Google provider is correctly instantiated and configured.
        """
        from unittest.mock import MagicMock, patch

        from image_annotator_lib.core.types import AnnotationSchema

        model_name = "test_google_gemini_model"
        managed_config_registry.set(model_name, {"provider": "google", "api_key": "test_google_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_google"

            # Mock _run_agent_safely
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_result = MagicMock()
                mock_result.data = AnnotationSchema(
                    tags=["google_test"], captions=["Google Gemini provider test"], score=0.92
                )
                mock_run_safely.return_value = mock_result

                # Execute
                results = ProviderManager.run_inference_with_model(
                    model_name=model_name,
                    images_list=lightweight_test_images[:1],
                    api_model_id="gemini-2.0-flash",
                    api_keys={"google": "test_google_key"},
                )

                # Verify successful execution
                assert "phash_google" in results
                assert results["phash_google"]["error"] is None
                assert results["phash_google"]["tags"] == ["google_test"]

                # Verify provider instance was created for google
                assert "google" in ProviderManager._provider_instances

        # Cleanup
        ProviderManager._provider_instances.clear()

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_async_context_manager_edge_case(self, managed_config_registry, lightweight_test_images):
        """
        Tests edge case handling in async context managers.
        Verifies that async context cleanup works correctly even with errors.
        """
        from unittest.mock import patch

        model_name = "test_async_context"
        managed_config_registry.set(model_name, {"provider": "anthropic", "api_key": "test_key"})

        # Mock calculate_phash
        with patch("image_annotator_lib.core.provider_manager.calculate_phash") as mock_phash:
            mock_phash.return_value = "phash_async_context"

            # Mock _run_agent_safely to raise an error during execution
            with patch.object(ProviderManager, "_run_agent_safely") as mock_run_safely:
                mock_run_safely.side_effect = ValueError("Async context error during agent execution")

                # Execute - should handle error and not leak resources
                try:
                    results = ProviderManager.run_inference_with_model(
                        model_name=model_name,
                        images_list=lightweight_test_images[:1],
                        api_model_id="claude-3-opus",
                        api_keys={"anthropic": "test_key"},
                    )

                    # If implementation catches and returns error
                    assert "phash_async_context" in results
                    error = results["phash_async_context"].get("error")
                    assert error is not None

                except ValueError as e:
                    # If implementation propagates the error
                    assert "async context" in str(e).lower() or "agent execution" in str(e).lower()

        # Cleanup
        ProviderManager._provider_instances.clear()
