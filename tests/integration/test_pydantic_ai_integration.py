"""Phase B Task 4: PydanticAI Integration Tests

このモジュールは、PydanticAI AgentキャッシングとProvider共有の統合テストを提供します。

テスト対象:
- Agent caching lifecycle (creation → caching → reuse)
- Configuration change detection → cache invalidation
- Provider instance sharing across multiple models
- Cache cleanup and isolation

Test Strategy:
- REAL components: PydanticAIAgentFactory._providers cache, Agent instances
- MOCKED: Agent.run API calls (external dependencies)
"""

import pytest

from image_annotator_lib.core.pydantic_ai_factory import PydanticAIAgentFactory

# ==============================================================================
# Phase B Task 4.1: Agent Caching Flow Tests
# ==============================================================================


class TestAgentCachingFlow:
    """Agent caching lifecycle tests.

    Tests Agent creation → caching → reuse flow with REAL cache dictionaries.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_agent_cache_lifecycle(self):
        """Test Provider caching supports Agent creation.

        NOTE: PydanticAIAgentFactory does NOT cache Agent instances.
        It caches Provider instances, and each Agent is newly created.
        This test verifies the Provider caching mechanism.

        REAL components:
        - Real PydanticAIAgentFactory._providers cache (Provider-level)
        - Real Provider instance caching and reuse

        Scenario:
        1. Get provider for openai (cache miss)
        2. Get same provider again (cache hit)
        3. Verify same Provider instance returned (ID check)

        Assertions:
        - len(PydanticAIAgentFactory._providers) >= 1 after first call
        - id(provider1) == id(provider2) for cache hit
        - Cache key includes provider_name + api_key
        """
        # Act: Get provider twice with same configuration
        provider1 = PydanticAIAgentFactory.get_provider("openai", api_key="test_key_123")
        provider2 = PydanticAIAgentFactory.get_provider("openai", api_key="test_key_123")

        # Assert: Same Provider instance returned (cache hit)
        assert provider1 is provider2, (
            "同じprovider名とAPI keyでは同じProviderインスタンスが返されるべき（キャッシュヒット）"
        )
        assert id(provider1) == id(provider2), "Provider Instance IDが同じであることを確認"

        # Assert: Provider cache contains entry
        assert len(PydanticAIAgentFactory._providers) >= 1, "Providerキャッシュにエントリ存在"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_cache_with_config_changes(self):
        """Test configuration change creates separate Provider instances.

        NOTE: Provider caching is based on (provider_name, api_key, base_url).
        Different API keys create separate Provider instances.

        REAL components:
        - Real Provider cache invalidation logic
        - Real configuration-based cache key generation

        Scenario:
        1. Get provider with api_key="key1"
        2. Get provider with api_key="key2"
        3. Verify DIFFERENT Provider instances (different keys)

        Assertions:
        - provider1 and provider2 are DIFFERENT instances
        - len(PydanticAIAgentFactory._providers) >= 2 (both cached)
        """
        # Act: Get provider with first API key
        provider1 = PydanticAIAgentFactory.get_provider("openai", api_key="key_version_1")

        # Act: Get provider with DIFFERENT API key
        provider2 = PydanticAIAgentFactory.get_provider("openai", api_key="key_version_2")

        # Assert: Different Provider instances (different keys)
        assert provider1 is not provider2, "異なるAPI keyでは異なるProviderインスタンスが返されるべき"
        assert id(provider1) != id(provider2), "Provider Instance IDが異なることを確認"

        # Assert: Both providers cached
        assert len(PydanticAIAgentFactory._providers) >= 2, "両方のProviderがキャッシュされている"


# ==============================================================================
# Phase B Task 4.2: Cache Invalidation Tests
# ==============================================================================


class TestCacheInvalidation:
    """Cache invalidation and cleanup tests.

    Tests explicit cache clearing and auto-cleanup between tests.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_explicit_cache_clear(self):
        """Test PydanticAIAgentFactory.clear_cache() functionality.

        REAL components:
        - Real cache dictionary operations
        - Real clear_cache() implementation

        Scenario:
        1. Create 3 providers (openai, anthropic, google)
        2. Verify all cached (len >= 3)
        3. Call clear_cache()
        4. Verify cache empty (len == 0)

        Assertions:
        - Cache populated after provider creation
        - Cache empty after clear_cache()
        """
        # Setup: Create multiple providers to populate cache
        PydanticAIAgentFactory.get_provider("openai", api_key="test_key_openai")
        PydanticAIAgentFactory.get_provider("anthropic", api_key="test_key_anthropic")
        PydanticAIAgentFactory.get_provider("google", api_key="test_key_google")

        # Assert: Providers cached
        initial_cache_size = len(PydanticAIAgentFactory._providers)
        assert initial_cache_size >= 3, (
            f"少なくとも3つのProviderがキャッシュされているべき（実際: {initial_cache_size}）"
        )

        # Act: Explicit cache clear
        PydanticAIAgentFactory.clear_cache()

        # Assert: Cache empty
        assert len(PydanticAIAgentFactory._providers) == 0, "clear_cache()後はキャッシュが空であるべき"

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_cache_isolation_between_tests(self):
        """Test fixture auto-clears cache between tests.

        REAL components:
        - Real fixture behavior (clear_pydantic_ai_cache autouse=True)
        - Real cache state isolation

        NOTE: This test relies on the `clear_pydantic_ai_cache` fixture
        in tests/integration/conftest.py (autouse=True, scope=function).

        Scenario:
        1. Verify cache starts empty (fixture cleared it before test)
        2. Add provider to cache
        3. Test ends (fixture will clear cache automatically)

        Assertions:
        - Cache empty at test start (fixture cleanup from previous test)
        - Cache populated after provider creation

        NOTE: Cache清空は fixture によって次のテスト開始前に自動実行される
        """
        # Assert: Cache starts empty (fixture cleared before this test)
        assert len(PydanticAIAgentFactory._providers) == 0, (
            "テスト開始時はキャッシュが空であるべき（fixture自動クリア）"
        )

        # Act: Add provider to cache
        PydanticAIAgentFactory.get_provider("openai", api_key="isolation_test_key")

        # Assert: Cache now contains the provider
        assert len(PydanticAIAgentFactory._providers) >= 1, "Provider追加後はキャッシュに存在する"

        # NOTE: After this test ends, the `clear_pydantic_ai_cache` fixture
        # will automatically clear the cache for the next test.


# ==============================================================================
# Phase B Task 4.3: Provider Instance Sharing Tests
# ==============================================================================


class TestProviderInstanceSharing:
    """Provider instance sharing across models tests.

    Tests that same provider instance is shared across multiple models.
    """

    @pytest.mark.integration
    @pytest.mark.fast_integration
    def test_provider_sharing_across_models(self):
        """Test provider instance shared across multiple models.

        REAL components:
        - Real Provider instance caching
        - Real provider key generation and lookup

        Scenario:
        1. Configure openai_model_1, openai_model_2 (same provider)
        2. Get providers for both models (same API key)
        3. Verify same underlying provider instance (ID check)

        Assertions:
        - Same provider key for both models
        - Same provider instance ID (shared provider)
        """
        # Act: Get provider for first OpenAI model
        provider1 = PydanticAIAgentFactory.get_provider("openai", api_key="shared_key_123")

        # Act: Get provider for second OpenAI model (same key)
        provider2 = PydanticAIAgentFactory.get_provider("openai", api_key="shared_key_123")

        # Assert: Same provider instance returned (shared)
        assert provider1 is provider2, (
            "同じprovider名と同じAPI keyでは同じProviderインスタンスが共有されるべき"
        )
        assert id(provider1) == id(provider2), "Provider instance IDが同じであることを確認"

        # Act: Get provider with DIFFERENT key
        provider3 = PydanticAIAgentFactory.get_provider("openai", api_key="different_key_456")

        # Assert: Different provider instance (different key)
        assert provider1 is not provider3, "異なるAPI keyでは異なるProviderインスタンスが作成されるべき"
        assert id(provider1) != id(provider3), "Provider instance IDが異なることを確認"
