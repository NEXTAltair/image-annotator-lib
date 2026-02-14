# tests/features/conftest.py
import asyncio
import copy

import pytest

from image_annotator_lib.core.config import config_registry


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the entire test session.
    This is a workaround for pytest-bdd and asyncio interaction issues.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def context():
    """BDDテスト用のコンテキスト辞書"""
    return {}


@pytest.fixture(scope="function")
def managed_config_registry():
    """A fixture to manage the global config_registry for BDD tests.

    Saves original state, allows modification during test, and restores afterward.
    """
    original_system = copy.deepcopy(config_registry._system_config_data)
    original_user = copy.deepcopy(config_registry._user_config_data)
    original_merged = copy.deepcopy(config_registry._merged_config_data)

    # Provide a clean registry for the test
    config_registry._system_config_data.clear()
    config_registry._user_config_data.clear()
    config_registry._merged_config_data.clear()

    def _set_config(model_name: str, config: dict):
        """Helper to set config for a test."""
        config_registry._merged_config_data[model_name] = config
        config_registry._user_config_data[model_name] = config

    # Temporarily replace the set method for test isolation
    original_set = config_registry.set
    config_registry.set = _set_config

    yield config_registry

    # Restore original state
    config_registry._system_config_data = original_system
    config_registry._user_config_data = original_user
    config_registry._merged_config_data = original_merged
    config_registry.set = original_set


@pytest.fixture(autouse=True)
def caplog_for_loguru(caplog):
    """Redirect loguru to caplog for test assertions."""
    import logging

    from loguru import logger

    # Create propagate handler for caplog
    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield caplog
    logger.remove(handler_id)
