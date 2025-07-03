# tests/features/conftest.py
import asyncio

import pytest


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
