"""BDD tests for API Model Discovery.

This test file loads scenarios from api_model_discovery.feature
and imports step definitions from step_definitions/api_model_discovery_steps.py.
"""

import pytest
from pytest_bdd import scenarios

# Load all scenarios from the feature file
scenarios("api_model_discovery.feature")

# Import step definitions (this makes them available to pytest-bdd)
from .step_definitions.api_model_discovery_steps import *  # noqa: F401, F403
