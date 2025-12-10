"""BDD tests for SimplifiedAgentWrapper.

This test file loads scenarios from simplified_agent_wrapper.feature
and imports step definitions from step_definitions/simplified_agent_wrapper_steps.py.
"""

import pytest
from pytest_bdd import scenarios

# Load all scenarios from the feature file
scenarios("simplified_agent_wrapper.feature")

# Import step definitions (this makes them available to pytest-bdd)
# The step definitions are already defined in step_definitions/simplified_agent_wrapper_steps.py
# but we need to import them to make them discoverable by pytest
from .step_definitions.simplified_agent_wrapper_steps import *  # noqa: F401, F403
