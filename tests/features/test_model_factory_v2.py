"""BDD tests for ModelLoad Factory Pattern.

This test file loads scenarios from model_factory_v2.feature
and imports step definitions from step_definitions/model_factory_steps.py.
"""

from pytest_bdd import scenarios

# Load all scenarios from the feature file
scenarios("model_factory_v2.feature")

# Import step definitions (this makes them available to pytest-bdd)
from .step_definitions.model_factory_steps import *  # noqa: F403
