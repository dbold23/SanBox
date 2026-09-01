"""Fixtures for the module-P tests (exclusions / pattern / drift).

Kept OUT of ``conftest.py`` on purpose: several prototype-05 modules are
being built in parallel into this one directory, and a shared ``conftest``
is the file most likely to be rewritten under us. Each module-P test file
does ``from module_p_fixtures import *``; pytest resolves fixtures from the
test module's own namespace, so these survive any conftest churn.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SCHEMA_PATH = (
    "/home/user/SanBox/phase1b/p0-sevengill-schema/keypoints_sevengill_v1.yaml"
)

# Small enough to keep every module far under 90 s, large enough that a
# default spot (radius 0.0055 s-units) is a few pixels across.
TEST_RESOLUTION = (192, 384)

__all__ = ["SCHEMA_PATH", "TEST_RESOLUTION", "schema_path", "schema",
           "stations", "regions", "individual"]


@pytest.fixture(scope="session")
def schema_path():
    assert os.path.exists(SCHEMA_PATH), "schema yaml missing: %s" % SCHEMA_PATH
    return SCHEMA_PATH


@pytest.fixture(scope="session")
def schema(schema_path):
    import exclusions

    return exclusions.load_schema(schema_path)


@pytest.fixture(scope="session")
def stations(schema):
    import exclusions

    return exclusions.default_stations(schema)


@pytest.fixture(scope="session")
def regions(schema, stations):
    import exclusions

    return tuple(exclusions.exclusion_regions(schema, stations=stations))


@pytest.fixture(scope="session")
def individual(regions):
    import pattern

    return pattern.Individual.generate(101, regions=regions)
