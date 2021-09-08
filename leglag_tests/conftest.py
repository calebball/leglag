import functools
import re
from decimal import Decimal
from typing import Callable, List

import pytest

from leglag.one_d_molecule import OneDMolecule

from leglag_tests.utilities import build_molecule


def pytest_addoption(parser):
    parser.addoption(
        "--inf-basis",
        dest="inf_basis",
        action="store",
        default=10,
        help="The maximum basis set size in the infinite domains.",
    )
    parser.addoption(
        "--fin-basis",
        dest="fin_basis",
        action="store",
        default=10,
        help="The maximum basis set size in the finite domains.",
    )
    parser.addoption(
        "--run-published",
        dest="run_published",
        action="store_true",
        default=False,
        help="When passed, run tests that published results are reproduced.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "published: mark test as reproducing a published result")


def pytest_collection_modifyitems(config, items):
    if config.option.run_published:
        return
    skip_published = pytest.mark.skip(reason="Only runs when --run-published is passed")
    for item in items:
        if "published" in item.keywords:
            item.add_marker(skip_published)


@pytest.fixture(scope="session")
def infinite_basis(pytestconfig) -> int:
    """The maximum basis set size in the infinite domains."""
    return pytestconfig.option.inf_basis


@pytest.fixture(scope="session")
def finite_basis(pytestconfig) -> int:
    """The maximum basis set size in the finite domains."""
    return pytestconfig.option.fin_basis
