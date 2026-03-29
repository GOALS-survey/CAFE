"""
Shared pytest fixtures for the CAFE test suite.

The most expensive fixture is `fit_result`, which runs a full CAFE spectral
fit on the NGC 7469 reference spectrum.  It is session-scoped so it executes
only once per pytest run and its result is shared by all tests that need it.
"""

import os
import json
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any other matplotlib import

# ---------------------------------------------------------------------------
# Path constants (shared via constants.py so test modules can import them
# directly without relying on conftest being importable as a regular module)
# ---------------------------------------------------------------------------
from tests.constants import (  # noqa: E402
    DATA_DIR, DATA_FILE, REDSHIFT, REFERENCE_JSON, SOURCE_FN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def reference_fluxes():
    """Load the golden-reference flux dictionary from JSON."""
    with open(REFERENCE_JSON) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def fit_result(tmp_path_factory):
    """
    Run a full CAFE fit on NGC 7469 once per test session.

    Returns the fitted `specmod` object.  All tests that need fitted results
    (functional checks, regression comparisons) share this single run.
    """
    from cafe.fitter import specmod
    from cafe.paths import get_package_data_path

    inppar_fn = get_package_data_path("inp_parfiles", "inpars_jwst_miri_AGN.yaml")
    optpar_fn = get_package_data_path("opt_parfiles", "default_opt.yaml")

    output_dir = str(tmp_path_factory.mktemp("cafe_output"))

    s = specmod(output_dir=output_dir)
    s.read_spec(SOURCE_FN, file_dir=DATA_DIR + os.sep, z=REDSHIFT)
    s.input_param(inppar_fn, optpar_fn)
    s.fit_spec(output_path=output_dir)

    return s
