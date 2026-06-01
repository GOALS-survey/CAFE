"""
Parameter-file validation tests.

These tests verify that:
  - All bundled YAML input/optimisation parameter files are syntactically
    valid and can be parsed by PyYAML without errors.
  - The specific files used by the NGC 7469 tutorial exist and are
    locatable via get_package_data_path().
  - Loading those files into a specmod object (without fitting) succeeds
    and produces the expected number of free parameters.
"""

import glob
import os

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_files(subdir):
    """Return all *.yaml files inside a package data subdirectory."""
    from cafe.paths import get_package_data_path
    directory = get_package_data_path(subdir)
    return glob.glob(os.path.join(directory, "*.yaml"))


# ---------------------------------------------------------------------------
# YAML syntax checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("yaml_path", _yaml_files("inp_parfiles"))
def test_input_parfile_yaml_valid(yaml_path):
    """Every bundled input parameter YAML file must parse without errors."""
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    assert data is not None, f"{yaml_path} parsed to None (empty file?)"


@pytest.mark.parametrize("yaml_path", _yaml_files("opt_parfiles"))
def test_opt_parfile_yaml_valid(yaml_path):
    """Every bundled optimisation parameter YAML file must parse without errors."""
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    assert data is not None, f"{yaml_path} parsed to None (empty file?)"


# ---------------------------------------------------------------------------
# File existence checks for the tutorial parameter files
# ---------------------------------------------------------------------------

def test_agn_inppar_file_exists():
    from cafe.paths import get_package_data_path
    path = get_package_data_path("inp_parfiles", "inpars_jwst_miri_AGN.yaml")
    assert os.path.isfile(path), f"AGN input parameter file not found: {path}"


def test_default_optpar_file_exists():
    from cafe.paths import get_package_data_path
    path = get_package_data_path("opt_parfiles", "default_opt.yaml")
    assert os.path.isfile(path), f"Default optimisation file not found: {path}"


# ---------------------------------------------------------------------------
# specmod parameter loading (no fitting)
# ---------------------------------------------------------------------------

def test_specmod_input_param_loads(tmp_path):
    """
    Verify that input_param() succeeds for the AGN + default_opt combination
    used in the NGC 7469 tutorial.
    """
    import matplotlib
    matplotlib.use("Agg")

    from cafe.fitter import specmod
    from cafe.paths import get_package_data_path
    from tests.constants import DATA_DIR, SOURCE_FN, REDSHIFT

    inppar_fn = get_package_data_path("inp_parfiles", "inpars_jwst_miri_AGN.yaml")
    optpar_fn  = get_package_data_path("opt_parfiles", "default_opt.yaml")

    s = specmod(output_dir=str(tmp_path))
    s.read_spec(SOURCE_FN, file_dir=DATA_DIR + os.sep, z=REDSHIFT)
    # This should not raise
    s.input_param(inppar_fn, optpar_fn)

    # The parameter generator should have produced a non-empty parameter set
    assert s.inpars  is not None
    assert s.inopts  is not None


def test_specmod_parameter_count(tmp_path):
    """
    After loading parameters for the AGN tutorial the parcube should contain
    more than 100 parameters (the NGC 7469 fit has 295 total, 129 free).

    input_param() builds self.parcube internally; its VALUE extension has shape
    (n_params, 1, 1), so we read the count from there rather than calling
    CAFE_param_generator directly (which requires the local `spec` object that
    input_param does not store on self).
    """
    import matplotlib
    matplotlib.use("Agg")

    from cafe.fitter import specmod
    from cafe.paths import get_package_data_path
    from tests.constants import DATA_DIR, SOURCE_FN, REDSHIFT

    inppar_fn = get_package_data_path("inp_parfiles", "inpars_jwst_miri_AGN.yaml")
    optpar_fn  = get_package_data_path("opt_parfiles", "default_opt.yaml")

    s = specmod(output_dir=str(tmp_path))
    s.read_spec(SOURCE_FN, file_dir=DATA_DIR + os.sep, z=REDSHIFT)
    s.input_param(inppar_fn, optpar_fn)

    n_params = s.parcube["VALUE"].data.shape[0]
    assert n_params > 100, (
        f"Expected >100 parameters in parcube, got {n_params}"
    )
