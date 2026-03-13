"""
Utility functions for locating CRETA package data files at runtime.

Uses importlib.resources so that paths resolve correctly whether the package
is installed via pip, conda, or in editable mode — regardless of the user's
current working directory.
"""
from importlib.resources import files


def get_psf_path() -> str:
    """Return the absolute path to CRETA's built-in PSFs directory."""
    return str(files('creta').joinpath('PSFs'))


def get_package_data_path(*parts) -> str:
    """Return the absolute path to any file inside the CRETA package data.

    Examples
    --------
    >>> get_package_data_path('param_files', 'NGC7469_single_params.txt')
    '/path/to/creta/param_files/NGC7469_single_params.txt'
    """
    return str(files('creta').joinpath(*parts))
