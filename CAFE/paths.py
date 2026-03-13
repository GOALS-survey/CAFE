"""
Utility functions for locating CAFE package data files at runtime.

Uses importlib.resources so that paths resolve correctly whether the package
is installed via pip, conda, or in editable mode — regardless of the user's
current working directory.
"""
import os
from importlib.resources import files


def get_table_path() -> str:
    """Return the absolute path to CAFE's built-in tables directory."""
    return str(files('cafe').joinpath('tables'))


def get_package_data_path(*parts) -> str:
    """Return the absolute path to any file inside the CAFE package data.

    Examples
    --------
    >>> get_package_data_path('inp_parfiles', 'inpars_jwst_miri.ini')
    '/path/to/cafe/inp_parfiles/inpars_jwst_miri.ini'
    """
    return str(files('cafe').joinpath(*parts))
