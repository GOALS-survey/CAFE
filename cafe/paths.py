"""
Utility functions for locating CAFE package data files at runtime.

Uses importlib.resources so that paths resolve correctly whether the package
is installed via pip, conda, or in editable mode — regardless of the user's
current working directory.
"""
import os
from importlib.resources import files


def get_table_path() -> str:
    """Return the absolute path to CAFE's built-in bundled tables directory."""
    return str(files('cafe').joinpath('tables'))


def resolve_table_file(custom_dir: str, *parts) -> str:
    """Resolve a table file path, falling back to the bundled tables if not
    found in the user's custom directory.

    This enables per-file overrides: place only the files you want to
    customise in your custom directory (set via TABPATH in the .ini/.yaml
    file), and every other file will be resolved from the bundled package
    tables automatically.

    Parameters
    ----------
    custom_dir : str
        Path to the user's custom table directory (from TABPATH in the
        parameter file).  Pass an empty string or None to skip the custom
        lookup and go straight to the bundled tables.
    *parts : str
        Path components relative to the tables directory, e.g.
        ``'pah_template_NGC628.txt'`` or ``'opacity', 'gauss_opacity.ecsv'``.

    Examples
    --------
    >>> resolve_table_file('/my/tables', 'pah_template_NGC628.txt')
    '/my/tables/pah_template_NGC628.txt'   # if the file exists there

    >>> resolve_table_file('/my/tables', 'pah_ratios.txt')
    '/path/to/site-packages/cafe/tables/pah_ratios.txt'  # fallback
    """
    if custom_dir:
        custom_path = os.path.join(custom_dir, *parts)
        if os.path.exists(custom_path):
            return custom_path
    return os.path.join(get_table_path(), *parts)


def get_package_data_path(*parts) -> str:
    """Return the absolute path to any file inside the CAFE package data.

    Examples
    --------
    >>> get_package_data_path('inp_parfiles', 'inpars_jwst_miri.ini')
    '/path/to/cafe/inp_parfiles/inpars_jwst_miri.ini'
    """
    return str(files('cafe').joinpath(*parts))
