"""
Smoke tests — verify that all CAFE (and CRETA) modules can be imported
without errors.  These are the cheapest tests in the suite and should
catch broken installs, missing bundled data files, and bad __init__.py
changes immediately.
"""


def test_import_cafe():
    import cafe  # noqa: F401


def test_import_cafe_fitter_specmod():
    from cafe.fitter import specmod  # noqa: F401


def test_import_cafe_fitter_cubemod():
    from cafe.fitter import cubemod  # noqa: F401


def test_import_cafe_io():
    from cafe.io import cafe_io  # noqa: F401


def test_import_cafe_params():
    from cafe.params import CAFE_param_generator  # noqa: F401


def test_import_cafe_lib():
    import cafe.lib  # noqa: F401


def test_import_cafe_mathfunc():
    from cafe import mathfunc  # noqa: F401


def test_import_cafe_dustgrainfunc():
    from cafe import dustgrainfunc  # noqa: F401


def test_import_cafe_component_model():
    from cafe import component_model  # noqa: F401


def test_import_cafe_paths():
    from cafe.paths import get_package_data_path, get_table_path  # noqa: F401


def test_import_creta():
    from creta.extractor import creta  # noqa: F401
