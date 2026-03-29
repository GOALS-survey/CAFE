"""
Unit tests for pure-function modules: cafe.mathfunc and cafe.dustgrainfunc.

These tests are entirely self-contained — they require no data files, no
network access, and do not run a spectral fit.  They should complete in
milliseconds and serve as a fast sanity-check for the core numerical
routines that underpin all fitting.
"""

import numpy as np
import pytest

from cafe.mathfunc import intTab, spline


# ===========================================================================
# spline()  —  cubic spline interpolation wrapper
# ===========================================================================

class TestSpline:

    def test_constant_function(self):
        """Interpolating a constant should return that constant everywhere."""
        x = np.linspace(0, 10, 50)
        y = np.full_like(x, 7.0)
        xnew = np.linspace(1, 9, 20)
        ynew = spline(xnew, x, y)
        np.testing.assert_allclose(ynew, 7.0, rtol=1e-10)

    def test_linear_function(self):
        """Cubic spline should reproduce a linear function exactly."""
        x = np.linspace(0, 10, 100)
        y = 3.5 * x - 1.2
        xnew = np.array([1.5, 4.0, 6.75, 9.0])
        ynew = spline(xnew, x, y)
        np.testing.assert_allclose(ynew, 3.5 * xnew - 1.2, rtol=1e-5)

    def test_quadratic_function(self):
        """Cubic spline should reproduce a quadratic function to high accuracy."""
        x = np.linspace(0, 5, 200)
        y = x ** 2
        xnew = np.array([1.0, 2.5, 4.0])
        ynew = spline(xnew, x, y)
        np.testing.assert_allclose(ynew, xnew ** 2, rtol=1e-5)

    def test_output_shape_matches_input(self):
        """Output array length must match xnew length."""
        x = np.linspace(0, 1, 50)
        y = np.sin(x)
        xnew = np.linspace(0.1, 0.9, 17)
        ynew = spline(xnew, x, y)
        assert ynew.shape == xnew.shape


# ===========================================================================
# intTab()  —  5-point Newton-Cotes (Boole's rule) integrator
# ===========================================================================

class TestIntTab:
    """
    intTab(f, h) integrates the array f over a uniform grid with step h.

    Boole's rule is exact for polynomials up to degree 5, so we use
    constant, linear, and quadratic test functions where the analytic
    integral is known.
    """

    def test_constant(self):
        """∫₀⁸ 3 dx = 24."""
        f = np.ones(9) * 3.0   # 9 points → 8 intervals (multiple of 4)
        h = 1.0
        result = intTab(f, h)
        assert abs(result - 24.0) < 1e-10

    def test_linear(self):
        """∫₀⁸ x dx = 32."""
        x = np.linspace(0, 8, 9)
        h = x[1] - x[0]
        result = intTab(x, h)
        assert abs(result - 32.0) < 1e-6

    def test_quadratic(self):
        """∫₀⁴ x² dx = 64/3 ≈ 21.333…"""
        x = np.linspace(0, 4, 9)   # 8 intervals
        h = x[1] - x[0]
        result = intTab(x ** 2, h)
        np.testing.assert_allclose(result, 64.0 / 3.0, rtol=1e-6)

    def test_positive_result_for_positive_function(self):
        """Integral of a strictly positive function must be positive."""
        x = np.linspace(1, 5, 9)
        h = x[1] - x[0]
        result = intTab(np.exp(x), h)
        assert result > 0

    def test_scale_linearity(self):
        """intTab(α·f, h) == α · intTab(f, h) for scalar α."""
        x = np.linspace(0, 4, 9)
        h = x[1] - x[0]
        f = np.sin(x) + 2.0
        alpha = 3.7
        result_scaled   = intTab(alpha * f, h)
        result_original = intTab(f, h)
        np.testing.assert_allclose(result_scaled, alpha * result_original, rtol=1e-10)


# ===========================================================================
# cafe.paths  —  package-data path resolution
# ===========================================================================

class TestPaths:

    def test_get_table_path_is_directory(self):
        from cafe.paths import get_table_path
        import os
        path = get_table_path()
        assert os.path.isdir(path), f"Table path is not a directory: {path}"

    def test_get_package_data_path_inp_parfiles(self):
        from cafe.paths import get_package_data_path
        import os
        path = get_package_data_path("inp_parfiles")
        assert os.path.isdir(path)

    def test_resolve_table_file_fallback_to_bundled(self):
        """resolve_table_file with an empty custom_dir should return a bundled file."""
        from cafe.paths import resolve_table_file
        import os
        # Use an opacity file that is definitely bundled
        resolved = resolve_table_file("", "opacity", "gauss_opacity.ecsv")
        assert os.path.isfile(resolved), f"Bundled table file not found: {resolved}"

    def test_resolve_table_file_custom_override(self, tmp_path):
        """resolve_table_file should prefer a file in custom_dir when it exists."""
        from cafe.paths import resolve_table_file
        # Create a fake file in the temp dir
        fake_file = tmp_path / "gauss_opacity.ecsv"
        fake_file.write_text("fake")
        resolved = resolve_table_file(str(tmp_path), "opacity", "gauss_opacity.ecsv")
        # Should NOT fall back to bundled because the custom file exists one level up
        # (resolve_table_file joins custom_dir + *parts, so we need:)
        fake2 = tmp_path / "opacity" / "gauss_opacity.ecsv"
        fake2.parent.mkdir()
        fake2.write_text("fake")
        resolved2 = resolve_table_file(str(tmp_path), "opacity", "gauss_opacity.ecsv")
        assert str(resolved2) == str(fake2)
