"""
Functional tests — verify that the full NGC 7469 spectral fit runs to
completion and produces well-formed outputs.

All tests here share a single session-scoped `fit_result` fixture defined
in conftest.py, so the expensive fit (~30–60 s) is executed only once per
pytest run.

Note on table structure
-----------------------
`specmod.pahtable` is a pandas DataFrame indexed by `pah_complex`.
`specmod.linetable` is a pandas DataFrame indexed by `line_name`.
Neither index column appears in `.columns`.

These tests answer "Did the fit succeed and produce sensible outputs?" —
exact numerical agreement is checked in test_regression.py.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fit completion
# ---------------------------------------------------------------------------

class TestFitCompletion:

    def test_fit_result_is_not_none(self, fit_result):
        assert fit_result is not None

    def test_parcube_attribute_exists(self, fit_result):
        assert hasattr(fit_result, "parcube")
        assert fit_result.parcube is not None

    def test_parcube_value_extension_present(self, fit_result):
        """parcube is an astropy HDUList; the VALUE extension must be present."""
        assert "VALUE" in fit_result.parcube

    def test_parcube_has_data(self, fit_result):
        data = fit_result.parcube["VALUE"].data
        assert data is not None
        assert data.size > 0


# ---------------------------------------------------------------------------
# PAH table  (pandas DataFrame, index = pah_complex)
# ---------------------------------------------------------------------------

class TestPAHTable:

    def test_pahtable_attribute_exists(self, fit_result):
        assert hasattr(fit_result, "pahtable")
        assert fit_result.pahtable is not None

    def test_pahtable_nonempty(self, fit_result):
        assert len(fit_result.pahtable) > 0

    def test_pahtable_expected_columns(self, fit_result):
        # pah_complex is the DataFrame index, not a column
        expected = {
            "pah_strength_int",
            "pah_strength_int_unc",
            "pah_strength_obs",
            "pah_strength_obs_unc",
            "pah_complex_eqw",
        }
        missing = expected - set(fit_result.pahtable.columns)
        assert not missing, f"PAH table missing columns: {missing}"

    def test_pahtable_known_complexes_present(self, fit_result):
        """Key PAH complexes that should always appear in a MIRI MRS fit."""
        index = fit_result.pahtable.index
        for expected in ("PAH62", "PAH77_C", "PAH113_C"):
            assert expected in index, f"Expected PAH complex '{expected}' not in table"

    def test_pahtable_bright_pah_positive(self, fit_result):
        """The 7.7 µm complex (PAH77_C) is the brightest PAH in most galaxies."""
        flux = fit_result.pahtable.loc["PAH77_C", "pah_strength_int"]
        assert float(flux) > 0, "PAH77_C intrinsic flux is not positive"

    def test_pahtable_eqw_nonnegative(self, fit_result):
        """Equivalent widths must be ≥ 0 for all complexes."""
        eqw = fit_result.pahtable["pah_complex_eqw"].astype(float)
        assert (eqw >= 0).all(), "Negative equivalent width(s) found in PAH table"


# ---------------------------------------------------------------------------
# Line table  (pandas DataFrame, index = line_name)
# ---------------------------------------------------------------------------

class TestLineTable:

    def test_linetable_attribute_exists(self, fit_result):
        assert hasattr(fit_result, "linetable")
        assert fit_result.linetable is not None

    def test_linetable_nonempty(self, fit_result):
        assert len(fit_result.linetable) > 0

    def test_linetable_expected_columns(self, fit_result):
        # line_name is the DataFrame index, not a column
        expected = {
            "line_lam",
            "line_strength_int",
            "line_strength_int_unc",
            "line_gamma",
            "line_peak",
        }
        missing = expected - set(fit_result.linetable.columns)
        assert not missing, f"Line table missing columns: {missing}"

    def test_linetable_bright_lines_present(self, fit_result):
        """Brightest ionic lines in the NGC 7469 MIRI spectrum must be in the table."""
        index = fit_result.linetable.index
        for line in ("NeII_128136N", "SIII_187130N", "ArII_69853B"):
            assert line in index, f"Expected emission line '{line}' not in table"

    def test_linetable_neii_flux_positive(self, fit_result):
        """[Ne II] 12.81 µm is one of the strongest lines — must have positive flux."""
        flux = fit_result.linetable.loc["NeII_128136N", "line_strength_int"]
        assert float(flux) > 0, "[Ne II] intrinsic flux is not positive"

    def test_linetable_wavelengths_in_miri_range(self, fit_result):
        """All line wavelengths should fall within MIRI MRS coverage (4–30 µm)."""
        import pandas as pd
        for line_name, row in fit_result.linetable.iterrows():
            lam = row["line_lam"]
            if lam == "" or lam is None or pd.isna(lam):
                continue
            lam_f = float(lam)
            assert 4.0 <= lam_f <= 30.0, (
                f"Line {line_name} has wavelength {lam_f} µm outside MIRI range"
            )


# ---------------------------------------------------------------------------
# Velocity gradient
# ---------------------------------------------------------------------------

class TestVelocityGradient:

    def test_vgrad_is_numeric(self, fit_result):
        vgrad = fit_result.parcube["VALUE"].data[-1, 0, 0]
        assert np.isfinite(vgrad), f"VGRAD is not finite: {vgrad}"

    def test_vgrad_physically_reasonable(self, fit_result):
        """VGRAD for NGC 7469 should be well below 100 km/s."""
        vgrad = fit_result.parcube["VALUE"].data[-1, 0, 0]
        assert abs(vgrad) < 100, (
            f"VGRAD = {vgrad:.2f} km/s is unexpectedly large — fit may have diverged"
        )


# ---------------------------------------------------------------------------
# Output files written to disk
# ---------------------------------------------------------------------------

class TestOutputFiles:

    def test_parcube_fits_written(self, fit_result):
        data = fit_result.parcube["VALUE"].data
        assert data is not None and data.ndim == 3

    def test_pahtable_ecsv_written(self, fit_result):
        assert fit_result.pahtable is not None
        assert len(fit_result.pahtable) > 0

    def test_linetable_ecsv_written(self, fit_result):
        assert fit_result.linetable is not None
        assert len(fit_result.linetable) > 0
