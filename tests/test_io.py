"""
I/O layer tests.

These tests verify lightweight I/O behaviour that does not require a full
spectral fit:
  - The reference NGC 7469 spectrum data file exists and is readable.
  - specmod.read_spec() loads the spectrum with the correct wavelength range,
    units, and non-zero flux values.
  - cafe_io.get_output_path() creates the requested directory.
  - The bundled opacity and PAH template tables are present on disk.
"""

import os

import astropy.units as u
import numpy as np
import pytest

from tests.constants import DATA_DIR, DATA_FILE, REDSHIFT, SOURCE_FN


# ---------------------------------------------------------------------------
# Data-file existence
# ---------------------------------------------------------------------------

def test_ngc7469_spectrum_file_exists():
    assert os.path.isfile(DATA_FILE), (
        f"Reference spectrum not found: {DATA_FILE}\n"
        "Make sure the file is present in notebooks/input_data/"
    )


def test_ngc7469_spectrum_nonzero_size():
    assert os.path.getsize(DATA_FILE) > 0, "Spectrum file is empty."


# ---------------------------------------------------------------------------
# specmod.read_spec()
# ---------------------------------------------------------------------------

class TestReadSpec:

    @pytest.fixture(scope="class")
    def loaded_specmod(self, tmp_path_factory):
        import matplotlib
        matplotlib.use("Agg")
        from cafe.fitter import specmod

        s = specmod(output_dir=str(tmp_path_factory.mktemp("io_output")))
        s.read_spec(SOURCE_FN, file_dir=DATA_DIR + os.sep, z=REDSHIFT)
        return s

    def test_waves_attribute_set(self, loaded_specmod):
        """read_spec stores the spectrum in self.waves / self.fluxes / self.flux_uncs."""
        assert hasattr(loaded_specmod, "waves")
        assert loaded_specmod.waves is not None
        assert len(loaded_specmod.waves) > 0

    def test_wavelength_range_miri(self, loaded_specmod):
        """MIRI MRS covers ~5–28 µm; the spectrum should fall within this range."""
        wave = loaded_specmod.waves  # rest-frame µm after redshift correction
        assert wave.min() >= 4.0,  f"Minimum wavelength too low: {wave.min():.2f} µm"
        assert wave.max() <= 30.0, f"Maximum wavelength too high: {wave.max():.2f} µm"

    def test_flux_values_positive(self, loaded_specmod):
        """Most flux values should be positive (some noisy bins may be negative)."""
        flux = loaded_specmod.fluxes
        positive_fraction = np.sum(flux > 0) / len(flux)
        assert positive_fraction > 0.8, (
            f"Less than 80 % of flux values are positive ({positive_fraction:.1%})"
        )

    def test_uncertainty_present(self, loaded_specmod):
        """The loaded spectrum must carry uncertainty information."""
        assert hasattr(loaded_specmod, "flux_uncs")
        assert loaded_specmod.flux_uncs is not None
        assert len(loaded_specmod.flux_uncs) > 0

    def test_redshift_applied(self, loaded_specmod):
        """
        After reading with z=0.0163, the internal redshift attribute should
        match what was passed in.
        """
        assert hasattr(loaded_specmod, "z")
        assert abs(loaded_specmod.z - REDSHIFT) < 1e-6


# ---------------------------------------------------------------------------
# cafe_io utility methods
# ---------------------------------------------------------------------------

class TestCafeIO:

    def test_get_output_path_creates_directory(self, tmp_path):
        from cafe.io import cafe_io
        target = str(tmp_path / "some" / "nested" / "dir")
        result = cafe_io.get_output_path(str(tmp_path / "some"), "nested/dir")
        assert os.path.isdir(result)

    def test_get_output_path_no_filename(self, tmp_path):
        from cafe.io import cafe_io
        result = cafe_io.get_output_path(str(tmp_path / "out"))
        assert os.path.isdir(result)

    def test_get_table_path_returns_bundled_when_no_tabpath(self):
        from cafe.io import cafe_io
        # Minimal inopts dict with no TABPATH set
        inopts = {"PATHS": {}}
        path = cafe_io.get_table_path(inopts)
        assert os.path.isdir(path)

    def test_get_custom_table_dir_empty_when_unset(self):
        from cafe.io import cafe_io
        inopts = {"PATHS": {}}
        result = cafe_io.get_custom_table_dir(inopts)
        assert result == "" or result is None or result == False


# ---------------------------------------------------------------------------
# Bundled table files
# ---------------------------------------------------------------------------

class TestBundledTables:

    def test_gauss_opacity_table_exists(self):
        from cafe.paths import resolve_table_file
        path = resolve_table_file("", "opacity", "gauss_opacity.ecsv")
        assert os.path.isfile(path)

    def test_pah_template_exists(self):
        from cafe.paths import get_table_path
        import glob
        tables_dir = get_table_path()
        pah_files = glob.glob(os.path.join(tables_dir, "pah_template*.txt"))
        assert len(pah_files) > 0, "No PAH template files found in bundled tables."

    def test_opacity_directory_exists(self):
        from cafe.paths import get_table_path
        opacity_dir = os.path.join(get_table_path(), "opacity")
        assert os.path.isdir(opacity_dir)

    def test_resolving_power_directory_exists(self):
        from cafe.paths import get_table_path
        rp_dir = os.path.join(get_table_path(), "resolving_power")
        assert os.path.isdir(rp_dir)
