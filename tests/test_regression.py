"""
Regression tests — verify that fitted fluxes are within 1 % of the
reference values stored in tests/reference/NGC7469_reference.json.

Reference values were produced by running the CAFE_tutorial_NGC7469_1D
notebook with:
  - Input spectrum : notebooks/input_data/NGC7469_SingleExt_r1.5as_MIRI.dat
  - Input params   : cafe/inp_parfiles/inpars_jwst_miri_AGN.yaml
  - Opt params     : cafe/opt_parfiles/default_opt.yaml
  - Redshift       : z = 0.0163

The tolerance is 1 % (rtol = 0.01) on the intrinsic (extinction-corrected)
flux.  Lines / PAH complexes with a reference flux of exactly zero are
excluded from comparison (they are constrained to zero by the model).

Note on table structure
-----------------------
`specmod.pahtable` is a pandas DataFrame indexed by `pah_complex`.
`specmod.linetable` is a pandas DataFrame indexed by `line_name`.
Lookups use `.loc[name, column]`.

To update the reference after an intentional algorithm change, re-run the
notebook and replace the two ecsv files, then update
tests/reference/NGC7469_reference.json with the new values.
"""

import numpy as np
import pytest

# Relative tolerance for all flux comparisons (1 %)
RTOL = 0.01

# Absolute floor: lines weaker than this (W/m²) are skipped to avoid
# failures driven by numerical noise on near-zero values.
FLUX_FLOOR = 1e-19


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pah_flux(fit_result, pah_name):
    """Return the intrinsic PAH flux for `pah_name`, or None if not found."""
    tbl = fit_result.pahtable
    if pah_name not in tbl.index:
        return None
    val = tbl.loc[pah_name, "pah_strength_int"]
    if val == "" or val is None:
        return None
    f = float(val)
    return None if np.isnan(f) else f


def _get_line_flux(fit_result, line_name):
    """Return the intrinsic line flux for `line_name`, or None if not found."""
    tbl = fit_result.linetable
    if line_name not in tbl.index:
        return None
    val = tbl.loc[line_name, "line_strength_int"]
    if val == "" or val is None:
        return None
    f = float(val)
    return None if np.isnan(f) else f


# ---------------------------------------------------------------------------
# PAH complex regression
# ---------------------------------------------------------------------------

class TestPAHRegression:

    @pytest.mark.parametrize("pah_name,ref_flux", [
        ("PAH62",    3.882747444153767e-15),
        ("PAH77_C",  1.2692313461394962e-14),
        ("PAH83",    1.0392641996566975e-15),
        ("PAH86",    2.645102393825928e-15),
        ("PAH113_C", 2.9532050718945684e-15),
        ("PAH120",   5.336338104388876e-16),
        ("PAH126_C", 2.3900526396342105e-15),
        ("PAH142",   7.034198603394191e-17),
        ("PAH164",   4.708730238136242e-16),
        ("PAH170_C", 2.1689218038740993e-15),
    ])
    def test_pah_intrinsic_flux(self, fit_result, pah_name, ref_flux):
        """Intrinsic PAH flux must be within 1 % of the reference value."""
        if ref_flux == 0.0 or ref_flux < FLUX_FLOOR:
            pytest.skip(f"Reference flux for {pah_name} is near zero — skipping")

        measured = _get_pah_flux(fit_result, pah_name)
        assert measured is not None, f"PAH complex '{pah_name}' not found in pahtable"

        np.testing.assert_allclose(
            measured, ref_flux,
            rtol=RTOL,
            err_msg=(
                f"{pah_name}: measured={measured:.4e} W/m², "
                f"reference={ref_flux:.4e} W/m², "
                f"deviation={abs(measured/ref_flux - 1)*100:.2f} %"
            ),
        )


# ---------------------------------------------------------------------------
# Emission line regression
# ---------------------------------------------------------------------------

class TestLineRegression:

    @pytest.mark.parametrize("line_name,ref_flux", [
        # Bright ionic / atomic lines
        ("NeII_128136N",   4.879468716121973e-16),
        ("SIII_187130N",   2.9795917120809755e-16),
        ("ArII_69853B",    3.6646935788999985e-16),
        ("NeIII_155551B",  2.433098794645204e-16),
        ("NeV_243175B",    1.7625939754358502e-16),
        ("OIV_258903B",    1.7875927331800454e-16),
        ("NeV_143217B",    1.093266231057733e-16),
        ("NeVI_76524B",    1.1957717729009573e-16),
        ("SIV_105105B",    8.987186124364113e-17),
        ("ArIII_89914B",   4.649296217073025e-17),
        ("NeIII_155551N",  5.552131595664301e-17),
        ("MgVII_55032B",   4.713968598113519e-17),
        ("FeIII_229250N",  2.4414389308924552e-17),
        ("NaVIII_51789B",  2.5014646080521925e-17),
        ("FeII_53402N",    2.0651514724247497e-17),
        ("FeVIII_54466B",  2.3669541513186694e-17),
        ("NeV_143217N",    1.5754058347511783e-17),
        ("H200S2_122790N", 1.7086947404669425e-17),
        ("H200S3_96649N",  1.7352390865966902e-17),
        ("H200S5_69091N",  2.3432921017732997e-17),
        ("H200S4_80258N",  1.1465794763230279e-17),
        ("H200S1_170350N", 2.0907161745702514e-17),
        ("SiVIII_52910B",  1.2488400999834211e-17),
        ("Pfund65_74578B", 1.6656011481073758e-17),
        ("MgV_56098B",     3.8806301492137523e-17),
        ("NeV_243175N",    1.760234249581668e-17),
        ("OIV_258903N",    3.2299451726320885e-17),
        ("FeII_259883N",   1.6880556751057488e-17),
        ("ClII_143678N",   9.736501417846989e-18),
        ("PIII_178850N",   6.038484227391127e-18),
        ("FeII_179359N",   6.203494458755844e-18),
        ("SIV_105105N",    6.002098241600515e-18),
        ("H200S7_55115N",  7.478642460819175e-18),
    ])
    def test_line_intrinsic_flux(self, fit_result, line_name, ref_flux):
        """Intrinsic emission-line flux must be within 1 % of the reference value."""
        if ref_flux == 0.0 or ref_flux < FLUX_FLOOR:
            pytest.skip(f"Reference flux for {line_name} is near zero — skipping")

        measured = _get_line_flux(fit_result, line_name)
        assert measured is not None, f"Line '{line_name}' not found in linetable"

        np.testing.assert_allclose(
            measured, ref_flux,
            rtol=RTOL,
            err_msg=(
                f"{line_name}: measured={measured:.4e} W/m², "
                f"reference={ref_flux:.4e} W/m², "
                f"deviation={abs(measured/ref_flux - 1)*100:.2f} %"
            ),
        )
