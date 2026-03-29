"""
Shared path constants for the CAFE test suite.

Imported by both conftest.py (fixtures) and individual test modules that
need data-file paths without going through a pytest fixture.
"""

import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(TESTS_DIR)

# Reference NGC 7469 spectrum
DATA_FILE  = os.path.join(
    REPO_ROOT, "notebooks", "input_data", "NGC7469_SingleExt_r1.5as_MIRI.dat"
)
DATA_DIR   = os.path.dirname(DATA_FILE)
SOURCE_FN  = os.path.basename(DATA_FILE)

# Fit parameters
REDSHIFT = 0.0163

# Reference JSON
REFERENCE_JSON = os.path.join(TESTS_DIR, "reference", "NGC7469_reference.json")
