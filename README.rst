CAFE
====

Purpose
-------

The new Continuum And Feature Extraction (``CAFE``) is a python version of the original ``CAFE`` IDL software –originally developed for fitting Spitzer/IRS spectra– that has been updated and optimized to work with JWST IFU data. ``CAFE`` is composed of two main tools: (1) the ``CAFE`` Region Extraction Tool Automaton (``CRETA``) and (2) the ``CAFE`` spectral fitting tool, or fitter. ``CRETA`` performs single-position and full-grid extractions from JWST IFU datasets; that is, from pipeline-processed cubes obtained with the NIRSpec IFU and MIRI MRS instruments. The ``CAFE`` fitter uses the spectra extracted by ``CRETA`` (or spectra provided by the user) and performs a spectral decomposition of the continuum emission (stellar and/or dust), as well as of a variety of common spectral features (in emission and absorption) present in the near- and mid-IR spectra of galaxies. The full dust treatment (size and composition) performed by ``CAFE`` (see Marshall et al. 2007) allows the dust continuum model components to fit not only spectra of typical star-forming galaxies, but also the spectra seen in more extreme, heavily dust-obscured starburst galaxies, such as luminous infrared galaxies (LIRGs and ULIRGs), active galactic nuclei (AGN), or very luminous quasars.

The current release of ``CAFE`` (v1.0.0) supports the ``CAFE`` spectral fitting tool. The ``CRETA`` extraction tool will be supported and more fully described in subsequent releases of the code.

If you use ``CAFE`` to fit your data, please reference it as ***Diaz-Santos et al. (2025)*** (see bibcode below) and add a link to the GitHub repository: https://github.com/GOALS-survey/CAFE

Documentation
-------------
Hosted by readthedocs: <https://goals-cafe.readthedocs.io/en/latest/>

Contributors
------------
* Tanio Diaz-Santos
* Thomas Lai
* Luke Finnerty
* George Privon
* Paolo Bonfini
* Kirsten Larson
* Jason Marshall
* Lee Armus
* Vassilis Charmandaris

Bibcode
-------
@software{2025ascl.soft01001D,
author = {{Diaz-Santos}, Tanio and {Lai}, Thomas S. -Y. and {Finnerty}, Luke and {Privon}, George and {Bonfini}, Paolo and {Larson}, Kirsten and {Marshall}, Jason and {Armus}, Lee and {Charmandaris}, Vassilis},
title = "{CAFE: Continuum And Feature Extraction tool}",
howpublished = {Astrophysics Source Code Library, record ascl:2501.001},
year = 2025,
month = jan,
eid = {ascl:2501.001},
adsurl = {https://ui.adsabs.harvard.edu/abs/2025ascl.soft01001D},
adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
