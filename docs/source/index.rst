####
CAFE
####

Purpose
=======

The new Continuum And Feature Extraction (``CAFE``) is a revamped version of the ``CAFE`` software –originally developed for fitting Spitzer/IRS spectra– that has been updated and optimized to work with the new JWST IFU data. The new ``CAFE`` is composed of two main tools: (1) the ``CAFE`` Region Extraction Tool Automaton (``CRETA``) and (2) the ``CAFE`` spectral fitting tool. ``CRETA`` performs single-position and full-grid extractions from JWST IFU datasets; that is, from pipeline-processed cubes obtained with the NIRSpec IFU and MIRI MRS instruments. The ``CAFE`` fitter uses the spectra extracted by ``CRETA`` (or spectra provided by the user) and performs a spectral decomposition of the continuum emission (stellar and/or dust), as well as of a variety of common spectral features (in emission and absorption) present in the near- and mid-IR spectra of galaxies. The full dust treatment (size and composition) performed by ``CAFE`` (see Marshall et al. 2007) allows the dust continuum model components to fit not only spectra typical of normal star-forming galaxies but also complex spectral profiles seen in more extreme, heavily dust-obscured starburst galaxies, such as luminous infrared galaxies (LIRGs), active galactic nuclei (AGN), or very luminous quasars.

Installation
============

.. toctree::
  :maxdepth: 2

   How to install <install>



.. CAFE documentation master file, created by
   sphinx-quickstart on Thu Jan  5 09:45:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CAFE's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
