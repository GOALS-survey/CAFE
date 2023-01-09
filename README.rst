CAFE
====

``CAFE`` is a revamped version of the CAFE software –originally developed for fitting Spitzer/IRS spectra– that has been updated and optimized to work with the new JWST IFU data. The new CAFE is composed of two main tools: (1) the CAFE Region Extraction Tool Automaton (CRETA) and (2) the CAFE the spectral fitting tool. CRETA performs single-position and full-grid extractions from JWST IFU datasets; that is, from pipeline-processed cubes obtained with the NIRSpec IFU and MIRI MRS instruments. The CAFE fitter uses the spectra extracted by CRETA (or spectra provided by the user) and performs a spectral decomposition of the continuum emission (stellar and/or dust), as well as of a variety of common spectral features (in emission and absorption) present in the near- and mid-IR spectra of galaxies. The full dust treatment (size and composition) performed by CAFE (see Marshall et al. 2007) allows the dust continuum model components to fit not only spectra typical of normal star-forming galaxies but also complex spectral profiles seen in more extreme, heavily dust-obscured starburst galaxies, such as luminous infrared galaxies (LIRGs), active galactic nuclei (AGN), or very luminous quasars.


Purpose

The new Continuum And Feature Extraction (CAFE) is a revamped version of the CAFE software –originally developed for fitting Spitzer/IRS spectra– that has been updated and optimized to work with the new JWST IFU data. The new CAFE is composed of two main tools: (1) the CAFE Region Extraction Tool Automaton (CRETA) and (2) the CAFE the spectral fitting tool. CRETA performs single-position and full-grid extractions from JWST IFU datasets; that is, from pipeline-processed cubes obtained with the NIRSpec IFU and MIRI MRS instruments. The CAFE fitter uses the spectra extracted by CRETA (or spectra provided by the user) and performs a spectral decomposition of the continuum emission (stellar and/or dust), as well as of a variety of common spectral features (in emission and absorption) present in the near- and mid-IR spectra of galaxies. The full dust treatment (size and composition) performed by CAFE (see Marshall et al. 2007) allows the dust continuum model components to fit not only spectra typical of normal star-forming galaxies but also complex spectral profiles seen in more extreme, heavily dust-obscured starburst galaxies, such as luminous infrared galaxies (LIRGs), active galactic nuclei (AGN), or very luminous quasars.

Installation

> pip install git+https://github.com/GOALS-survey/CAFE.git

Current Release

CAFE v1.0 (2023/01/18)

The current release of CAFE supports the extraction and fitting of any single spectrum extracted from any combination (or all) of the MIRI/MRS sub-band cubes (ch1_short, ch1_medium, ch1_long, ch2_short, ch2_medium, ch2_long, ch3_short, ch3_medium, ch3_long, ch4_short, ch4_medium, ch4_long), covering the wavelength range from ~5 to 28μm. NIRSpec/IFU spectral extractions and fitting will be supported soon in subsequent releases. Nevertheless, we note that CAFE already includes some of these capabilities, but they have not been fully tested and therefore are not documented here. The users, however, should feel free to experiment with them if they wish, but no support will be provided.

Usage

The current version of CAFE (v1.0) is released together with a jupyter notebook that walks the user through the process of extraction and fitting of a spectrum obtained from MIRI/MRS cubes produced by the data pipeline. For the user’s convenience, there is also a python (.py) script for command line executions that users may edit as they see fit.

The user can use CAFE from two starting points:

JWST IFU cubes: The user can employ CRETA to extract a continuous or discontinuous spectrum from one, some or all data cubes that the user has copied and made available in the data directory within CRETA. Currently CRETA supports the extraction of individual spectra; that is, extractions along a single line-of-sight, or position in the sky. CRETA will extract the spectrum based on a set of parameters provided by the user in a parameter file. In it, the user can specify:

cubes: The cubes to be extracted (currently, only MIRI/MRS)
user_r_ap: The radius of the circular aperture used for the extraction
user_ra, user_dec: RA and Dec coordinates of the source
point_source: The method of extractions (point source: cone extraction, with radius increasing linearly with wavelength; extended source: cylinder extraction, with constant radius)
lambda_ap: Wavelength reference for the definition of aperture radius (if point source extractions; ignored otherwise)
aperture_correction: Whether to perform aperture correction based on PSF cubes
centering: Whether to perform a centroid centering on the user provided coordinates
lambda_cent: Wavelength at which the centering will be performed (ignored otherwise)
background_sub: Whether to perform an annulus-based background subtraction prior to the aperture photometry
r_ann_in: Inner radius of the background annulus (ignored otherwise)
ann_width: Width of the background annulus (ignored otherwise)

Options for directory setup (specified in the command execution only):

data_path (default: CRETA/data/)
PSFs_path (default: CRETA/PSFs/)
output_path (default: CRETA/extractions/)
output_filebase_name (default: ‘last_result’)
parfile_path (default: CRETA/param_files/)
parfile_name (default: ‘single_params.txt’)

CRETA will return a ‘*_cube.fits’ file containing the extracted spectrum, which can be fed directly to CAFE for fitting.

The specific steps to achieve this can be found in the appropriate jupyter notebook inside the notebooks folder in the Git repository

An individual, 1D spectrum: CAFE is able to read spectra that have been either extracted from CRETA or provided by the user in a simple ‘.txt’ file containing a table with columns reporting wavelength, flux, and error flux.

Once the spectrum is read, it can be plotted together with the initial (default) model decomposition for visual inspection. The user can tweak the initial model by modifying keywords in a number of files (described in the following section). These keywords refer either to model parameters themselves (e.g., peak line fluxes, dust component temperatures, etc.), constraints for each or a combination of model parameters (e.g., line width variation limits, optical depth variation limits for dust components, etc.), specific names of files, or in general info needed for the setup.

Once the user is satisfied with the initial, guess model, the spectral fitting can be run. CAFE uses the LMFIT python package to minimize the data-model residuals using the Trust Region Reflective least-squares method (‘least_squares’), and based on the χ2 statistic.

The CAFE fitter returns a parameter object containing the best/optimized parameters from which physical quantities and observables can be extracted (e.g., temperatures) or constructed (e.g., fluxes, based on the feature peak and width). The parameter information can be dumped into python dictionaries for further use, or stored in data tables. In addition, the parameter object is saved to disk as a .fits file in a ‘parameter cube’, which can be read at a later stage to run further fits or generate new dictionaries or data tables. The parameter cubes are stored in the ‘output/’ folder using a default name that is the same as the input spectrum file.

CAFE Setup Files

CAFE performs spectral decomposition using the following components:

Reprocessed continua: Fully characterized (including grain size and composition) dust continuum emission, defined by their BB emissivity equilibrium temperatures: CLD (cold), COO (cool), WRM (warm), and HOT (hot).

Direct light continua: STR (stellar component mimicking the average interstellar radiation field, ISRF), STB (combination of 2, 10 and 100Myr starburst templates), and DSK (multiple power law SED characteristic of an accretion disk).

PAHs: Described with Drude profiles (set up read from table; see below)

Emission lines: Hydrogen-recombination lines, atomic lines, and vibrational and pure-rotational molecular hydrogen (H2) lines, described with Gaussian profiles (set up read from tables; see below).

Absorption features: Broad continuum absorption and extinction profiles from amorphous graphitic and silicate grains. Additional absorption features are modeled (a) as templates: water ices at 3.0 and 6.1μm (ICE3, ICE6), CO2 at 4.27μm (CO2), aliphatic hydrocarbons at 3.4 and 6.85μm (HAC), CO ro-vib absorption at 4.67μm (CORV), and crystalline silicates at 23.3μm (CRYSI); or (b) as user-defined optical depths described with Gaussian distributions (set up read from table; see below).

The parameters that define these components are initialized via a number of files that the user can modify. These files are:

> ‘inpars_*.ini’ within the ‘init_parfiles’ folder:

Within this file the user specifies the following:

[METADATA]: Not necessary for the current CAFE release (v1.0).

[COMPONENT SOURCE SEDs]: SEDs to be used as sources for the different dust components.

[MODULES & TABLES]: Instrument modules (NIRSpec/IFU gratings or MIRI/MRS sub-bands) used to extract the spectra. If a module is missing, features within the wavelength range of the missing module will not be fitted, even if they exist in the spectrum). Tables containing the names and wavelengths (together with the widths and peaks in some cases) of the H-recomb., atomic and molecular lines, PAH features, and gaussian opacities to be fitted. These tables (located in the ‘tables/’ directory) also contain a column (MASK) that allows the user to switch on (0) or off (1) specific features if the user think they are not present in the spectra, depending on the nature of the target (PDR, normal star-forming galaxy, starburst, AGN). In addition, the H-recomb., atomic and molecular tables contain an additional column that allows the user to add a broad component to each line, also characterized with a Gaussian profile.

[PAH & LINE OPTIONS]: Fit* keywords specify whether the wavelengths and widths of the lines or PAHs are allowed to vary or not. If they are, the EPS* keywords specify by how much (in relative or absolute terms, depending on the feature and parameter).

[CONTINUA INITIAL VALUES AND OPTIONS]: Dust continuum components are defined by the following parameters: (relative) flux (_FLX), temperature (_TMP), depth (_TAU, referenced to 9.7μm), fraction of screen/mix obscuration geometry (_MIX), covering factor (_COV). For each parameter, the value, whether the parameter is fitted or not, its minimum and maximum limits, and a tie constraint (to other parameters) can be specified, in that order, via comma separated values. The fluxes are specified via the relative contribution of that component at a reference wavelength (defined in the ‘*_opt.cafe’ file; see below).

> ‘*_opt.cafe’ within the ‘opt_parfiles/’ folder:

Disclaimer: We highly discourage the modification of this file, as not all the switches and keywords have been fully tested.

[PATHS]: Not necessary for the current CAFE release (v1.0). Data paths are directly defined during execution of the command. Other paths are defined automatically.

[FIT OPTIONS]: Tolerance of the fit, on-the-fly dust temperature interpolation, whether to fit analytic features: lines, PAHs and user-defined opacities, perform checks on the fitted parameters and allow re-fitting up to a maximum number of iterations, and maximum relative errors allowed to keep features and not to fix them.

[SWITCHES]: Impose Onion geometry where the optical depth of higher temperature dust components is progressively higher than lower temperature ones (not supported by the current CAFE v1.0 release). Add a minimum relative error to the provided error spectrum.

[OUTPUT FILE OPTIONS]: Print output tables.

[PLOT_OPTIONS]: Make alternative plots.

[MODEL OPTIONS]: Keywords related to accommodating the fit of supplementary photometric data, in addition to spectra (not supported by the current CAFE v1.0 release). Use extinction or absorption curves and selection of dust model.

[REFERENCE WAVELENGTHS]: Reference wavelengths for the scaling of model component fluxes (_FLX keywords in ‘.ini’ file).

