############
Introduction
############

Usage
-----

The current version of ``CAFE`` (v1.0.0) is released together with a jupyter notebook that walks the user through the process of fitting an spectrum obtained from JWST MIRI/MRS cubes produced by the pipeline. The jupyter notebook, named **CAFE_tutorial_NGC7469_1D.ipynb**, can be found in the ./notebooks/ folder at the GitHub repository, and can be copied to a working directory created by the user.

``CAFE`` is able to read any spectrum provided by the user in a simple *‘.txt’* file containing a table with columns reporting: wavelength, flux, and error flux. (In future versions of ``CAFE``, the ``CRETA`` extraction tool will be supported, and produce outputs/extractions that can be fed directly to the ``CAFE`` fitter as well).

In the notebook, after setting up the necessary paths pointing to where the data are located and where the output results from the fitting will be stored, an spectrum of a region from the galaxy NGC 7469 is downloaded from the cloud. Next, the redshift of the source is set, together with the name of the file containing the spectrum, and the *.ini* and *.cafe* files necessary for the initialization of the fit (located in the *CAFE/inp_parfiles/* and *CAFE/opt_parfiles/* folders). These two last files control, respectively: (1) the initial value of the parameters related to the continuum components, as well as what parameters are allowed to vary and by how much for each feature type (continua, lines, PAHs, user opacities); and (2) whether some features can be fitted, as well as other more code-specific variables (we do not recommend changing any of these, as some have not been tested). We refer the user to the next section for a more detailed description of these files.

After, the spectrum can be read from disk and plotted together with the initial (default) model decomposition for visual inspection. The user can tweak the initial model by modifying keywords in the *.ini* parameter file. These keywords refer either to model parameters themselves (e.g., peak line fluxes, dust component temperatures, etc.), constraints for each or a combination of model parameters (e.g., line width variation limits, optical depth variation limits for dust components, etc.), specific names of files, or in general info needed for the setup.

Once the user is satisfied with the initial guess model, the spectral fitting can be run. ``CAFE`` uses the ``LMFIT`` python package to minimize the data-model residuals using the Trust Region Reflective least-squares method (``least_squares``), and based on the χ2 statistic.



CAFE Setup Files
----------------

``CAFE`` performs spectral decomposition using the following components:

* Reprocessed continua: Fully characterized (including grain size and composition) dust continuum emission, defined by their BB emissivity equilibrium temperatures: *CLD* (cold), *COO* (cool), *WRM* (warm), and *HOT* (hot).

* Direct light continua: *STR* (stellar component mimicking the average interstellar radiation field, ISRF), *STB* (combination of 2, 10 and 100Myr starburst templates), and *DSK* (multiple power law SED characteristic of an accretion disk).

* PAHs: Described with Drude profiles (set up read from table; see below)

* Emission lines: Hydrogen-recombination lines, atomic lines, and vibrational and pure-rotational molecular hydrogen (H2) lines, described with Gaussian profiles (set up read from tables; see below).

* Absorption features: Broad continuum absorption and extinction profiles from amorphous graphitic and silicate grains. Additional absorption features are modeled (a) as templates: water ices at 3.0 and 6.1μm (*ICE3*, *ICE6*), CO2 at 4.27μm (*CO2*), aliphatic hydrocarbons at 3.4 and 6.85μm (*HAC*), CO ro-vib absorption at 4.67μm (*CORV*), and crystalline silicates at 23.3μm (*CRYSI*); or (b) as user-defined optical depths described with Gaussian distributions (set up read from table; see below).


The parameters that define these components are initialized via a number of files that the user can modify. These files are:


*‘inpars_?.ini’* within the *init_parfiles/* folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within this file the user can specify the following:

**[METADATA]**: Not necessary for the current ``CAFE`` release (v1.0).

**[COMPONENT SOURCE SEDs]**: SEDs to be used as sources for the different dust components.

**[MODULES & TABLES]**: Instrument modules (NIRSpec/IFU gratings or MIRI/MRS sub-bands) used to extract the spectra. If a module is missing, features within the wavelength range of the missing module will not be fitted, even if they exist in the spectrum). Tables containing the names and wavelengths (together with the widths and peaks in some cases) of the hydrogen-recombination, atomic and molecular lines, PAH features, and gaussian opacities to be fitted. These tables (located in the *tables/* directory) also contain a column (*MASK*) that allows the user to switch on (0) or off (1) specific features if the user think they are not present in the spectra, depending on the nature of the target (PDR, normal star-forming galaxy, starburst, AGN, etc.). In addition, the H-recomb., atomic and molecular tables contain an additional column that allows the user to add a broad component to each line, also characterized with a Gaussian profile.

**[PAH & LINE OPTIONS]**: *Fit** keywords specify whether the wavelengths and widths of the lines or PAHs are allowed to vary or not. If they are, the *EPS** keywords specify by how much (in relative or absolute terms, depending on the feature and parameter).

**[CONTINUA INITIAL VALUES AND OPTIONS]**: Dust continuum components are defined by the following parameters: (relative) flux (*_FLX*), temperature (*_TMP*), depth (*_TAU*, referenced to 9.7μm), fraction of screen/mix obscuration geometry (*_MIX*), covering factor (*_COV*). For each parameter, the value, whether the parameter is fitted or not, its minimum and maximum limits, and a tie constraint (to other parameters) can be specified, in that order, via comma separated values. The fluxes are specified via the relative contribution of that component at a reference wavelength (defined in the *‘_opt.cafe’* file; see below).


*'_opt.cafe'* within the *opt_parfiles/* folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Disclaimer: We highly discourage the modification of this file, as not all the switches and keywords have been fully tested.*

Within this file the user can specify the following:

**[PATHS]**: Not necessary for the current ``CAFE`` release (v1.0.0). Data paths are directly defined during execution of the command. Other paths are defined automatically.

**[FIT OPTIONS]**: Tolerance of the fit, on-the-fly dust temperature interpolation, whether to fit analytic features: lines, PAHs and user-defined opacities, perform checks on the fitted parameters and allow re-fitting up to a maximum number of iterations, and maximum relative errors allowed to keep features and not to fix them.

**[SWITCHES]**: Impose Onion geometry where the optical depth of higher temperature dust components is progressively higher than lower temperature ones (not supported by the current ``CAFE`` v1.0.0 release). Add a minimum relative error to the provided error spectrum.

**[OUTPUT FILE OPTIONS]**: Print output tables.

**[PLOT_OPTIONS]**: Make alternative plots.

**[MODEL OPTIONS]**: Keywords related to accommodating the fit of supplementary photometric data, in addition to spectra (not supported by the current ``CAFE`` v1.0.0 release). Use extinction or absorption curves and selection of dust model.

**[REFERENCE WAVELENGTHS]**: Reference wavelengths for the scaling of model component fluxes (*_FLX* keywords in *‘.ini’* file).

CAFE Output files
-----------------

The ``CAFE`` fitter returns a parameter object containing the best/optimized parameters from which physical quantities and observables can be extracted (e.g., temperatures) or constructed (e.g., fluxes, based on the feature peak and width). All the parameter information is dumped into python dictionaries for further use and stored in data tables on disk, in the *cafe_output/* directory. In addition, the parameter object is saved to disk as a .fits file in a ‘parameter cube‘, which can be read at a later stage to run further fits or generate new dictionaries or data tables (see jupyter notebook section "RESTORE CAFE SESSION FROM DISK"). All these optputs are stored in the *‘cafe_output/’* folder using a default name that is the same as the input spectrum file. In summary the output files from ``CAFE`` are:

* *_cafefit.asdf* : A file containing dictionaries with all the parameter information
* *_parcube.fits* : A file containing all the info necessary to recover a previous CAFE fitting session
* *_fitpars.ini* : A file containing all the fitted parameters in the format of an *.ini* file, which can be used to initialize the fit of other spectra
* *_fitfigure.png* : A file containing a figure of the last fit performed
* *_linetable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted emission lines
* *_pahtable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted PAH features

All this steps can be found in the jupyter notebook inside the *notebooks/* folder in the GitHub repository.


Caveat
------
