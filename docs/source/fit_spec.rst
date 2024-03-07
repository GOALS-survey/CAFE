################
Spectral Ftting
################

CAFE Fitting of an individual, 1D spectrum (``CAFE`` v1.0.0)
------------------------------------------------------------

``CAFE`` performs spectral decomposition using the following components:

* Reprocessed continua: Fully characterized dust continuum emission components (including accounting for grain size distribution and composition) defined by their BB emissivity at the equilibrium temperature, which itself depends on the dust grain size and composition, as well as on the heating source (direct light, see below). The continuum components are labeled as: *CIR* (cirrus), *CLD* (cold), *COO* (cool), *WRM* (warm), and *HOT* (hot).

* Direct light continua: *STR* (stellar component mimicking the average interstellar radiation field, ISRF), *STB* (combination of 2, 10 and 100Myr starburst templates), and *DSK* (multiple power law SED characteristic of an accretion disk).

* Emission lines: Hydrogen recombination lines, atomic lines, and ro-vibrational and pure-rotational molecular hydrogen (H2) lines; all described with Gaussian profiles. The lines are read from the following files (contained in the *CAFE/tables/ folder): *lines.H.recombination_?.txt*, *lines.atomic_?.txt* and *lines.molecular_?.txt*. The columns of each of these tables contain: (1) the name of the line, (2) the wavelength (in micron), (3) whether to mask it (1 = do NOT fit) or not (0 = fit); and (4) whether to add a broad component to it (double = 1) or not (double = 0).

* PAHs: All described with Drude profiles. The features are read from the file *pah_template_?.txt* contained in the *CAFE/tables/* folder. The columns of this table contain: (1) the wavelength of the feature; (2) the width, expressed as gamma (= 1/R; FWHM = gamma * wave0); (3) the initial relative peak; (4) the PAH complex to which they belong (useful to group PAHs an get an output flux for the sum of all the sub-components).

* Absorption features: Broad continuum absorption or extinction profiles from amorphous graphitic and silicate grains (by default the OHMC attenuation curve). Additional absorption features can be modeled (a) as templates: water ices at 3.0 and 6.1μm (*ICE3*, *ICE6*), CO2 at 4.27μm (*CO2*), aliphatic hydrocarbons at 3.4 and 6.85μm (*HAC*), CO ro-vib absorption at 4.67μm (*CORV*), and crystalline silicates at 23.3μm (*CRYSI*) -- these features can be controlled from the *.ini* parameter file (see below); or (b) as user-defined absorption bands described by Gaussian distributions -- these features are read from the file *gauss_opacity_?.ecsv* contained in the *CAFE/tables/opacity/* folder. The columns of this table contain: (1) the name of the feature, (2) the central wavelength, (3) the width, expressed as gamma (= 1/R; FWHM = gamma * wave0), (4) the peak (= tau at the central wavelength), (5) whether to mask it (1 = do NOT fit) or not (0 = fit).

The reprocessed and direct light continuum components, as well as the absorption features that are modeled as templates, are initialized using the *.ini* file (in particular, see below [CONTINUA INITIAL VALUES AND OPTIONS]). This file contains (1) the initial values of the parameters definig each component, (2) whether to fit them (True = yes; False = no), (3) the allowed lower and the upper limits, and (4) whether to impose any tie to other parameter. These options are equivalent to the options available within a typical ``LMFIT`` parameter object.


CAFE Setup Files
----------------

*‘inpars_?.ini’* within the *CAFE/init_parfiles/* folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file can be generic or modified accordingly to the initialization needs of the object to be fitted. Within the *.ini* file the user can specify the following:

**[METADATA]**: Not necessary for the current ``CAFE`` release (v1.0.0).

**[COMPONENT SOURCE SEDs]**: SEDs to be used as sources for the different dust components (*ISRF*, *AGN*, *SB2MYR*, *SB10MYR* or *SB100MYR*).

**[MODULES & TABLES]**: (resolutions) Instrument modules (NIRSpec/IFU gratings or MIRI/MRS sub-bands) used to extract the spectra. If a module is missing, features within the wavelength range of the missing module will not be fitted, even if they exist in the spectrum). (???_input) Tables containing the names and wavelengths (together with the widths and peaks in some cases) of the hydrogen-recombination, atomic and molecular lines, PAH features, and gaussian opacities to be fitted. These tables (located in the *tables/* directory) also contain a column (*MASK*) that allows the user to switch on (0) or off (1) specific features if the user think they are not present in the spectra, depending on the nature of the target (PDR, normal star-forming galaxy, starburst, AGN, etc.). In addition, the H-recomb., atomic and molecular tables contain an additional column that allows the user to add a broad component to each line, also characterized with a Gaussian profile.

**[PAH & LINE OPTIONS]**: *Fit???* keywords specify whether the wavelengths, widths (gammas) and peaks of the lines (*LIN*), PAHs (*_PAH*) or Gaussian opacities (*_OPC*) are allowed to vary or not. If they are allowed (= True), the *EPS???* keywords specify by how much (in relative or absolute terms, depending on the feature and parameter). The suffixes (*_N* and *_B*) for the emission line parameters refer to their narrow and broad components, respectively (see above). The EPSVgrad parameter allows for a simulataneous gradient drift of all the line, PAH and Gaussian opacity features, resembling a velocity gradient. This is useful when the redshift of the source is not precisely known.

**[CONTINUA INITIAL VALUES AND OPTIONS]**: Dust continuum components are defined by the following parameters: (relative) flux (*_FLX*), temperature (*_TMP*), depth (*_TAU*, referenced to 9.7μm), fraction of screen/mix obscuration geometry (*_MIX*), covering factor (*_COV*). For each parameter, the value, whether the parameter is fitted or not, its minimum and maximum limits, and a tie constraint (to other parameters) can be specified, in that order, via comma separated values. The fluxes are defined in terms of the relative contribution of that component to the observed spectrum at the reference wavelength of the component (defined in the *_opt.cafe* file; see below).


*'_opt.cafe'* within the *opt_parfiles/* folder:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Disclaimer: We highly discourage the modification of this file, as not all the switches and keywords have been fully tested.*

Within this file the user can specify the following:

**[PATHS]**: Not necessary for the current ``CAFE`` release (v1.0.0). Data paths are directly defined during execution of the command. Other paths are defined automatically.

**[FIT OPTIONS]**: Tolerance of the fit, on-the-fly dust temperature interpolation, whether to fit analytic features: lines, PAHs and user-defined opacities, perform checks on the fitted parameters and allow re-fitting up to a maximum number of iterations, and maximum relative errors allowed to keep features and not to fix them.

**[SWITCHES]**: Impose Onion geometry where the optical depth of higher temperature dust components is progressively higher than lower temperature ones (not supported by the current ``CAFE`` v1.0.0 release). Add a minimum relative error to the provided error spectrum.

**[OUTPUT FILE OPTIONS]**: Print output tables.

**[PLOT_OPTIONS]**: Make alternative plots.

**[MODEL OPTIONS]**: Keywords related to including supplementary photometric data for fitting, in addition to the spectrum (not supported by the current ``CAFE`` v1.0.0 release). Use extinction or absorption curves, and selection of dust model.

**[REFERENCE WAVELENGTHS]**: Reference wavelengths for the scaling of model component fluxes (*_FLX* keywords in *‘.ini’* file).
