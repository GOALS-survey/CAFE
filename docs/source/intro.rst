############
Introduction
############

Usage
-----

The current version of ``CAFE`` (v1.0.0) is released together with a jupyter notebook that walks the user through the process of fitting a spectrum obtained from JWST MIRI/MRS data. The jupyter notebook, named **CAFE_tutorial_NGC7469_1D.ipynb**, can be found in the *notebooks/* folder in the GitHub repository, and can be copied to a working directory created by the user.

``CAFE`` is able to read any spectrum provided by the user in a simple *.dat* or *.txt* file format **containing a table with columns reporting: wavelength, flux density, and flux density uncertainty** (header/title/comment lines must be preceeded by a #). (The ``CRETA`` extraction tool, which can be used to produce output spectra that can be fed directly to the ``CAFE`` fitter, exists in the code, but not supported in this first release).

In the notebook, after setting up the necessary paths pointing to where the data are located and where the output results from the fitting will be stored, a spectrum of a region from the galaxy NGC 7469 is downloaded from the cloud. Next, the redshift of the source is set, together with the name of the file containing the spectrum, and the *.ini* and *.cafe* files necessary for the initialization of the fit (they are located in the *CAFE/inp_parfiles/* and *CAFE/opt_parfiles/* folders). These two last files control, respectively: (1) the initial value of the parameters and fit variables for each continuum component (continua, lines, PAHs, user opacities); (2) code-specific variables (we do not recommend changing any of these, as some have not been tested). We refer the user to the `Spectral Fitting <https://github.com/GOALS-survey/CAFE/blob/master/docs/source/fit_spec.rst>` page for a more detailed description.

The spectrum can be plotted together with the initial (default) model decomposition for visual inspection. The user can tweak the initial model by modifying keywords in the *.ini* parameter file. These keywords refer either to model parameters themselves (e.g., peak line fluxes, dust component temperatures, etc.), constraints for each or a combination of model parameters (e.g., line width variation limits, optical depth variation limits for dust components, etc.), specific names of files, or general info needed for the setup.

Once the user is satisfied with the initial guess model, the spectral fitting can be run. ``CAFE`` uses the ``LMFIT`` python package to minimize the data-model residuals using the Trust Region Reflective least-squares method (``least_squares``), and based on the χ2 statistic.

All this steps can be found in the jupyter notebook inside the *notebooks/* folder in the GitHub repository.



Output
------

The ``CAFE`` fitter returns a parameter (``LMFIT``) object containing the best/optimized parameters from which physical quantities (e.g., temperatures) and observables (e.g. line centroids) can be extracted or constructed (e.g., fluxes, based on the line centroid and width). All the parameter information is dumped into python dictionaries for further use and stored in data tables on disk, in the *cafe_output/* directory. In addition, the parameter object is saved to disk as a *.fits* file in a ‘parameter cube‘, which can be read at any later stage to run further fits or generate new dictionaries or data tables (see the jupyter notebook section "RESTORE CAFE SESSION FROM DISK"). The optputs are stored in the *cafe_output/* folder using a default name that is the same as the input spectrum file. In summary the output files from ``CAFE`` are:

* *_cafefit.asdf* : A file containing dictionaries with all the parameter information
* *_parcube.fits* : A file containing all the info necessary to recover a previous CAFE fitting session
* *_fitpars.ini* : A file containing all the fitted parameters in the format of an *.ini* file, which can be used to initialize the fit of any other spectra
* *_fitfigure.png* : A file containing a figure of the last fit performed
* *_linetable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted emission lines
* *_pahtable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted PAH features



Caveats
-------
