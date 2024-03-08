############
Introduction
############

Usage
-----

The current version of ``CAFE`` (v1.0.0) is released together with a jupyter notebook that walks the user through the process of fitting a spectrum obtained from JWST MIRI/MRS cubes produced by the pipeline. The jupyter notebook, named **CAFE_tutorial_NGC7469_1D.ipynb**, can be found in the *notebooks/* folder at the GitHub repository, and can be copied to a working directory created by the user.

``CAFE`` is able to read any spectrum provided by the user in a simple *.dat* or *.txt* file **containing a table with columns reporting: wavelength, flux, and flux uncertainty** (header/title/comment lines must be preceeded by a #). (The ``CRETA`` extraction tool is already implemented in the code and can produce outputs/extractions that can be fed directly to the ``CAFE`` fitter as well. However, while it can be used, it is not supported in the current version of ``CAFE``).

In the notebook, after setting up the necessary paths pointing to where the data are located and where the output results from the fitting will be stored, a spectrum of a region from the galaxy NGC 7469 is downloaded from the cloud. Next, the redshift of the source is set, together with the name of the file containing the spectrum, and the *.ini* and *.cafe* files necessary for the initialization of the fit (they are located in the *CAFE/inp_parfiles/* and *CAFE/opt_parfiles/* folders). These two last files control, respectively: (1) the initial value of the parameters related to the continuum components, as well as what parameters are allowed to vary and by how much for each feature type (continua, lines, PAHs, user opacities); (2) whether some features can be fitted, as well as other more code-specific variables (we do not recommend changing any of these, as some have not been tested). We refer the user to the :ref:`Spectral Fitting<fit_spec.rst>` section for a more detailed description of these files.

After, the spectrum can be read from disk and plotted together with the initial (default) model decomposition for visual inspection. The user can tweak the initial model by modifying keywords in the *.ini* parameter file. These keywords refer either to model parameters themselves (e.g., peak line fluxes, dust component temperatures, etc.), constraints for each or a combination of model parameters (e.g., line width variation limits, optical depth variation limits for dust components, etc.), specific names of files, or in general info needed for the setup.

Once the user is satisfied with the initial guess model, the spectral fitting can be run. ``CAFE`` uses the ``LMFIT`` python package to minimize the data-model residuals using the Trust Region Reflective least-squares method (``least_squares``), and based on the χ2 statistic.

All this steps can be found in the jupyter notebook inside the *notebooks/* folder in the GitHub repository.



Output
------

The ``CAFE`` fitter returns a parameter (``LMFIT``) object containing the best/optimized parameters from which physical quantities (e.g., temperatures) and observables can be extracted (e.g., line peaks) or constructed (e.g., fluxes, based on the feature peak and width). All the parameter information is dumped into python dictionaries for further use and stored in data tables on disk, in the *cafe_output/* directory. In addition, the parameter object is saved to disk as a *.fits* file in a ‘parameter cube‘, which can be read at any later stage to run further fits or generate new dictionaries or data tables (see the jupyter notebook section "RESTORE CAFE SESSION FROM DISK"). All these optputs are stored in the *cafe_output/* folder using a default name that is the same as the input spectrum file. In summary the output files from ``CAFE`` are:

* *_cafefit.asdf* : A file containing dictionaries with all the parameter information
* *_parcube.fits* : A file containing all the info necessary to recover a previous CAFE fitting session
* *_fitpars.ini* : A file containing all the fitted parameters in the format of an *.ini* file, which can be used to initialize the fit of any other spectra
* *_fitfigure.png* : A file containing a figure of the last fit performed
* *_linetable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted emission lines
* *_pahtable_???.ecsv* : A file containing the ??? (=int: intrinsic; =obs: observed) fluxes of the fitted PAH features



Caveats
-------
