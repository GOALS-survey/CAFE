##########################
Frequently Asked Questions
##########################

**-- How precise the redshift of my source needs to be?**

``CAFE`` is not a redshift finder tool, so the redshift should be at least as precise as the resolving power (FWHM) of the instrument used to obtain the data. ``CAFE`` performs an initial search for the emission lines the user has requested to fit via the *lines.H.recombination_?.txt*, *lines.atomic_?.txt* and *lines.molecular_?.txt* files contained in the *CAFE/inp_parfiles/* folder. The search is done at the expected rest-frame wavelength of the line (after correcting the observed spectrum to rest frame), so if the redshift is not correct, or the line is too red/blue-shifted with respect to the systemic redshift, the finder will fail and ``CAFE`` will ignore the line and will not attempt to fit it. The list of ignored lines are printed in the output given by the jupyter notebook cell that calls for the fit.


**-- What if some of the emission lines in my spectrum have a broad component?**

``CAFE`` can fit up to two Gaussian components to a given emission line. In particular it can be set up to fit specific emission lines with a narrow and a broad components, whose parameters (peak, gamma/width and wavelength) can be controlled independently (i.e., whether to fit them, and if so, by how much they are allowed to vary). To assign a broad component to the narrow component fit by default, the user needs to use the last column found in the tables that control which lines ``CAFE`` will try to fit, that is, the *lines.H.recombination_?.txt*, *lines.atomic_?.txt* and *lines.molecular_?.txt* files contained in the *CAFE/inp_parfiles/* folder. Then the behavior of the parameters can be controlled in the **[PAH & LINE OPTIONS]** in the *.ini* file (see :ref:`Spectral Fitting<fit_spec.rst>` section).


**-- Can I add and fit photometry together with my spectrum?**

The current version of ``CAFE`` does not support adding photometric data to fit together with a spectrum. However, it will be implemented soon in the next release version.
