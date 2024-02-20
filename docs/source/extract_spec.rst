###################
Spectral Extraction
###################

Extracting a 1D spectrum from the JWST IFU cube with CRETA
----------------------------------------------------------

(Not supported in the current v1.0.0 of ``CAFE``)

The user can employ ``CRETA`` to extract a continuous or discontinuous spectrum from one, some, or all data cubes that the user has copied and made available in the *input_data/* directory. Currently ``CRETA`` supports the extraction of individual spectra; that is, extractions along a single line-of-sight, or position in the sky. ``CRETA`` will extract the spectrum based on a set of parameters provided by the user in a parameter file. Inside it, the user can specify:

   ``cubes``: The cubes to be extracted (currently, only MIRI/MRS; the names provided here need to match or be a sub-string of the cube file names)

   ``user_r_ap``: The radius of the circular aperture used for the extraction

   ``user_ra``, ``user_dec``: RA and Dec coordinates of the source

   ``point_source``: The method of extractions (point source: cone extraction, with radius increasing linearly with wavelength; extended source: cylinder extraction, with constant radius)

   ``lambda_ap``: Wavelength reference for the definition of aperture radius (if point source extractions; ignored otherwise)

   ``aperture_correction``: Whether to perform aperture correction based on PSF cubes

   ``centering``: Whether to perform a centroid centering on the user provided coordinates

   ``lambda_cent``: Wavelength at which the centering will be performed (ignored otherwise)

   ``background_sub``: Whether to perform an annulus-based background subtraction prior to the aperture photometry

   ``r_ann_in``: Inner radius of the background annulus (ignored otherwise)

   ``ann_width``: Width of the background annulus (ignored otherwise)

Options for directory setup (specified in the command execution only):

   ``data_path`` (default: *CRETA/data/*)

   ``PSFs_path`` (default: *CRETA/PSFs/*)

   ``output_path`` (default: *CRETA/extractions/*)

   ``output_filebase_name`` (default: *‘last_result’*)

   ``parfile_path`` (default: *CRETA/param_files/*)

   ``parfile_name`` (default: *‘single_params.txt’*)


``CRETA`` will return a *‘_cube.fits’* file containing the extracted spectrum, which can be fed directly to ``CAFE`` for fitting.

The specific steps to achieve this will be included in future releases in a jupyter notebook in the *notebooks/* folder in the GitHub repository.
