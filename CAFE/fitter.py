import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from specutils import Spectrum1D, SpectrumList
from astropy.nddata import StdDevUncertainty
import lmfit as lm  # https://dx.doi.org/10.5281/zenodo.11813
import time, datetime
import warnings
import astropy.units as u
from astropy import constants as const
from astropy.stats import mad_std
import pandas as pd
from astropy.io import fits
import astropy
import pickle

# import importlib as imp

from cafe.io import *
from cafe.lib import *
from cafe.params import *
from cafe.get_fit_sequence import get_fit_sequence

cafeio = cafe_io()

# import ipdb


# Helper functions
def _resolve_file_path(file_dir, file_name):
    if file_dir == "extractions/":
        file_dir = os.path.join("./", file_dir)
    return os.path.join(file_dir, file_name)


# def _read_data(file_path, flux_key, flux_unc_key):
#     try:
#         return cafeio.read_cretacube(file_path, flux_key, flux_unc_key)
#     except Exception as exc:
#         raise IOError("Could not open fits file") from exc


def _check_spectrum_units(spec):
    """Check if the spectrum's wavelength unit is in microns."""
    if spec.spectral_axis.unit != "micron":
        raise ValueError("Spectrum1D wavelength must be in micron")


def _adjust_for_redshift(data, z):
    return data / (1 + z)


def cafe_grinder(cafe_model, params, spec, phot):
    """
    Performs iterative fitting of spectral/photometric data.

    Parameters
    ----------
    cafe_model : specmod
        The CAFE spectral modeling object containing configuration and profiles
    params : lmfit.Parameters
        Parameters of the model
    spec : dict
        Spectrum to be fitted with keys 'wave', 'flux', 'flux_unc'
    phot : dict or None
        Photometry to be fitted
    """
    ### Read in global fit settings
    ftol = cafe_model.inopts["FIT OPTIONS"]["FTOL"]
    nproc = cafe_model.inopts["FIT OPTIONS"]["NPROC"]

    acceptFit = False
    ### Limit the number of times the fit can be rerun to avoid infinite loop
    niter = 1
    show = False
    f_pert = 1.01

    while acceptFit is False:

        start = time.time()
        print(
            "Iteration "
            + str(niter)
            + " / "
            + str(cafe_model.inopts["FIT OPTIONS"]["MAX_LOOPS"])
            + "(max):",
            datetime.datetime.now(),
            "-------------",
        )

        old_params = copy.copy(params)  # Parameters of the previous iteration

        # def cafe_callback(params, iter, resid, *args, **kws):
        #     print(f"Iteration {iter}: Current parameters: {params}")

        ### Note that which fitting method is faster here is pretty uncertain, changes by target
        # method = 'leastsq'
        method = "least_squares"  # "leastsq"  #'nelder' if len(spec['wave']) >= 10000 else 'least_squares'
        fitter = lm.Minimizer(
            chisquare,
            params,
            nan_policy="omit",
            fcn_args=(spec, phot, cafe_model.cont_profs, show),
            # iter_cb=cafe_callback,
        )

        try:
            result = fitter.minimize(
                method=method,
                ftol=ftol,
                max_nfev=200 * (len(params) + 1),
            )
        except:
            if result.success is not True:
                raise ValueError("The fit has not been successful.")

        # Do checks on parameters and rerun fit if no errors are returned or the fit is unsuccessful
        if result.success == True:
            end = time.time()
            print(
                "The fitter reached a solution after",
                result.nfev,
                "steps in",
                np.round(end - start, 2),
                "seconds",
            )

            fit_params = copy.copy(
                result.params
            )  # parameters of the current iteration that will not be changed
            params = (
                result.params
            )  # Both params AND result.params will be modified by check_fit_parameters

            acceptFit = check_fit_pars(
                cafe_model,
                spec["wave"],
                spec["flux_unc"],
                fit_params,
                params,
                old_params,
                result.errorbars,
            )

        else:
            end = time.time()
            raise RuntimeError(
                "The fitter reached the maximum number of function evaluations after",
                result.nfev,
                "steps in",
                np.round(end - start, 2) / 60.0,
                "minutes",
            )
            acceptFit = False

        if (
            cafe_model.inopts["FIT OPTIONS"]["FIT_CHK"]
            and niter < cafe_model.inopts["FIT OPTIONS"]["MAX_LOOPS"]
        ):
            if acceptFit is True:
                print(
                    "Successful fit -------------------------------------------------"
                )
            else:
                print("Rerunning fit")
                niter += 1

                # Perturbe the values of parameters that are scaling parameters
                for par in params.keys():
                    if params[par].vary == True:
                        parnames = par.split("_")
                        if (
                            parnames[-1] == "Peak"
                            or parnames[-1] == "FLX"
                            or parnames[-1] == "TMP"
                            or parnames[-1] == "TAU"
                            or parnames[-1] == "RAT"
                        ):
                            if params[par].value * f_pert >= params[par].max:
                                if params[par].value / f_pert > params[par].min:
                                    params[par].value /= f_pert
                            else:
                                params[par].value *= f_pert

        else:
            if acceptFit is True:
                print(
                    "Successful fit -------------------------------------------------"
                )
            else:
                if result.success == True:
                    print(
                        "Hit maximum number of refitting loops. The fit was successful but no errors were returned. Continuing to next spaxel (if any left)"
                    )
                else:
                    print(
                        "Hit maximum number of refitting loops. The fitting was unsuccessful. Continuing to next spaxel (if any left)"
                    )
                acceptFit = True

    return result


class specmod(cafe_io):
    """
    CAFE object for spectral modeling. When initialized, it contains the functionalities
    needed for fitting a 1D spectrum and plotting the results
    """

    def __init__(self, output_dir=None):
        """Initialize the specmod class.

        Parameters
        ----------
        output_dir : str, optional
            Directory where CAFE will write output files (parameter cubes,
            plots, ASDF files). Defaults to a cafe_output/ folder in the
            current working directory.
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'cafe_output')

        self.output_dir = output_dir
        self.file_name = None
        self.result_file_name = None
        self.extract = None
        self.cafe_obj = None

    # def read_parcube_file(self, file_name, file_dir="cafe_results/"):

    #     if file_dir == "cafe_results/":
    #         file_dir = "./" + file_dir
    #     parcube = fits.open(file_dir + file_name)
    #     parcube.info()
    #     self.parcube = parcube
    #     self.parcube_dir = file_dir
    #     self.parcube_name = file_name.replace(".fits", "")
    #     self.result_file_name = self.parcube_name.replace("_parcube", "")
    # parcube.close()

    # def read_spec(
    #     self,
    #     file_name,
    #     xy=None,
    #     file_dir="./extractions/",
    #     flux_key="Flux_st",
    #     flux_unc_key="ERR_st",
    #     trim=True,
    #     keep_next=False,
    #     z=0.0,
    #     is_SED=False,
    #     read_columns=None,
    #     flux_unc=None,
    #     wave_min=None,
    #     wave_max=None,
    #     rwave_min=None,
    #     rwave_max=None,
    # ):

    def _read_file(
        self,
        file_path,
        flux_key,
        flux_unc_key,
        read_columns=None,
        flux_unc=None,
        is_SED=False,
    ):
        """Helper method to read data from various file formats.

        Multiple attributes are stored in the specmod object inherited from 
        cafe_io methods (e.g. read_cretacube()), including:
        
        cube: The cube object
        header: The header of the cube
        waves: The wavelengths of the cube
        fluxes: The fluxes of the cube
        flux_uncs: The flux uncertainties of the cube\
        flux_key: The flux key of the cube
        flux_unc_key: The flux uncertainty key of the cube
        masks: The masks of the cube
        bandnames: The bandnames of the cube
        nz, ny, nx: The number of spaxels in each dimension of the cube
        """
        _, ext = os.path.splitext(file_path)

        if ext == ".fits":
            try:
                # Try reading as CRETA cube first
                self.read_cretacube(file_path, flux_key, flux_unc_key)
                spec = None  # CRETA cube already provides needed format

                # For CRETA cubes, verify units
                if self.cube["FLUX"].header["CUNIT3"] != "um":
                    raise ValueError("The cube wavelength units are not micron")
            except ValueError:
                # If not a CRETA cube, try as custom FITS
                spec = self.customFITSReader(file_path, flux_key, flux_unc_key)
                _check_spectrum_units(spec)
                self.cube = _create_dummy_cube(spec)
        elif ext in [".txt", ".dat", ".csv"]:
            # Handle tabular data
            tab = _read_table_data(file_path, read_columns, flux_unc, is_SED)
            spec = _create_spectrum_from_table(tab)
            _check_spectrum_units(spec)
            self.cube = self._create_dummy_cube(
                spec
            )  # Insert a spectrum into a "3d" cube
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _overlap_removal_and_trim(
        self, trim, keep_next, wave_min, wave_max, rwave_min, rwave_max, z, xy
    ):
        # Remove the overlapping wavelengths between the spectral modules
        # This is implmented only when bandnames are provided
        if hasattr(self, "bandnames"):
            val_inds = (
                trim_overlapping(self.bandnames, keep_next)
                if trim is True
                else np.full(len(self.waves), True)
            )
        else:
            val_inds = np.full(len(self.waves), True)

        # Trim the wavelengths based on the input limits
        if (wave_min is not None and rwave_min is not None) or (
            wave_max is not None and rwave_max is not None
        ):
            raise ValueError(
                "Observed and rest-frame wavelength limits cannot be set simultaneously"
            )
        minWave = (
            wave_min
            if wave_min is not None
            else np.nanmin(self.waves[val_inds])
        )
        maxWave = (
            wave_max
            if wave_max is not None
            else np.nanmax(self.waves[val_inds])
        )
        minWave = (
            rwave_min * (1 + z)
            if rwave_min is not None
            else np.nanmin(self.waves[val_inds])
        )
        maxWave = (
            rwave_max * (1 + z)
            if rwave_max is not None
            else np.nanmax(self.waves[val_inds])
        )

        fit_wave_inds = (self.waves[val_inds] >= minWave) & (
            self.waves[val_inds] <= maxWave
        )

        # Check if the input range is within the spectral range
        if any(fit_wave_inds) is False:
            raise ValueError(
                f"Requested wavelength range ({minWave:.2f}-{maxWave:.2f} μm) "
                + f"is outside the available spectral range"
            )

        # Update the attributes of the specmod object based on trimming
        self.waves = self.waves[val_inds][fit_wave_inds]
        self.bandnames = self.bandnames[val_inds][fit_wave_inds]
        self.header = self.header
        if xy is not None:  # Input data is cube based
            self.nx = 1
            self.ny = 1
            (x, y) = xy
            self.fluxes = self.fluxes[val_inds, y, x][fit_wave_inds]
            self.flux_uncs = self.flux_uncs[val_inds, y, x][fit_wave_inds]
            self.masks = self.masks[val_inds, y, x][fit_wave_inds]
        else:  # Input data is a 1D spectrum
            self.fluxes = self.fluxes[val_inds][fit_wave_inds]
            self.flux_uncs = self.flux_uncs[val_inds][fit_wave_inds]
            self.masks = self.masks[val_inds][fit_wave_inds]

    def _mask_spec(self, x=0, y=0):
        """
        This function masks extreme outliers in the data by
        masking out data points that are more than 1000*MAD away from the median

        Parameters
        ----------
        data: CAFE object that contains the data before masking
        x: int, optional
            x-index of the data point to mask
        y: int, optional
            y-index of the data point to mask
        """

        if self.masks.ndim != 1:  # This is a cube
            mask = self.masks[:, y, x] != 0
            mask[np.isnan(self.fluxes[:, y, x])] = True
            mask[
                (
                    self.fluxes[:, y, x]
                    > np.nanmedian(self.fluxes[:, y, x])
                    + mad_std(self.fluxes[:, y, x], ignore_nan=True) * 1e3
                )  # (data.fluxes[:,y,x] < np.nanmedian(data.fluxes[:,y,x])-mad_std(data.fluxes[:,y,x], ignore_nan=True)*1e3) |\
                | (self.fluxes[:, y, x] < 0)
            ] = True  # | (data.fluxes[:,y,x] < 0)
            flux = self.fluxes[~mask, y, x]
            flux_unc = self.flux_uncs[~mask, y, x]
        else:  # This is a 1D spectrum
            mask = self.masks != 0
            mask[np.isnan(self.fluxes)] = True
            mask[
                (
                    self.fluxes
                    > np.nanmedian(self.fluxes)
                    + mad_std(self.fluxes, ignore_nan=True) * 1e3
                )  # (data.fluxes < np.nanmedian(data.fluxes)-mad_std(data.fluxes, ignore_nan=True)*1e3) |\
                | (self.fluxes < 0)
            ] = True  # | (data.fluxes < 0)
            flux = self.fluxes[~mask]
            flux_unc = self.flux_uncs[~mask]

        wave = self.waves[~mask]
        bandname = self.bandnames[~mask]
        mask_inds = mask[~mask]

        return wave, flux, flux_unc, bandname, mask_inds

    def _redshift_handling(self, waves, fluxes, flux_uncs, z):
        """Adjust wavelengths, fluxes, and flux uncertainties for redshift."""
        if z == 0.0:
            print(
                "WARNING: No redshift provided. Assuming object is already in rest-frame (z=0)."
            )
        adjusted_waves = _adjust_for_redshift(waves, z)
        adjusted_fluxes = _adjust_for_redshift(fluxes, z)
        adjusted_flux_uncs = _adjust_for_redshift(flux_uncs, z)

        return adjusted_waves, adjusted_fluxes, adjusted_flux_uncs

    def _determine_fnu_unit(self, fluxes):
        """Determine the appropriate flux unit based on the median flux value."""
        log_f_med = np.log10(np.nanmedian(fluxes))
        # import pdb

        # pdb.set_trace()
        if log_f_med >= -2:
            return u.Jy
        elif (log_f_med < -2) & (log_f_med >= -5):
            return u.mJy
        elif (log_f_med < -5) & (log_f_med >= -8):
            return u.uJy
        elif (log_f_med < -8) & (log_f_med >= -11):
            return u.nJy
        else:
            return u.pJy
            print(
                "The input flux density values are too small or the median of the flux is negative."
            )
            # raise IOError("The input flux density values are too small.")

    def read_spec(
        self,
        file_name,
        file_dir="./extractions/",
        xy=None,
        flux_key="Flux_st",
        flux_unc_key="Err_st",
        z=0.0,
        trim=True,
        keep_next=False,
        is_SED=False,
        read_columns=None,
        flux_unc=None,
        wave_min=None,
        wave_max=None,
        rwave_min=None,
        rwave_max=None,
    ):
        """Read and process spectral data from various file formats.

        Parameters
        ----------
        file_name : str
            Name of the input file. Supported formats:
            - Plain text (.txt, .dat) with wavelength, flux and uncertainty columns
            - CSV file from CRETA
            - FITS cube from CRETA
        xy : tuple, optional
            (x,y) coordinates of spaxel to extract when reading a creta or FITS cube
        file_dir : str, default: './extractions/'
            Directory containing the input file
        flux_key : str, default: 'Flux_st'
            Column name for flux values in the input files
        flux_unc_key : str, default: 'ERR_st'
            Column name for flux uncertainties in the input files
        trim : bool, default: True
            Whether to trim overlapping wavelengths in CRETA spectra
        keep_next : bool, default: False
            If trim=True, whether to keep longer wavelength data in overlaps
        z : float, default: 0.
            Source redshift
        is_SED : bool, default: False
            Whether input is a low-resolution SED (currently unsupported)
        read_columns : list of str, optional
            Names of columns to read from table input ('wave', 'flux', 'flux_unc')
        flux_unc : float, optional
            Override flux uncertainties with this fraction of flux values
        wave_min : float, optional
            Minimum observed wavelength to keep
        wave_max : float, optional
            Maximum observed wavelength to keep
        rwave_min : float, optional
            Minimum rest-frame wavelength to keep
        rwave_max : float, optional
            Maximum rest-frame wavelength to keep

        Notes
        -----
        wave_min/wave_max and rwave_min/rwave_max cannot be used simultaneously
        """
        # Read and process spectral data from various file formats.
        file_path = _resolve_file_path(file_dir, file_name)

        # Read and process data from various file formats and store the attributes
        self._read_file(
            file_path,
            flux_key,
            flux_unc_key,
            read_columns=read_columns,
            flux_unc=flux_unc,
            is_SED=is_SED,
        )

        self.file_name = file_name  # cube.cube.filename().split('/')[-1]
        self.result_file_name = os.path.splitext(self.file_name)[0]

        # Overlap removal and trimming
        self._overlap_removal_and_trim(
            trim, keep_next, wave_min, wave_max, rwave_min, rwave_max, z, xy
        )

        # Redshift handling
        self.z = z
        self.waves, self.fluxes, self.flux_uncs = self._redshift_handling(
            self.waves, self.fluxes, self.flux_uncs, z
        )

        # Determine which flux unit should be used internally to make the flux values close to unity
        self._fnu_unit = self._determine_fnu_unit(self.fluxes)

        return self

    def read_phot(self, file_name, file_dir="./extractions/"):

        # tab = Table.read(file_dir+file_name, format='ascii.basic', names=['name', 'wave', 'flux', 'flux_unc', 'width'], data_start=0, comment='#')
        # self.pwaves = tab['wave'] / self.z
        # self.pfluxes = tab['flux'] / self.z
        # self.pflux_uncs = tab['flux_unc'] / self.z
        # self.pbandnames = tab['name']
        # self.pwidths = tab['width']

        print("Phot data:", file_dir + file_name)
        tab = np.genfromtxt(file_dir + file_name, comments="#", dtype="str")
        self.pwaves = tab[:, 1].astype(float) / (1 + self.z)
        self.pfluxes = tab[:, 2].astype(float) / (1 + self.z)
        self.pflux_uncs = tab[:, 3].astype(float) / (1 + self.z)
        self.pbandnames = tab[:, 0]
        self.pwidths = tab[:, 4].astype(float) / (1 + self.z)

    def input_param(
        self,
        inparfile,
        optfile,
        init_parcube=None,
        cont_profs=None,
    ):
        """
        Function to read in the init parameters used in the fit.
        """
        # Check that the input param files exist
        if not os.path.exists(inparfile):
            raise FileNotFoundError(f"Parameter file not found: {inparfile}")
        if not os.path.exists(optfile):
            raise FileNotFoundError(f"Options file not found: {optfile}")

        self.inparfile = inparfile
        self.optfile = optfile

        self.inpars = cafeio.read_inifile(inparfile)
        self.inopts = cafeio.read_inifile(optfile)

        self.init_parcube = init_parcube
        self.cont_profs = cont_profs

        # ++++++++++++++++++++++++++++++++++
        # Output result from lmfit
        # ++++++++++++++++++++++++++++++++++
        # Get the spectrum after masking extreme outliers.
        # Data points that are more than 1000*MAD away from the median
        wave, flux, flux_unc, bandname, mask = self._mask_spec()

        # Assemble it in a Spectrum1D for the profile generator and in a dictionary for the fitting and plotting
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            flux=flux * (u.Jy.to(self._fnu_unit)) * u.Jy,
            uncertainty=StdDevUncertainty(
                flux_unc * u.Jy.to(self._fnu_unit) * u.Jy
            ),
            redshift=self.z,
        )

        # Create spec_dict and phot_dict which will be used in the actual fitting
        # ---------------------------------------------------------------------
        spec_dict = {
            "wave": wave,
            "flux": flux * u.Jy.to(self._fnu_unit),
            "flux_unc": flux_unc * u.Jy.to(self._fnu_unit),
        }

        self.spec_dict = spec_dict

        # See if the user wants to fit photometric data and check whether they have been read
        self.fitphot = self.inopts["MODEL OPTIONS"]["FITPHOT"]
        if self.fitphot is True:
            if hasattr(self, "pwaves") is False:
                raise AttributeError(
                    "You are trying to fit photometry but the data have not "
                    "been loaded. Use cafe.read_phot() to do so."
                )
            phot_dict = {
                "wave": self.pwaves,
                "flux": self.pfluxes * u.Jy.to(self._fnu_unit),
                "flux_unc": self.pflux_uncs * u.Jy.to(self._fnu_unit),
                "width": self.pwidths,
            }
        else:
            phot_dict = None

        self.phot_dict = phot_dict
        # ---------------------------------------------------------------------

        # Initiate CAFE param generator
        param_gen = CAFE_param_generator(
            spec, self.inpars, self.inopts
        )

        print("Generating parameter cube with initial/full parameter object")
        all_params = param_gen.make_parobj(get_all=True)
        cube_gen = CAFE_cube_generator(self)

        # This is the parcube before the fit
        self.parcube = cube_gen.make_parcube(all_params)

        print("Generating parameter object")
        init_params = param_gen.make_parobj()

        if self.init_parcube is not None:  # If initial parameters are provided
            print(
                "The parameters in the parcube provided for initialization will be used to initialize the parameter object"
            )

            # cube_params is the input parameter object, which is different from the auto generated init_params
            cube_params = parcube2parobj(
                self.init_parcube, init_parobj=init_params
            )
            params = param_gen.make_parobj(
                updated_parobj=cube_params,
                get_all=True,
                init_parobj=init_params,
            )
        else:  # If initial parameters are not provided
            params = init_params

        # Initiate CAFE profile loader
        print("Generating continuum profiles")
        prof_gen = CAFE_prof_generator(
            spec, self.inpars, self.inopts, phot_dict
        )

        if self.cont_profs is None:
            self.cont_profs = (
                prof_gen.make_cont_profs()
            )  # Generate the selected unscaled continuum profiles
        # else:
        #     self.cont_profs = cont_profs

        unfixed_params = [
            True if params[par].vary == True else False for par in params.keys()
        ]
        print(
            "Fitting",
            unfixed_params.count(True),
            "unfixed parameters, out of the",
            len(params),
            "defined in the parameter object",
        )

        self.params = params

    def fit_spec(
        self,
        output_path=None,
        # init_parcube=False, # This has been moved to the input_param function
        cont_profs=None,
    ):
        # """
        # Output result from lmfit
        # """
        # # Get the spectrum after masking extreme outliers.
        # # Data points that are more than 1000*MAD away from the median
        # wave, flux, flux_unc, bandname, mask = self._mask_spec()

        # # Assemble it in a Spectrum1D for the profile generator and in a dictionary for the fitting and plotting
        # spec = Spectrum1D(
        #     spectral_axis=wave * u.micron,
        #     flux=flux * (u.Jy.to(self._fnu_unit)) * u.Jy,
        #     uncertainty=StdDevUncertainty(
        #         flux_unc * u.Jy.to(self._fnu_unit) * u.Jy
        #     ),
        #     redshift=self.z,
        # )

        # # Create spec_dict and phot_dict which will be used in the actual fitting
        # # ---------------------------------------------------------------------
        # spec_dict = {
        #     "wave": wave,
        #     "flux": flux * u.Jy.to(self._fnu_unit),
        #     "flux_unc": flux_unc * u.Jy.to(self._fnu_unit),
        # }

        # self.spec_dict = spec_dict

        # # See if the user wants to fit photometric data and check whether they have been read
        # self.fitphot = self.inopts["MODEL OPTIONS"]["FITPHOT"]
        # if self.fitphot is True:
        #     if hasattr(self, "pwaves") is False:
        #         raise AttributeError(
        #             "You are trying to fit photometry but the data have not "
        #             "been loaded. Use cafe.read_phot() to do so."
        #         )
        #     phot_dict = {
        #         "wave": self.pwaves,
        #         "flux": self.pfluxes * u.Jy.to(self._fnu_unit),
        #         "flux_unc": self.pflux_uncs * u.Jy.to(self._fnu_unit),
        #         "width": self.pwidths,
        #     }
        # else:
        #     phot_dict = None

        # self.phot_dict = phot_dict
        # # ---------------------------------------------------------------------

        # # Initiate CAFE param generator
        # param_gen = CAFE_param_generator(
        #     spec, self.inpars, self.inopts, cafe_path=self.cafe_dir
        # )
        # _, outPath = cafeio.init_paths(
        #     self.inopts,
        #     cafe_path=self.cafe_dir,
        #     file_name=self.result_file_name,
        #     output_path=output_path,
        # )

        # print("Generating parameter cube with initial/full parameter object")
        # all_params = param_gen.make_parobj(get_all=True)
        # cube_gen = CAFE_cube_generator(self)

        # # This is the parcube before the fit
        # parcube = cube_gen.make_parcube(all_params)

        # print("Generating parameter object")
        # init_params = param_gen.make_parobj()

        # if self.init_parcube is not None:  # If initial parameters are provided
        #     print(
        #         "The parameters in the parcube provided for initialization will be used to initialize the parameter object"
        #     )

        #     # cube_params is the input parameter object, which is different from the auto generated init_params
        #     cube_params = parcube2parobj(
        #         self.init_parcube, init_parobj=init_params
        #     )
        #     params = param_gen.make_parobj(
        #         updated_parobj=cube_params,
        #         get_all=True,
        #         init_parobj=init_params,
        #     )
        # else:  # If initial parameters are not provided
        #     params = init_params

        # # Initiate CAFE profile loader
        # print("Generating continuum profiles")
        # prof_gen = CAFE_prof_generator(
        #     spec, self.inpars, self.inopts, phot_dict, cafe_path=self.cafe_dir
        # )

        # if self.cont_profs is None:
        #     self.cont_profs = (
        #         prof_gen.make_cont_profs()
        #     )  # Generate the selected unscaled continuum profiles
        # # else:
        # #     self.cont_profs = cont_profs

        # unfixed_params = [
        #     True if params[par].vary == True else False for par in params.keys()
        # ]
        # print(
        #     "Fitting",
        #     unfixed_params.count(True),
        #     "unfixed parameters, out of the",
        #     len(params),
        #     "defined in the parameter object",
        # )

        # Set IO paths
        outPath = cafeio.get_output_path(self.output_dir, self.result_file_name)

        # Fit the spectrum
        result = cafe_grinder(self, self.params, self.spec_dict, self.phot_dict)
        print(
            "The VGRAD of the spectrum is:",
            result.params["VGRAD"].value,
            "[km/s]",
        )

        self.params = result.params

        # Inject the result into the pre-existing parameter cube and update parcube with the new result
        parcube = parobj2parcube(result.params, self.parcube)
        self.parcube = parcube

        # Get compdict
        flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, vgrad = (
            get_model_fluxes(
                self.params,
                self.spec_dict["wave"],
                self.cont_profs,
                verbose_output=True,
            )
        )

        compdict = {
            "CompFluxes": CompFluxes,
            "CompFluxes_0": CompFluxes_0,
            "extComps": extComps,
            "e0": e0,
            "tau0": tau0,
        }

        self.compdict = compdict

        self.save_products(outPath)

        return self

    def save_products(self, product_dir):

        # Save products to disk
        self.product_dir = product_dir

        # Parcube
        self.parcube_name = self.result_file_name + "_parcube"
        output_path_parameters = os.path.join(
            self.product_dir, self.parcube_name + ".fits"
        )
        print(f"Saving parameters in cube to disk: {output_path_parameters}")
        self.parcube.writeto(output_path_parameters, overwrite=True)

        ## Save fCON cube to disk
        cube_gen = CAFE_cube_generator(self)
        self.contcube = cube_gen.make_profcube(
            self.inpars,
            self.inopts,
            prof_names={
                "fCon": ["Comp", "fCON"],
                "fDSK": ["Comp", "fDSK"],
                "fHOT": ["Comp", "fHOT"],
            },
        )
        self.contcube_name = self.result_file_name + "_contcube"

        output_path_total_conti = os.path.join(
            self.product_dir, self.contcube_name + ".fits"
        )
        print(
            f"Saving total continuum profile in cube to disk: {output_path_total_conti}"
        )
        self.contcube.writeto(output_path_total_conti, overwrite=True)

        # self.compdict_name = self.result_file_name+'_compdict'
        # print('Saving component profiles in dictionary to disk:',self.compdict_dir+self.compdict_name+'.pkl')
        ##compcube.writeto(self.compcube_dir+self.compdict_name+'.fits', overwrite=True)
        # with open(self.compcube_dir+self.compdict_name+'.pkl', 'wb') as f:
        #    pickle.dump(compdict, f)

        # Write best fit as an .ini parameter file
        output_path_init_file = os.path.join(
            self.product_dir, self.result_file_name + "_fitpars.ini"
        )
        print(f"Saving init file to disk: {output_path_init_file}")
        cafeio.write_inifile(self.params, self.inpars, output_path_init_file)

        # Save component dictionary .asdf to disk
        output_path_asdf = os.path.join(
            self.product_dir, self.result_file_name + "_cafefit"
        )
        print(f"Saving parameters in asdf to disk: {output_path_asdf}")
        cafeio.save_asdf(self, file_name=output_path_asdf)

        output_path_fit_figure = os.path.join(
            self.product_dir, self.result_file_name + "_fitfigure.png"
        )
        print(f"Saving figure in png to disk: {output_path_fit_figure}")

        flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, vgrad = (
            get_model_fluxes(
                self.params,
                self.spec_dict["wave"],
                self.cont_profs,
                verbose_output=True,
            )
        )
        gauss, drude, gauss_opc = get_feat_pars(
            self.params, apply_vgrad2waves=True
        )  # params consisting all the fitted parameters

        self.plot_spec_fit(savefig=output_path_fit_figure)

        # Save the PAH table
        if self.inopts["FIT OPTIONS"]["FITPAHS"] is True:
            output_fn = os.path.join(
                self.product_dir, self.result_file_name + "_pahtable.ecsv"
            )
            self.pahtable = cafeio.pah_table(
                self.parcube,
                fnu_unit=self._fnu_unit,
                compdict=self.compdict,
                pah_obs=True,
                savetbl=output_fn,
            )

        # Save the line table
        if self.inopts["FIT OPTIONS"]["FITLINS"] is True:
            output_fn = os.path.join(
                self.product_dir, self.result_file_name + "_linetable.ecsv"
            )
            self.linetable = cafeio.line_table(
                self.parcube,
                fnu_unit=self._fnu_unit,
                compdict=self.compdict,
                line_obs=True,
                savetbl=output_fn,
            )

    def plot_spec_ini(self):
        """
        Plot the SED generated by the inital parameters
        """
        wave, flux, flux_unc, bandname, mask = self._mask_spec()
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            # flux=flux*u.Jy*(u.Jy*self.fnu_unit),
            flux=flux * u.Jy,  # *(u.Jy.to(self.fnu_unit)),
            uncertainty=StdDevUncertainty(flux_unc),
            redshift=self.z,
        )
        spec_dict = {"wave": wave, "flux": flux, "flux_unc": flux_unc}

        self.fitphot = self.inopts["MODEL OPTIONS"]["FITPHOT"]
        if self.fitphot is True:
            if hasattr(self, "pwaves") is False:
                raise AttributeError(
                    "You are trying to fit photometry but the data have not been loaded. Use cafe.read_phot() to do so."
                )
            phot_dict = {
                "wave": self.pwaves,
                "flux": self.pfluxes,
                "flux_unc": self.pflux_uncs,
                "width": self.pwidths,
            }
        else:
            phot_dict = None

        # Plot features based on inital intput parameters
        # -----------------------------------------------

        # Initiate CAFE param generator and make parameter file
        print(
            "Generating continuum profiles for guess model from the .ini file"
        )
        param_gen = CAFE_param_generator(
            spec, self.inpars, self.inopts
        )
        params = param_gen.make_parobj()

        if self.init_parcube is not None:
            print(
                "The initial parameters will be set to the values from the parameter cube provided"
            )
            cube_params = parcube2parobj(self.init_parcube, init_parobj=params)
            params = param_gen.make_parobj(
                updated_parobj=cube_params, get_all=True
            )

        # Initiate CAFE profile loader and make cont_profs
        prof_gen = CAFE_prof_generator(
            spec, self.inpars, self.inopts, phot_dict
        )
        cont_profs = (
            prof_gen.make_cont_profs()
        )  # load the selected unscaled continuum profiles

        # Scale continuum profiles with parameters and get spectra
        flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, _ = (
            get_model_fluxes(params, wave, cont_profs, verbose_output=True)
        )

        # Get feature spectrum out of the feature parameters
        gauss, drude, gauss_opc = get_feat_pars(params, apply_vgrad2waves=True)

        # spec_dict and phot_dict have the flux unit of Jy
        (fig, ax1, ax2) = cafeplot(
            spec_dict,
            phot_dict,
            self._fnu_unit,
            CompFluxes,
            gauss,
            drude,
            pahext=extComps["extPAH"],
        )

        return (fig, ax1, ax2)

    def plot_spec_fit(self, savefig=None):
        """
        Plot the spectrum itself. If params already exists, plot the fitted results as well.
        """
        wave, flux, flux_unc, bandname, mask = self._mask_spec()
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            # flux=flux*u.Jy*(u.Jy*self.fnu_unit),
            flux=flux * u.Jy,  # *(u.Jy.to(self._fnu_unit)),
            uncertainty=StdDevUncertainty(flux_unc),
            redshift=self.z,
        )
        # spec = Spectrum1D(spectral_axis=wave*u.micron,
        #                   flux=flux*(u.Jy.to(self._fnu_unit))*self._fnu_unit,
        #                   uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
        spec_dict = {"wave": wave, "flux": flux, "flux_unc": flux_unc}

        self.fitphot = self.inopts["MODEL OPTIONS"]["FITPHOT"]
        if self.fitphot is True:
            if hasattr(self, "pwaves") is False:
                raise AttributeError(
                    "You are trying to fit photometry but the data have not been loaded. Use cafe.read_phot() to do so."
                )
            phot_dict = {
                "wave": self.pwaves,
                "flux": self.pfluxes,
                "flux_unc": self.pflux_uncs,
                "width": self.pwidths,
            }
        else:
            phot_dict = None

        if hasattr(self, "parcube") is False:
            raise ValueError("The spectrum is not fitted yet")
        else:
            params = parcube2parobj(self.parcube)

        prof_gen = CAFE_prof_generator(
            spec, self.inpars, self.inopts, phot_dict
        )
        cont_profs = prof_gen.make_cont_profs()

        flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, vgrad = (
            get_model_fluxes(params, wave, cont_profs, verbose_output=True)
        )

        gauss, drude, gauss_opc = get_feat_pars(
            params, apply_vgrad2waves=True
        )  # params consisting all the fitted parameters

        # sedfig, chiSqrFin = sedplot(wave, flux, flux_unc, CompFluxes, weights=weight, npars=result.nvarys)
        # cafefig, ax1, ax2 = cafeplot(spec_dict, phot_dict, CompFluxes, gauss, drude, fnu_unit=self.fnu_unit, vgrad=vgrad, pahext=extComps['extPAH'])

        # spec_dict and phot_dict have the flux unit of Jy
        fig, ax1, ax2 = cafeplot(
            spec_dict,
            phot_dict,
            self._fnu_unit,
            CompFluxes,
            gauss,
            drude,
            pahext=extComps["extPAH"],
        )

        if savefig is not None:
            fig.savefig(savefig, dpi=500, format="png", bbox_inches="tight")

        return (fig, ax1, ax2)

    def plot_spec(self, savefig=None):

        wave, flux, flux_unc, bandname, mask = self._mask_spec()
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            flux=flux * u.Jy,
            uncertainty=StdDevUncertainty(flux_unc),
            redshift=self.z,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            spec.spectral_axis, spec.flux, linewidth=1, color="k", alpha=0.8
        )
        ax.scatter(
            spec.spectral_axis, spec.flux, marker="o", s=8, color="k", alpha=0.7
        )
        ax.errorbar(
            spec.spectral_axis.value,
            spec.flux.value,
            yerr=spec.uncertainty.quantity.value,
            fmt="none",
            ecolor="gray",
            alpha=0.4,
        )

        ax.set_xlabel(
            "Wavelength (" + spec.spectral_axis.unit.to_string() + ")"
        )
        ax.set_ylabel("Flux (" + spec.flux.unit.to_string() + ")")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if savefig is not None:
            cafefig.savefig(savefig, dpi=500, format="png", bbox_inches="tight")

        return (fig, ax)

    # TO BE DEPRECATED AS ALL READ AND WRITE FUNCTIONS SHOULD BE IN CAFE IO
    def save_result(
        self, asdf=True, pah_tbl=True, line_tbl=True, file_name=None
    ):

        if hasattr(self, "parcube") is False:
            raise AttributeError(
                "The spectrum is not fitted yet. Missing fitted result - parcube."
            )

        params = self.parcube.params
        wave = self.spec.spectral_axis.value

        if asdf is True:
            params_dict = params.valuesdict()

            # Get fitted results
            gauss, drude, gauss_opc = get_feat_pars(
                params, apply_vgrad2waves=True
            )  # params consisting all the fitted parameters
            flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, _ = (
                get_model_fluxes(
                    params, wave, self.cont_profs, verbose_output=True
                )
            )

            # Get PAH powers (intrinsic/extinguished)
            pah_power_int = drude_int_fluxes(CompFluxes["wave"], drude)
            pah_power_ext = drude_int_fluxes(
                CompFluxes["wave"], drude, ext=extComps["extPAH"]
            )

            # Quick hack for output PAH and line results
            output_gauss = {
                "wave": gauss[0],
                "width": gauss[1],
                "peak": gauss[2],
                "name": gauss[3],
                "strength": np.zeros(len(gauss[3])),
            }  #  Should add integrated gauss
            output_drude = {
                "wave": drude[0],
                "width": drude[1],
                "peak": drude[2],
                "name": drude[3],
                "strength": pah_power_int.value,
            }

            # Make dict to save in .asdf
            obsspec = {
                "wave": self.wave,
                "flux": self.flux,
                "flux_unc": self.flux_unc,
            }
            cafefit = {
                "cafefit": {
                    "obsspec": obsspec,
                    "fitPars": params_dict,
                    "CompFluxes": CompFluxes,
                    "CompFluxes_0": CompFluxes_0,
                    "extComps": extComps,
                    "e0": e0,
                    "tau0": tau0,
                    "gauss": output_gauss,
                    "drude": output_drude,
                }
            }

            # Save output result to .asdf file
            # Note: asdf 3.0 removed overwrite= parameter; write_to overwrites by default
            target = AsdfFile(cafefit)
            if file_name is None:
                out_path = os.path.join(self.output_dir, "last_unnamed_cafefit.asdf")
            else:
                out_path = file_name + ".asdf"
            target.write_to(out_path)


class cubemod(specmod):
    """
    CAFE object for cube fitting. Inherits from specmod.
    """

    def __init__(self, output_dir=None):
        """Initialize the cubemod class, inheriting from specmod.

        Parameters
        ----------
        output_dir : str, optional
            Directory where CAFE will write output files. Defaults to a
            cafe_output/ folder in the current working directory.
        """
        # Initialize parent class
        super().__init__(output_dir=output_dir)

        # Add cube-specific attributes
        self.nx = None
        self.ny = None
        self.nz = None

    def _create_parcube_3d(self, wave, flux, flux_unc):
        """
        Create the 3D parcube to store the results based on the spaxel that
        has the highest SNR
        """
        # Get the Spectrum1D object for CAFE_prof_generator()
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            flux=flux * (u.Jy.to(self._fnu_unit)) * u.Jy,
            uncertainty=StdDevUncertainty(
                flux_unc * u.Jy.to(self._fnu_unit) * u.Jy
            ),
        )

        # Initiate CAFE param generator
        param_gen = CAFE_param_generator(
            spec, self.inpars, self.inopts
        )

        print(
            "Generating parameter cube for cubefit with initial/full parameter object"
        )
        all_params = param_gen.make_parobj(get_all=True)
        cube_gen = CAFE_cube_generator(self)

        # This is the parcube before performing the fit
        parcube = cube_gen.make_parcube(all_params)

        return parcube

    def cubefit(
        self,
        file_name,
        file_dir,
        inparfile,
        optfile,
        z=0.0,
        flux_key="FLUX_ST",
        flux_unc_key="ERR_ST",
        pattern="snr",
        rwave_min=None,
        rwave_max=None,
        output_path=None,
    ):
        """
        Fit the entire cube using the specified parameters and options.

        Parameters
        ----------
        file_name : str
            The filename of the source cube. The input file requires to be a creta cube.
        file_dir : str
            The directory where the source file is located.
        inparfile : str
            Path to the initial parameter file.
        optfile : str
            Path to the options file.
        z : float, optional
            Redshift of the source. Default is 0.0.
        flux_key : str, optional
            Key for the flux data in the cube. Default is 'FLUX_ST'.
        flux_unc_key : str, optional
            Key for the flux uncertainty data in the cube. Default is 'ERR_ST'.
        rwave_min : float, optional
            Minimum rest-frame wavelength to consider.
        rwave_max : float, optional
            Maximum rest-frame wavelength to consider.
        output_path : str, optional
            Directory to save the output results.

        Returns
        -------
        None
            This function does not return any value. It processes the cube and
            saves the results to the specified output path.
        """

        # Read and process spectral data from various file formats.
        file_path = _resolve_file_path(file_dir, file_name)

        # Get the fitting sequence
        hdul = fits.open(file_path)

        # Check if the flux_key and flux_unc_key exist
        if flux_key not in [hdu.name for hdu in hdul]:
            raise ValueError(
                f"The flux key {flux_key} does not exist in the cube"
            )
        if flux_unc_key not in [hdu.name for hdu in hdul]:
            raise ValueError(
                f"The flux uncertainty key {flux_unc_key} does not exist in the cube"
            )

        # Check if the cube has the correct dimension
        if len(np.shape(hdul[flux_key].data)) != 3:
            raise ValueError("The cube is not in 3D")

        fluxes = hdul[flux_key].data
        flux_uncs = hdul[flux_unc_key].data

        snr_cube = fluxes / flux_uncs

        # Get the median SNR map
        snr_image = np.nanmedian(snr_cube[:, :, :], axis=0)

        # Get sequence of spaxels to fit, with ind_seq being (y, x)
        ind_seq, ref_ind_seq = get_fit_sequence(snr_image, sorting_seq=pattern)
        print(
            f"Highest SNR spaxel is: {np.flip((ind_seq[0][0], ind_seq[1][0]))}"
        )

        # Read and process data from various file formats and store the attributes
        self._read_file(
            file_path,
            flux_key=flux_key,
            flux_unc_key=flux_unc_key,
        )

        # For creating the 3D parcube, masked spectrum is not necessary,
        # so there is no need to call _mask_spec()

        processed_spx = 0
        total_spx = len(ind_seq[0])
        for i in range(total_spx):
            processed_spx += 1
            percentage = (processed_spx / total_spx) * 100
            print(
                f"\rProcessing cubefit spaxel {processed_spx}/{total_spx} ({percentage:.1f}%)",
                end="",
                flush=True,
            )

            x, y = ind_seq[1][i], ind_seq[0][i]
            # Read the spectrum from the cube

            self.read_spec(
                file_name,
                file_dir,
                xy=(x, y),
                flux_key=flux_key,
                flux_unc_key=flux_unc_key,
                z=z,
                trim=True,
                keep_next=False,
                # is_SED=False,
                wave_min=None,
                wave_max=None,
                rwave_min=rwave_min,
                rwave_max=rwave_max,
            )

            # Initialize the parameter file
            # -----------------------------
            if i == 0:  # First spaxel
                # Create the 3D parcube to store the results based on the spaxel
                # with the highest SNR
                wave = self.waves
                flux = self.fluxes
                flux_unc = self.flux_uncs

                # Initialize the parameter file
                self.input_param(inparfile=inparfile, optfile=optfile)

                # This is the parcube before performing the fit
                parcube_3d = self._create_parcube_3d(wave, flux, flux_unc)

                self.plot_spec_ini()

            else:  # Subsequent spaxels
                self.input_param(
                    inparfile=inparfile,
                    optfile=optfile,
                    init_parcube=self.parcube,
                )

                # cube_params = parcube2parobj(
                #     self.parcube,
                #     ref_ind_seq[1, y, x],
                #     ref_ind_seq[0, y, x],
                #     init_parobj=init_params,
                # )  # indexation is (x=1,y=0)

                # # The params file is regenerated but with the VARY, LIMS and ARG reset based on the new VALUES injected
                # params = param_gen.make_parobj(
                #     parobj_update=cube_params,
                #     get_all=True,
                #     init_parobj=init_params,
                # )
            self.inpars["CONTINUA INITIAL VALUES AND OPTIONS"]["COO_FLX"] = [
                0.0,
                False,
                0,
                2,
            ]
            self.inpars["CONTINUA INITIAL VALUES AND OPTIONS"]["HOT_FLX"] = [
                0.85,
                False,
                0,
                2,
            ]
            self.inpars["CONTINUA INITIAL VALUES AND OPTIONS"]["WRM_FLX"] = [
                1.5,
                True,
                0,
                5,
            ]

            cafe_obj = self.fit_spec(output_path=output_path)

            # Get the parameters for the current spaxel
            # cube_params = parcube2parobj(init_parcube, init_parobj=init_params)

            # We inject the common params in the parameter cube of the reference (fitted) spaxel
            # assigned for the initialization of the current spaxel to the current spaxel params
            parcube_3d = parcube2parobj(
                cafe_obj.parcube,
                x=x,
                y=y,
                init_parobj=init_params,
            )  # indexation is (x=1,y=0)

    #         # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #         # if it's not the first spaxel
    #         if snr_ind != (ind_seq[0][0], ind_seq[1][0]):
    #             print(
    #                 "Current spaxel",
    #                 np.flip(snr_ind),
    #                 "will be initialized with results from spaxel",
    #                 np.flip(
    #                     (
    #                         ref_ind_seq[0, snr_ind[0], snr_ind[1]],
    #                         ref_ind_seq[1, snr_ind[0], snr_ind[1]],
    #                     )
    #                 ),
    #             )  # ,
    #             #      'and set to a SB inppar file')
    #             #
    #             # self.inpar_fns[snr_ind[0], snr_ind[1]] = inparfile  # (y,x)

    #             # We inject the common params in the parameter cube of the reference (fitted) spaxel
    #             # assigned for the initialization of the current spaxel to the current spaxel params
    #             cube_params = parcube2parobj(
    #                 parcube,
    #                 ref_ind_seq[1, snr_ind[0], snr_ind[1]],
    #                 ref_ind_seq[0, snr_ind[0], snr_ind[1]],
    #                 init_parobj=init_params,
    #             )  # indexation is (x=1,y=0)

    #             # The params file is regenerated but with the VARY, LIMS and ARG reset based on the new VALUES injected
    #             params = param_gen.make_parobj(
    #                 parobj_update=cube_params,
    #                 get_all=True,
    #                 init_parobj=init_params,
    #             )

    #         else:  # The first spaxel
    #             if init_parcube is None:
    #                 print(
    #                     "The params will be set to the parameters of the parcube provided for initialization"
    #                 )
    #                 cube_params = parcube2parobj(
    #                     init_parcube, init_parobj=init_params
    #                 )
    #                 params = param_gen.make_parobj(
    #                     parobj_update=cube_params,
    #                     get_all=True,
    #                     init_parobj=init_params,
    #                 )
    #             else:
    #                 params = init_params

    #         unfixed_params = [
    #             True if params[par].vary == True else False
    #             for par in params.keys()
    #         ]

    #         print(
    #             "Fitting",
    #             unfixed_params.count(True),
    #             "unfixed parameters, out of the",
    #             len(params),
    #             "defined in the parameter object",
    #         )
    #     # self.read_cube(file_name, file_dir, flux_key, flux_unc_key, z=z)

    #     # # Get fluxes for get_fit_sequence
    #     # fluxes = self.fluxes

    #     # # Set up initial parameters
    #     # self.input_param(
    #     #     inparfile=inparfile,
    #     #     optfile=optfile,
    #     #     init_parcube=None,
    #     #     cont_profs=None,
    #     # )

    #     # # Fit the cube
    #     # self.fit_cube(
    #     #     output_path=output_path, rwave_min=rwave_min, rwave_max=rwave_max
    #     # )

    #     # # Save the results
    #     # self.save_products(output_path)

    # # def read_parcube_file(self, file_name, file_dir="cafe_results/"):

    # #     if file_dir == "cafe_results/":
    # #         file_dir = "./" + file_dir
    # #     parcube = fits.open(file_dir + file_name)
    # #     parcube.info()
    # #     self.parcube = parcube
    # #     self.parcube_dir = file_dir
    # #     self.parcube_name = file_name.replace(".fits", "")
    # #     self.result_file_name = self.parcube_name.replace("_parcube", "")
    # # parcube.close()

    # # This perhaps can be merged with the read_parcube_file above
    # # def input_param(
    # #     self,
    # #     inparfile,
    # #     optfile,
    # #     init_parcube=False,
    # #     cont_profs=None,
    # # ):
    # #     """

    # #     Function to read in the init parameters used in the fit.
    # #     """
    # #     # Check that the input param files exist
    # #     if not os.path.exists(inparfile):
    # #         raise FileNotFoundError(f"Parameter file not found: {inparfile}")
    # #     if not os.path.exists(optfile):
    # #         raise FileNotFoundError(f"Options file not found: {optfile}")

    # #     self.inparfile = inparfile
    # #     self.optfile = optfile

    # #     self.inpars = cafeio.read_inifile(inparfile)
    # #     self.inopts = cafeio.read_inifile(optfile)

    # #     self.init_parcube = init_parcube
    # #     self.cont_profs = cont_profs

    # def read_cube(
    #     self,
    #     file_name,
    #     file_dir="./extractions/",
    #     flux_key="Flux_st",
    #     flux_unc_key="Err_st",
    #     trim=True,
    #     keep_next=False,
    #     z=0.0,
    # ):
    #     """Read and process spectral cube data.

    #     Parameters
    #     ----------
    #     file_name : str
    #         Name of the cube file to read
    #     file_dir : str, default: './extractions/'
    #         Directory containing the input file
    #     extract : str, default: 'Flux_st'
    #         Column name containing stitched spectra when reading CRETA-produced cubes
    #     trim : bool, default: True
    #         Whether to trim overlapping wavelengths between bands/channels in CRETA cubes
    #     keep_next : bool, default: False
    #         If trim=True, whether to keep longer wavelength data in overlaps
    #     z : float, optional
    #         Source redshift. If None, assumes z=0 (rest-frame)

    #     Notes
    #     -----
    #     CRETA cubes retain band/channel information which can result in duplicate
    #     wavelengths. The trim parameter removes these overlaps.
    #     """

    #     # file_name [str]: Name of the cube to read
    #     # file_dir [str]: Folder where the data are
    #     # extract [str]: In case of ingesting a CRETA-produced cube, read the column that have the spectra stitched
    #     # trim [bool]: CRETA cubes retain info on the bands/channels, which is used to trim the spectra and avoid wavelength duplications (default: True)
    #     # keep_next [bool]: The trimming process keeps the shortest wavelength data in overlapping bands/channels (default: False)
    #     # z (float): Redshift of the source (default: 0.)

    #     if file_dir == "extractions/":
    #         file_dir = os.path.join("./", file_dir)

    #     try:
    #         cube = cafeio.read_cretacube(
    #             os.path.join(file_dir, file_name), flux_key, flux_unc_key
    #         )
    #     except Exception as exc:
    #         raise IOError("Could not open fits file") from exc
    #     else:
    #         if cube.cube["FLUX"].header["CUNIT3"] != "um":
    #             raise ValueError("The cube wavelength units are not micron")

    #     self.file_name = file_name  # cube.cube.filename().split('/')[-1]
    #     self.result_file_name = "p".join(
    #         self.file_name.split(".")[0:-1]
    #     )  # Substitute dots by "p"'s to avoid confusion with file type
    #     self.extract = flux_key

    #     # Remove the overlapping wavelengths between the spectral modules
    #     val_inds = (
    #         trim_overlapping(cube.bandnames, keep_next)
    #         if trim == True
    #         else np.full(len(cube.waves), True)
    #     )
    #     waves = cube.waves[val_inds]
    #     fluxes = cube.fluxes[val_inds, :, :]
    #     flux_uncs = cube.flux_uncs[val_inds, :, :]
    #     masks = cube.masks[val_inds, :, :]
    #     bandnames = cube.bandnames[val_inds]
    #     header = cube.header

    #     # Warning if z=0
    #     if z == 0.0:
    #         print(
    #             "WARNING: No redshfit provided. Assuming object is already in rest-frame (z=0)."
    #         )

    #     self.z = z
    #     self.waves = waves / (1 + z)
    #     self.fluxes = fluxes / (1 + z)
    #     self.flux_uncs = flux_uncs / (1 + z)
    #     self.masks = masks
    #     self.bandnames = bandnames
    #     self.header = header
    #     self.nx, self.ny, self.nz = cube.nx, cube.ny, cube.nz
    #     self.cube = cube

    # def input_param(
    #     self,
    #     inparfile,
    #     optfile,
    #     init_parcube=None,
    #     cont_profs=None,
    # ):
    #     """
    #     Function to read in the init parameters used in the fit of the first spaxel.
    #     """
    #     # Initialize parent class input_param
    #     super().input_param(
    #         inparfile, optfile, init_parcube=init_parcube, cont_profs=cont_profs
    #     )

    # def fit_cube(
    #     self,
    #     output_path=None,
    #     init_parcube=None,
    #     cont_profs=None,
    #     pattern="snr",
    # ):
    #     """
    #     Main function setting up the parameters and profiles for cafe_grinder()
    #     """

    #     cube = self.cube

    #     # Get the fitting sequence
    #     snr_image = np.nansum(
    #         self.fluxes[10:20, :, :], axis=0
    #     )  # Temp: 10:20 is a hardcoded value. Need to be fixed
    #     ind_seq, ref_ind_seq = get_fit_sequence(snr_image, sorting_seq=pattern)
    #     print(
    #         f"Highest SNR spaxel is: {np.flip((ind_seq[0][0], ind_seq[1][0]))}"
    #     )

    #     # def fit_spec(
    #     #     self,
    #     #     output_path=None,
    #     #     # init_parcube=False, # This has been moved to the input_param function
    #     #     cont_profs=None,
    #     # ):

    #     # Fit spectrum
    #     """
    #     self.fit_spec(
    #         output_path=output_path,
    #         # init_parcube=init_parcube,
    #         cont_profs=cont_profs,
    #     )
    #     """

    #     # Convert the highest SNR spaxel to a spectrum1D
    #     wave, flux, flux_unc, bandname, mask = self._mask_spec(
    #         x=ind_seq[1][0], y=ind_seq[0][0]
    #     )
    #     spec = Spectrum1D(
    #         spectral_axis=wave * u.micron,
    #         flux=flux * u.Jy,
    #         uncertainty=StdDevUncertainty(flux_unc),
    #         redshift=self.z,
    #     )

    #     # self.inpar_fns = np.full((self.ny, self.nx), '')
    #     # self.inpar_fns[ind_seq[0][0], ind_seq[1][0]] = inparfile  # (y,x)

    #     # Initialize CAFE param generator for the highest SNR spaxel
    #     print(
    #         "Generating initial/full parameter object with all potential lines"
    #     )
    #     param_gen = CAFE_param_generator(
    #         spec, self.inpars, self.inopts, cafe_path=self.cafe_dir
    #     )
    #     # These are keywords used by deeper layers of cafe
    #     _, outPath = cafeio.init_paths(
    #         self.inopts,
    #         cafe_path=self.cafe_dir,
    #         file_name=self.result_file_name,
    #         output_path=output_path,
    #     )
    #     self.outPath = outPath

    #     # Make parameter object with all features available
    #     print(
    #         "Generating parameter cube using the initial/full parameter object"
    #     )
    #     all_params = param_gen.make_parobj(get_all=True)
    #     # Parcube is initialized with all possible parameters
    #     # Then only the ones fitted will be injected in the appropiate keys
    #     cube_gen = CAFE_cube_generator(self)
    #     parcube = cube_gen.make_parcube(all_params)

    #     ## Initiate CAFE profile loader
    #     # print('Generating continuum profiles')
    #     # prof_gen = CAFE_prof_generator(spec, inparfile, optfile, None, cafe_path=self.cafe_dir)
    #     #
    #     # if cont_profs is None: # Use default run option file
    #     #    start = time.time()
    #     #    self.cont_profs = prof_gen.make_cont_profs()
    #     #    end = time.time()
    #     #    print(np.round(end-start,2), 'seconds to make continnum profiles')

    #     # ### Create logfile
    #     # if not self.inopts['OUTPUT FILE OPTIONS']['OVERWRITE']:
    #     #     logFile = open(outpath+obj+'.log', 'a')
    #     # else:
    #     #     logFile = open(outpath+obj+'.log', 'w+')
    #     # self.inpars['METADATA']['OUTDIR'] = outpath
    #     # self.inpars['METADATA']['LOGFILE'] = outpath+obj+'.log'
    #     # ### FIXME - RA/DEC/Spaxel info should go here, once we have a spectrum format that uses it

    #     start_cube = time.time()
    #     spax = 1
    #     for snr_ind in zip(ind_seq[0], ind_seq[1]):  # (y,x)

    #         wave, flux, flux_unc, bandname, mask = self._mask_spec(
    #             x=snr_ind[1], y=snr_ind[0]
    #         )
    #         weight = 1.0 / flux_unc**2

    #         if np.isnan(flux).any():
    #             # ipdb.set_trace()
    #             raise ValueError(
    #                 "Some of the flux values in the spectrum are NaN, which should not happen"
    #             )

    #         spec = Spectrum1D(
    #             spectral_axis=wave * u.micron,
    #             flux=flux * u.Jy,
    #             uncertainty=StdDevUncertainty(flux_unc),
    #             redshift=self.z,
    #         )
    #         spec_dict = {
    #             "wave": wave,
    #             "flux": flux,
    #             "flux_unc": flux_unc,
    #             "weight": weight,
    #         }

    #         print(
    #             "##############################################################################################################"
    #         )
    #         print(
    #             "Regenerating parameter object for current spaxel:",
    #             np.flip(snr_ind),
    #             "(",
    #             spax,
    #             "/",
    #             len(ind_seq[0]),
    #             ")",
    #         )
    #         param_gen = CAFE_param_generator(
    #             spec, self.inpars, self.inopts, cafe_path=self.cafe_dir
    #         )
    #         init_params = param_gen.make_parobj()

    #         print("Regenerating continuum profiles")
    #         prof_gen = CAFE_prof_generator(
    #             spec, self.inpars, self.inopts, None, cafe_path=self.cafe_dir
    #         )
    #         self.cont_profs = prof_gen.make_cont_profs()

    #         # if spax == 1:
    #         #    if 'AGN' in inparfile: inparfile.replace('AGN', 'SB')

    #         # If not the first spaxel
    #         if snr_ind != (ind_seq[0][0], ind_seq[1][0]):
    #             print(
    #                 "Current spaxel",
    #                 np.flip(snr_ind),
    #                 "will be initialized with results from spaxel",
    #                 np.flip(
    #                     (
    #                         ref_ind_seq[0, snr_ind[0], snr_ind[1]],
    #                         ref_ind_seq[1, snr_ind[0], snr_ind[1]],
    #                     )
    #                 ),
    #             )  # ,
    #             #      'and set to a SB inppar file')
    #             #
    #             # self.inpar_fns[snr_ind[0], snr_ind[1]] = inparfile  # (y,x)

    #             # We inject the common params in the parameter cube of the reference (fitted) spaxel
    #             # assigned for the initialization of the current spaxel to the current spaxel params
    #             cube_params = parcube2parobj(
    #                 parcube,
    #                 ref_ind_seq[1, snr_ind[0], snr_ind[1]],
    #                 ref_ind_seq[0, snr_ind[0], snr_ind[1]],
    #                 init_parobj=init_params,
    #             )  # indexation is (x=1,y=0)

    #             # The params file is regenerated but with the VARY, LIMS and ARG reset based on the new VALUES injected
    #             params = param_gen.make_parobj(
    #                 parobj_update=cube_params,
    #                 get_all=True,
    #                 init_parobj=init_params,
    #             )

    #         else:
    #             if init_parcube is None:
    #                 print(
    #                     "The params will be set to the parameters of the parcube provided for initialization"
    #                 )
    #                 cube_params = parcube2parobj(
    #                     init_parcube, init_parobj=init_params
    #                 )
    #                 params = param_gen.make_parobj(
    #                     parobj_update=cube_params,
    #                     get_all=True,
    #                     init_parobj=init_params,
    #                 )
    #             else:
    #                 params = init_params

    #         unfixed_params = [
    #             True if params[par].vary == True else False
    #             for par in params.keys()
    #         ]
    #         print(
    #             "Fitting",
    #             unfixed_params.count(True),
    #             "unfixed parameters, out of the",
    #             len(params),
    #             "defined in the parameter object",
    #         )
    #         # Fit the spectrum
    #         result = cafe_grinder(self, params, spec_dict, None)
    #         print(
    #             "The VGRAD of the current spaxel is:",
    #             result.params["VGRAD"].value,
    #             "[km/s]",
    #         )

    #         # Inject the result into the parameter cube
    #         parcube = parobj2parcube(
    #             result.params, parcube, snr_ind[1], snr_ind[0]
    #         )  # indexation is (x,y)

    #         spax += 1

    #     self.parcube = parcube

    #     end_cube = time.time()
    #     print(
    #         "Cube fitted in",
    #         np.round(end_cube - start_cube, 2) / 60.0,
    #         "minutes",
    #     )

    #     # Save parcube to disk
    #     self.parcube_dir = outPath
    #     self.parcube_name = self.result_file_name + "_parcube"
    #     print(
    #         "Saving parameters in cube to disk:",
    #         self.parcube_dir + self.parcube_name + ".fits",
    #     )
    #     parcube.writeto(
    #         self.parcube_dir + self.parcube_name + ".fits", overwrite=True
    #     )

    #     # Write .ini file as paramfile
    #     print(
    #         "Saving init file of the central spaxel to disk:",
    #         self.parcube_dir + self.result_file_name + "_fitpars.ini",
    #     )
    #     cafeio.write_inifile(
    #         parcube2parobj(parcube, x=ind_seq[1][0], y=ind_seq[0][0]),
    #         self.inpars,
    #         self.parcube_dir + self.result_file_name + "_fitpars.ini",
    #     )

    #     ## Make and save tables (IMPROVE FOR CUBES: NOW WILL ONLY WRITE DOWN THE CENTRAL SPAXEL)
    #     ## Save .asdf to disk
    #     # print('Saving components of the central spaxel in asdf to disk:',self.parcube_dir+self.result_file_name+'_cafefit.asdf')
    #     # cafeio.save_asdf(self, x=ind_seq[1][0], y=ind_seq[0][0], file_name=self.parcube_dir+self.result_file_name+'_cafefit')

    #     return self

    #     # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def make_map(self, parname, map_dir=""):

        if map_dir == "":
            if hasattr(self, "outPath") is True:
                self.map_dir = self.outPath
            else:
                raise ValueError(
                    'A directory where to store the map must be provided with the keyword map_dir=""'
                )
        else:
            self.map_dir = map_dir

        cube_gen = CAFE_cube_generator(self)
        self.parmap = cube_gen.make_map(parname)

        # Feature map
        self.map_name = self.result_file_name + "_" + parname + "_map"
        print("Saving map to disk:", self.map_dir + self.map_name + ".fits")
        self.parmap.writeto(
            self.map_dir + self.map_name + ".fits", overwrite=True
        )

    def plot_cube_ini(
        self,
        x,
        y,
        inparfile,
        optfile,
        cont_profs=None,
    ):
        """
        Plot the SED generated by the inital parameters
        """

        wave, flux, flux_unc, bandname, mask = self._mask_spec(x, y)

        if np.isnan(flux).any() == True:
            raise ValueError("Requested spaxel has NaN values")

        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            flux=flux * u.Jy,
            uncertainty=StdDevUncertainty(flux_unc),
            redshift=self.z,
        )
        spec_dict = {"wave": wave, "flux": flux, "flux_unc": flux_unc}

        # Plot features based on inital intput parameters
        # -----------------------------------------------

        # Initiate CAFE param generator and make parameter file
        print(
            "Generating continuum profiles for guess model from the .ini file"
        )
        param_gen = CAFE_param_generator(
            spec, self.inpars, self.inopts
        )
        params = param_gen.make_parobj()

        # Initiate CAFE profile loader and make cont_profs
        prof_gen = CAFE_prof_generator(
            spec, self.inpars, self.inopts, None
        )
        cont_profs = prof_gen.make_cont_profs()

        # Scale continuum profiles with parameters and get spectra
        flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, _ = (
            get_model_fluxes(params, wave, cont_profs, verbose_output=True)
        )

        # Get spectrum out of the feature parameters
        gauss, drude, gauss_opc = get_feat_pars(params, apply_vgrad2waves=True)

        cafefig = cafeplot(
            spec_dict, None, CompFluxes, gauss, drude, pahext=extComps["extPAH"]
        )

    def plot_cube_fit(self, x, y, inparfile, optfile, savefig=None):
        """
        Plot the spectrum itself. If params already exists, plot the fitted results as well.
        """
        if hasattr(self, "parcube") is False:
            raise ValueError("The spectrum is not fitted yet")
        else:
            params = parcube2parobj(self.parcube, x, y)

            wave, flux, flux_unc, bandname, mask = self._mask_spec(x, y)
            spec = Spectrum1D(
                spectral_axis=wave * u.micron,
                flux=flux * u.Jy,
                uncertainty=StdDevUncertainty(flux_unc),
                redshift=self.z,
            )
            spec_dict = {"wave": wave, "flux": flux, "flux_unc": flux_unc}

            prof_gen = CAFE_prof_generator(
                spec, self.inpars, self.inopts, None
            )
            cont_profs = prof_gen.make_cont_profs()

            flux, CompFluxes, CompFluxes_0, extComps, e0, tau0, vgrad = (
                get_model_fluxes(params, wave, cont_profs, verbose_output=True)
            )

            gauss, drude, gauss_opc = get_feat_pars(
                params, apply_vgrad2waves=True
            )  # params consisting all the fitted parameters

            # sedfig, chiSqrFin = sedplot(wave, flux, flux_unc, CompFluxes, weights=weight, npars=result.nvarys)
            cafefig = cafeplot(
                spec_dict,
                None,
                CompFluxes,
                gauss,
                drude,
                vgrad=vgrad,
                pahext=extComps["extPAH"],
            )

            # figs = [sedfig, cafefig]

            # with PdfPages(outpath+obj+'_fitplots'+tstamp+'.pdf') as pdf:
            #     for fig in figs:
            #         plt.figure(fig.number)
            #         pdf.savefig(bbox_inches='tight')

            if savefig is not None:
                cafefig.savefig(savefig)

    def plot_spec(self, x, y, savefig=None):

        wave, flux, flux_unc, bandname, mask = self._mask_spec(x, y)
        spec = Spectrum1D(
            spectral_axis=wave * u.micron,
            flux=flux * u.Jy,
            uncertainty=StdDevUncertainty(flux_unc),
            redshift=self.z,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            spec.spectral_axis, spec.flux, linewidth=1, color="k", alpha=0.8
        )
        ax.scatter(
            spec.spectral_axis, spec.flux, marker="o", s=8, color="k", alpha=0.7
        )
        ax.errorbar(
            spec.spectral_axis.value,
            spec.flux.value,
            yerr=spec.uncertainty.quantity.value,
            fmt="none",
            ecolor="gray",
            alpha=0.4,
        )

        ax.set_xlabel(
            "Wavelength (" + spec.spectral_axis.unit.to_string() + ")"
        )
        ax.set_ylabel("Flux (" + spec.flux.unit.to_string() + ")")
        ax.set_xscale("log")
        ax.set_yscale("log")

        if savefig is not None:
            fig.savefig(savefig)

        return ax

        # TBD


# ================================================================


def plot_cafefit(asdf_fn):
    """Recover the CAFE plot based on the input asdf file
    INPUT:
        asdf_fn: the asdf file that store the CAFE fitted parameters
    OUTPUT:
        A mpl axis object that can be modified for making the figure
    """
    af = asdf.open(asdf_fn)

    wave = np.asarray(af.tree["cafefit"]["obsspec"]["wave"])
    flux = np.asarray(af["cafefit"]["obsspec"]["flux"])
    flux_unc = np.asarray(af["cafefit"]["obsspec"]["flux_unc"])

    comps = af["cafefit"]["CompFluxes"]
    extPAH = af["cafefit"]["extComps"]["extPAH"]
    g = af["cafefit"]["gauss"]
    d = af["cafefit"]["drude"]

    gauss = [g["wave"], g["gamma"], g["peak"]]
    drude = [d["wave"], d["gamma"], d["peak"]]

    spec_dict = {"wave": wave, "flux": flux, "flux_unc": flux_unc}

    # Assuming there is no phot input.
    # TODO: include phot_dict as input.
    (cafefig, ax1, ax2) = cafeplot(
        spec_dict, None, comps, gauss, drude, pahext=extPAH
    )


def _read_table_data(file_path, read_columns, flux_unc, is_SED):
    """Helper function to read data from text/csv files."""
    from astropy.table import Table

    # If specific columns are requested
    if read_columns is not None:
        tab = Table.read(
            file_path,
            format="ascii.basic",
            data_start=0,
            comment="#",
        )
        tab_col_names = read_columns.copy()
        import string

        for i in range(len(tab[0]) - len(read_columns)):
            tab_col_names.append(string.ascii_lowercase[i])
        tab = Table.read(
            file_path,
            format="ascii.basic",
            names=tab_col_names,
            data_start=0,
            comment="#",
        )
        if flux_unc is not None:
            tab["flux_unc"] = tab["flux"] * flux_unc
        return tab

    # Handle different file types
    _, extension = os.path.splitext(file_path)

    if extension in [".dat", ".txt"]:
        names = (
            ["name", "wave", "flux", "flux_unc", "width"]
            if is_SED
            else ["wave", "flux", "flux_unc"]
        )
        return Table.read(
            file_path,
            format="ascii.basic",
            names=names,
            data_start=0,
            comment="#",
        )

    if extension == ".csv":
        df = pd.read_csv(file_path, skiprows=31)
        expected_cols = [
            "Wave",
            "Band_name",
            "Flux_ap",
            "Err_ap",
            "R_ap",
            "Flux_ap_st",
            "Err_ap_st",
            "DQ",
        ]

        if sum(df.columns == expected_cols) == 8:
            out_df = df[["Wave", "Flux_ap_st", "Err_ap_st"]]
            tab = Table.from_pandas(out_df)
            tab.rename_column("Wave", "wave")
            tab.rename_column("Flux_ap_st", "flux")
            tab.rename_column("Err_ap_st", "flux_unc")
            return tab
        raise IOError("Only CAFE produced csv files can be ingested.")

    raise IOError(f"Unsupported file extension: {extension}")


def _create_spectrum_from_table(tab):
    """Helper function to create Spectrum1D from table data."""
    spec = Spectrum1D(
        spectral_axis=tab["wave"] * u.micron,
        flux=tab["flux"] * u.Jy,
        uncertainty=StdDevUncertainty(tab["flux_unc"]),
    )
    spec.mask = np.full(len(spec.flux), 0)
    spec.meta = {"bandname": [None]}
    return spec
