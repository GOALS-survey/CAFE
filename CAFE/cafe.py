import numpy as np 
import copy
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from specutils import Spectrum1D, SpectrumList
from astropy.nddata import StdDevUncertainty
import lmfit as lm # https://dx.doi.org/10.5281/zenodo.11813
import time, datetime
import warnings
import astropy.units as u
from astropy import constants as const
from astropy.stats import mad_std
import pandas as pd

import cafe_io
from cafe_io import *
cafeio = cafe_io()
import cafe_lib
from cafe_lib import *
import cafe_helper
from cafe_helper import *

import asdf
from asdf import AsdfFile
from astropy.io import fits 

import astropy
import pdb, ipdb


def cafe_grinder(self, params, wave, flux, flux_unc, weight):

    ### Read in global fit settings
    ftol = self.inopts['FIT OPTIONS']['FTOL']
    nproc = self.inopts['FIT OPTIONS']['NPROC']
    
    acceptFit = False
    ### Limit the number of times the fit can be rerun to avoid infinite loop
    niter = 1
    show = False

    ini_params = copy.copy(params) # Initial parameters
    
    while acceptFit is False:
        
        start = time.time()
        print('Iteration '+str(niter)+'/'+str(self.inopts['FIT OPTIONS']['MAX_LOOPS'])+'(max):',datetime.datetime.now(),'-----------------')
        
        method = 'least_squares' #'nelder' if len(wave) >= 10000 else 'least_squares'
        fitter = lm.Minimizer(chisquare, params, fcn_args=(wave, flux, flux_unc, weight, self.cont_profs, show))
        ### Note that which fitting method is faster here is pretty uncertain, changes by target
        result = fitter.minimize(method=method, ftol=ftol) #, method='leastsq')
        #print(lm.fit_report(result))
        
        if result.success:
            ### Print to terminal and log file
            print(result.success, 'in', result.nfev, 'steps')
            end = time.time()
            print(np.round(end-start, 2), 'seconds elapsed')
            #logFile.write(str(result.success) +  ' in ' +  str(result.nfev) + ' iterations\n')
            #logFile.write(str(np.round(end-start1,2)) + ' seconds elapsed\n')
            
            old_params = copy.copy(params) # Parameters of the previous iteration
            fit_params = copy.copy(result.params) # parameters of the current iteration that will not be changed
            params = result.params # Both params AND result.params will be modified by check_fit_parameters

            # Do checks on parameters and rerun fit if no errors are returned
            if self.inopts['FIT OPTIONS']['FIT_CHK'] and niter < self.inopts['FIT OPTIONS']['MAX_LOOPS']:
                acceptFit = check_fit_pars(self, wave, flux_unc, fit_params, params, old_params, result.errorbars)
                
                if acceptFit is False:
                    print('Rerunning fit')
                    #logFile.write('Rerunning fit\n')
                    niter+=1
                    
                    # Perturbe the values of parameters that are scaling parameters
                    for par in params.keys():
                        if params[par].vary == True:
                            parnames = par.split('_')
                            if parnames[-1] == 'Peak' \
                               or parnames[-1] == 'FLX' or parnames[-1] == 'TMP' \
                               or parnames[-1] == 'TAU' or parnames[-1] == 'RAT':
                                params[par].value *= (1. + 0.01)
            else:
                #logFile.write('Hit maximum number of refitting loops\n')
                print('Hit maximum number of refitting loops. Continuing to next spaxel (if any left).')
                acceptFit = True
                
    #print(lm.fit_report(result))
    if niter < self.inopts['FIT OPTIONS']['MAX_LOOPS']: print('Successful fit ----------------------------------------------------')

    return result




class cubemod:

    def __init__(self):

        pass


    def read_parcube_file(self, file_name, file_dir='./output/'):

        parcube = fits.open(file_dir+file_name)
        parcube.info()
        self.parcube = parcube
        #parcube.close()



    def read_cube(self, file_name, file_dir='./input/data/', extract='Flux_st', trim=True, keep_next=False, z=None):

        try:
            cube = cafeio.read_cretacube(file_dir+file_name, extract)
        except:
            raise IOError('Could not open fits file')
        else:
            if cube.cube['FLUX'].header['CUNIT3'] != 'um':
                raise ValueError("The cube wavelength units are not micron")
        
        
        self.file_name = file_name #cube.cube.filename().split('/')[-1]
        self.extract = extract
        
        # Remove the overlapping wavelengths between the spectral modules
        val_inds = trim_overlapping(cube.bandnames, keep_next) if trim == True else np.full(len(cube.waves),True)
        waves = cube.waves[val_inds]
        fluxes = cube.fluxes[val_inds,:,:]
        flux_uncs = cube.flux_uncs[val_inds,:,:]
        masks = cube.masks[val_inds,:,:]
        bandnames = cube.bandnames[val_inds]
        header = cube.header
        
        # Warning if z=0
        if z == 0.0: print('WARNING: No redshfit provided. Assuming object is already in rest-frame (z=0).')        
        
        self.z = z
        self.waves = waves / (1+z)
        self.fluxes = fluxes / (1+z)
        self.flux_uncs = flux_uncs / (1+z)
        self.masks = masks
        self.bandnames = bandnames
        self.header = header
        self.nx, self.ny, self.nz = cube.nx, cube.ny, cube.nz
        self.cube = cube
        
    
    
    def get_fit_sequence(self):
        
        # Create SNR image based on the first ten channels of the cube
        snr_image = np.nansum(self.fluxes[0:10,:,:], axis=0) #/ np.sqrt(np.sum(self.flux_uncs[0:11,:,:]**2, axis=0))
        
        # Store the 2D indices of the SNR image ranked from max to min SNR
        snr_ind_seq = np.unravel_index(np.flip(np.argsort(snr_image, axis=None)), snr_image.shape)
        
        # Initialize dictionary containing a matrix that will store, in each pixel, the indices of the spaxel
        # to be used for the parameter initialization, as well as a tracking image used to know what spaxels
        # have been already "fitted" in previous steps
        param_ind_seq = {'parini_spx_ind': np.full((2,)+snr_image.shape, np.nan), 'track': np.full(snr_image.shape, False)}
        
        x, y = np.meshgrid(np.arange(snr_image.shape[1]), np.arange(snr_image.shape[0]))
        
        # For each spaxel
        for snr_ind in zip(snr_ind_seq[0], snr_ind_seq[1]): # (y,x)
            # First spaxel
            if snr_ind == (snr_ind_seq[0][0], snr_ind_seq[1][0]):
                param_ind_seq['parini_spx_ind'][:,snr_ind[0],snr_ind[1]] = snr_ind
                param_ind_seq['track'][snr_ind] = True
            else:
                # Mask with closest neighbors
                neighbors = np.sqrt((x-snr_ind[1])**2 + (y-snr_ind[0])**2) <= 1.5
                # Mask with closest neighbors that have been already fitted
                fitted_neighbor_inds = np.logical_and(param_ind_seq['track'], neighbors)
                # If there is any
                if fitted_neighbor_inds.any():
                    # Chose the one with the highest SNR
                    max_snr_fitted_neighbor_ind = np.where(snr_image == np.nanmax(snr_image[fitted_neighbor_inds]))
                else:
                    # Chose the first, highest SNR spaxel in the image
                    max_snr_fitted_neighbor_ind = (np.array([snr_ind_seq[0][0]]), np.array([snr_ind_seq[1][0]]))
                    ## Assign its own index
                    #max_snr_fitted_neighbor_ind = (np.array([snr_ind[0]]), np.array([snr_ind[1]]))
                # Assign the indices to the sequence
                param_ind_seq['parini_spx_ind'][:,snr_ind[0],snr_ind[1]] = np.concatenate(max_snr_fitted_neighbor_ind)
                param_ind_seq['track'][snr_ind] = True
                
        # Returns the indices of the SNR sorted image, and the initiaziation indices of each spaxel
        return snr_ind_seq, param_ind_seq['parini_spx_ind'].astype(int) # (y,x)
    
    
    
    def fit_cube(self,
                 inparfile,
                 optfile,
                 fit_pattern = 'default',
                 cont_profs = None,
                 ):
        """
        Output result from lmfit
        """
        
        cube = self.cube # cube is in observed-frame and not trimmed
        #cube_flux = self.fluxes # (z,y,x) = (lambda,dec,RA) in rest-frame
        #cube_flux_unc = self.flux_uncs # (z,y,x) = (lambda,dec,RA) in rest-frame
        
        print('Generating fitting sequence')
        snr_ind_seq, param_ind_seq = self.get_fit_sequence()
        print('Highest SNR spaxel is:',snr_ind_seq[0][0],snr_ind_seq[1][0])
        
        # Convert the highest SNR spaxel to a spectrum1D
        wave, flux, flux_unc, bandname, mask = mask_spec(self,snr_ind_seq[1][0],snr_ind_seq[0][0])
        spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
        
        # Initialize CAFE param generator for the highest SNR spaxel
        print('Generating initial/full parameter object with all potential lines')        
        param_gen = CAFE_param_generator(spec, inparfile, optfile)
        self.inpars = param_gen.inpars
        self.inopts = param_gen.inopts
        tablePath, outPath = cafeio.init_paths(param_gen.inopts, ''.join(self.file_name.split('.')[0:-1]))

        # Make parameter object with all features available
        init_params = param_gen.make_parobj(get_all=True)
        self.init_params = init_params
        #params = copy.copy(init_params)
        

        ## Initiate CAFE profile loader
        #print('Generating continuum profiles')
        #prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
        #
        #if cont_profs is None: # Use default run option file
        #    start = time.time()
        #    self.cont_profs = prof_gen.make_cont_profs()
        #    end = time.time()
        #    print(np.round(end-start,2), 'seconds to make continnum profiles')
        
        # Parcube is initialized with all possible parameters
        # Then only the ones fitted will be injected in the appropiate keys
        print('Generating parameter cube with initial/full parameter object')
        parcube_gen = CAFE_parcube_generator(self, init_params, inparfile, optfile)
        parcube = parcube_gen.make_parcube()
        
        # ### Create logfile
        # if not self.inopts['OUTPUT FILE OPTIONS']['OVERWRITE']:
        #     logFile = open(outpath+obj+'.log', 'a')
        # else:
        #     logFile = open(outpath+obj+'.log', 'w+')        
        # self.inpars['METADATA']['OUTDIR'] = outpath
        # self.inpars['METADATA']['LOGFILE'] = outpath+obj+'.log'
        # ### FIXME - RA/DEC/Spaxel info should go here, once we have a spectrum format that uses it
        
        #parcube = parcube_gen.parobj2parcube(init_params, parcube, snr_ind_seq[1][0], snr_ind_seq[0][0]) # indexation is (x,y)
        
        spax = 0
        for snr_ind in zip(snr_ind_seq[0], snr_ind_seq[1]): # (y,x)

            wave, flux, flux_unc, bandname, mask = mask_spec(self, x=snr_ind[1], y=snr_ind[0])
            weight = 1./flux_unc**2
            
            spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)

            print('Regenerating continuum profiles')
            prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
            self.cont_profs = prof_gen.make_cont_profs()
            
            print('Regenerating parameter object for current spaxel',np.flip(snr_ind))        
            param_gen = CAFE_param_generator(spec, inparfile, optfile)
            params = param_gen.make_parobj()
            
            if snr_ind != (snr_ind_seq[0][0], snr_ind_seq[1][0]):
                print('Current spaxel',np.flip(snr_ind),'will be initialized with results from spaxel',
                      np.flip((param_ind_seq[0,snr_ind[0],snr_ind[1]], param_ind_seq[1,snr_ind[0],snr_ind[1]])))
                
                # We inject the common params in the parameter cube of the reference (fitted) spaxel
                # assigned for the initialization of the current spaxel to the current spaxel params
                cube_params = parcube2parobj(parcube, 
                                             param_ind_seq[1,snr_ind[0],snr_ind[1]],
                                             param_ind_seq[0,snr_ind[0],snr_ind[1]],
                                             parobj=params) # indexation is (x,y)
                
                # We regenerate the params file but with the new VARY, LIMS and ARG, based on the new VALUES injected
                params = param_gen.make_parobj(get_all=True, parobj_update=cube_params)
                
            # Fit the spectrum
            result = cafe_grinder(self, params, wave, flux, flux_unc, weight)

            # Inject the result into the parameter cube
            parcube = parobj2parcube(result.params, parcube, snr_ind[1], snr_ind[0]) # indexation is (x,y)
            
        self.parcube = parcube
        
        # Save to disk
        source_fn = ''.join(self.file_name.split('.')[0:-1])
        parcube.writeto(outPath+source_fn+'_parcube.fits', overwrite=True)
        
        return self



    def plot_cube_ini(self, x, y,
                      inparfile, 
                      optfile, 
                      cont_profs=None):
        """
        Plot the SED generated by the inital parameters
        """

        wave, flux, flux_unc, bandname, mask = mask_spec(self,x,y)
        spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
        
        # Plot features based on inital intput parameters
        # -----------------------------------------------
        
        # Initiate CAFE param generator and make parameter file
        print('Generating continuum profiles for guess model')
        param_gen = CAFE_param_generator(spec, inparfile, optfile)
        params = param_gen.make_parobj()

        # Initiate CAFE profile loader and make cont_profs
        prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
        cont_profs = prof_gen.make_cont_profs()

        # Scale continuum profiles with parameters and get spectra
        CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(params, wave, cont_profs, comps=True)
        
        # Get spectrum out of the feature parameters
        gauss, drude, gauss_opc = get_feat_pars(params)

        cafefig = cafeplot(wave, flux, flux_unc, CompFluxes, gauss, drude, plot_drude=True, pahext=extComps['extPAH'])
        


    def plot_cube_fit(self, x, y,
                      inparfile, 
                      optfile,
                      savefig=None):
        """
        Plot the spectrum itself. If fitPars already exists, plot the fitted results as well.
        """
        if hasattr(self, 'parcube') is False:
            raise ValueError('The spectrum is not fitted yet.')
        else:
            fitPars = parcube2parobj(self.parcube, x, y)

            wave, flux, flux_unc, bandname, mask = mask_spec(self, x, y)
            spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)

            prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
            cont_profs = prof_gen.make_cont_profs()

            CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(fitPars, wave, cont_profs, comps=True)

            gauss, drude, gauss_opc = get_feat_pars(fitPars)  # fitPars consisting all the fitted parameters

            #sedfig, chiSqrFin = sedplot(wave, flux, flux_unc, CompFluxes, weights=weight, npars=result.nvarys)
            cafefig = cafeplot(wave, flux, flux_unc, CompFluxes, gauss, drude, plot_drude=True, pahext=extComps['extPAH'])

            # figs = [sedfig, cafefig]
            
            # with PdfPages(outpath+obj+'_fitplots'+tstamp+'.pdf') as pdf:
            #     for fig in figs:
            #         plt.figure(fig.number)
            #         pdf.savefig(bbox_inches='tight')

            if savefig is not None:
                cafefig.savefig(savefig)

            return cafefig



    def make_map(self,feat_name,
                 savefig=None):

        if hasattr(self, 'parcube') is False:
            raise ValueError('The spectrum is not fitted yet.')
        else:
            fitPars = parcube2parobj(self.parcube, x, y)

        #TBD


# ================================================================
class specmod:
    """ 
    CAFE model. When initialized, it contains the spec
    """
    def __init__(self):
                
        pass


    def read_parcube_file(self, file_name, file_dir='./output/'):

        parcube = fits.open(file_dir+file_name)
        parcube.info()
        self.parcube = parcube
        #parcube.close()


    def read_spec(self, file_name, file_dir='./input/data/', extract='Flux_st', trim=True, keep_next=False, z=0.):

        print('Load data:',file_dir+file_name)
        try:
            cube = cafeio.read_cretacube(file_dir+file_name, extract)
        except:
            hdu = fits.PrimaryHDU()
            dummy = fits.ImageHDU(np.full(1,np.nan), name='Flux')
            dummy.header['EXTNAME'] = 'FLUX'
            hdulist = fits.HDUList([hdu, dummy])
            hdulist.writeto('./dummy.fits', overwrite=True)
            cube = fits.open('./dummy.fits')
            try:
                spec = cafeio.customFITSReader(file_dir+file_name, extract) # Returns a spectrum1D
            except:
                try:
                    from astropy.table import Table
                    tab = Table.read(file_dir+file_name, format='ascii.basic', names=['wave', 'flux', 'flux_unc'])
                except:
                    raise IOError('The file is not a valid .txt (column-based) or .fits (CRETA output) file.')
                else:
                    spec = Spectrum1D(spectral_axis=tab['wave']*u.micron, flux=tab['flux']*u.Jy, uncertainty=StdDevUncertainty(tab['flux_unc']), redshift=z)
                    spec.mask = np.full(len(spec.flux), 0)
                    spec.meta = {'bandname':[None]}

            else:
                cube.bandnames = spec.meta['band_name'][0]
            #finally:
            if spec.spectral_axis.unit != 'micron':
                raise ValueError("Make sure wavelength is in micron")                            
            cube.waves = spec.spectral_axis.value
            cube.fluxes = spec.flux.value
            cube.flux_uncs = spec.uncertainty.quantity.value
            cube.masks = spec.mask
            cube.nx = 1
            cube.ny = 1
            cube.nz = spec.flux.shape
            cube.header = dummy.header #spec.meta
            cube.bandnames = np.full(len(cube.waves),'UNKNOWN')
            trim = False
            cube.close()
        else:
            cube.header = cube.cube['FLUX'].header
            if cube.cube['FLUX'].header['CUNIT3'] != 'um':
                raise ValueError("The cube wavelength units are not micron")


        self.file_name = file_name #cube.cube.filename().split('/')[-1]
        self.extract = extract
        
        # Remove the overlapping wavelengths between the spectral modules
        val_inds = trim_overlapping(cube.bandnames, keep_next) if trim == True else np.full(len(cube.waves),True)
        waves = cube.waves[val_inds]
        fluxes = cube.fluxes[val_inds]
        flux_uncs = cube.flux_uncs[val_inds]
        masks = cube.masks[val_inds]
        bandnames = cube.bandnames[val_inds]
        header = cube.header
        
        # Warning if z=0
        if z == 0.0: print('WARNING: No redshift provided. Assuming object is already in rest-frame (z=0).')
        
        self.z = z
        self.waves = waves / (1+z)
        self.fluxes = fluxes / (1+z)
        self.flux_uncs = flux_uncs / (1+z)   
        self.masks = masks
        self.bandnames = bandnames
        self.header = header
        self.nx, self.ny, self.nz = cube.nx, cube.ny, cube.nz
        self.cube = cube



    def fit_spec(self, 
                 inparfile, 
                 optfile, 
                 cont_profs=None,
                 ):
        """
        Output result from lmfit
        """

        wave, flux, flux_unc, bandname, mask = mask_spec(self)
        weight = 1./flux_unc**2
        spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
        

        # Initiate CAFE param generator
        print('Generating parameter object')        
        param_gen = CAFE_param_generator(spec, inparfile, optfile)
        self.inpars = param_gen.inpars
        self.inopts = param_gen.inopts
        tablePath, outPath = cafeio.init_paths(param_gen.inopts, ''.join(self.file_name.split('.')[0:-1]))

        init_params = param_gen.make_parobj()
        self.init_params = init_params
        params = copy.copy(init_params)
        
        # Initiate CAFE profile loader
        print('Generating continuum profiles')
        prof_gen = CAFE_prof_generator(spec, inparfile, optfile)

        if cont_profs is None: # Use default run option file
            start = time.time()
            self.cont_profs = prof_gen.make_cont_profs()
            end = time.time()
            print(np.round(end-start,2), 'seconds to make continnum profiles')
            
        print('Generating parameter cube')
        parcube_gen = CAFE_parcube_generator(self, params, inparfile, optfile)
        parcube = parcube_gen.make_parcube()

        print('Fitting',len(params),'parameters')
        # Fit the spectrum
        result = cafe_grinder(self, params, wave, flux, flux_unc, weight)

        #print(lm.fit_report(result))
        # Inject the result into the parameter cube
        parcube = parobj2parcube(result.params, parcube, 0, 0) # indexation is (x,y)
            
        self.parcube = parcube
        
        # Save to disk
        source_fn = ''.join(self.file_name.split('.')[0:-1])
        parcube.writeto(outPath+source_fn+'_parcube.fits', overwrite=True)

        return self

    

    def plot_spec_ini(self,
                      inparfile, 
                      optfile, 
                      cont_profs=None):
        """
        Plot the SED generated by the inital parameters
        """

        wave, flux, flux_unc, bandname, mask = mask_spec(self)
        spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
 
        # Plot features based on inital intput parameters
        # -----------------------------------------------

        # Initiate CAFE param generator and make parameter file
        print('Generating continuum profiles for guess model')
        param_gen = CAFE_param_generator(spec, inparfile, optfile)
        params = param_gen.make_parobj()
        
        # Initiate CAFE profile loader and make cont_profs
        prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
        cont_profs = prof_gen.make_cont_profs()

        # Scale continuum profiles with parameters and get spectra
        CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(params, wave, cont_profs, comps=True)

        # Get spectrum out of the feature parameters
        gauss, drude, gauss_opc = get_feat_pars(params)
        
        cafefig = cafeplot(wave, flux, flux_unc, CompFluxes, gauss, drude, plot_drude=True, pahext=extComps['extPAH'])



    def plot_spec_fit(self,
                      inparfile, 
                      optfile, 
                      savefig=None):
        """
        Plot the spectrum itself. If fitPars already exists, plot the fitted results as well.
        """
        if hasattr(self, 'parcube') is False:
            raise ValueError('The spectrum is not fitted yet.')
        else:
            fitPars = parcube2parobj(self.parcube)

            wave, flux, flux_unc, bandname, mask = mask_spec(self)
            spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)

            prof_gen = CAFE_prof_generator(spec, inparfile, optfile)
            cont_profs = prof_gen.make_cont_profs()

            CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(fitPars, wave, cont_profs, comps=True)

            gauss, drude, gauss_opc = get_feat_pars(fitPars)  # fitPars consisting all the fitted parameters
                    
            #sedfig, chiSqrFin = sedplot(wave, flux, flux_unc, CompFluxes, weights=weight, npars=result.nvarys)
            cafefig = cafeplot(wave, flux, flux_unc, CompFluxes, gauss, drude, plot_drude=True, pahext=extComps['extPAH'])

            # figs = [sedfig, cafefig]
            
            # with PdfPages(outpath+obj+'_fitplots'+tstamp+'.pdf') as pdf:
            #     for fig in figs:
            #         plt.figure(fig.number)
            #         pdf.savefig(bbox_inches='tight')

            if savefig is not None:
                cafefig.savefig(savefig)

            return cafefig



    def plot_spec(self, savefig=None):

        wave, flux, flux_unc, bandname, mask = mask_spec(self)
        spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=self.z)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(spec.spectral_axis, spec.flux, linewidth=1, color='k', alpha=0.8)
        ax.scatter(spec.spectral_axis, spec.flux, marker='o', s=8, color='k', alpha=0.7)
        ax.errorbar(spec.spectral_axis.value, spec.flux.value, yerr=spec.uncertainty.quantity.value, fmt='none', ecolor='gray', alpha=0.4)

        ax.set_xlabel('Wavelength (' + spec.spectral_axis.unit.to_string() + ')')
        ax.set_ylabel('Flux (' + spec.flux.unit.to_string() + ')')
        ax.set_xscale('log')
        ax.set_yscale('log')

        if savefig is not None:
            fig.savefig(savefig)

        return ax



    def save_result(self, asdf=True, pah_tbl=True, line_tbl=True, output_dirc=None):
        if asdf is True:

            if hasattr(self, 'parcube') is False:
                raise ValueError('The spectrum is not fitted yet.')
            else:
                fitPars = parcube2parobj(self.parcube, 0, 0)

            # Get fitted results
            CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(fitPars, wave, self.cont_profs, comps=True)

            # Narrow gauss and drude components have now an extra "redshift" from the VGRAD parameter
            gauss, drude, gauss_opc = get_feat_pars(self.fitPars)

            # Get PAH powers (intrinsic/extinguished)
            pah_power_int = drude_int_fluxes(CompFluxes['wave'], drude)
            pah_power_ext = drude_int_fluxes(CompFluxes['wave'], drude, ext=extComps['extPAH'])

            # Quick hack for output PAH and line results
            output_gauss = {'wave':gauss[0], 'width':gauss[1], 'peak':gauss[2], 'name':gauss[3], 'strength':np.zeros(len(gauss[3]))} #  Should add integrated gauss
            output_drude = {'wave':drude[0], 'width':drude[1], 'peak':drude[2], 'name':drude[3], 'strength':pah_power_int.value}

            # Make dict to save in .asdf
            obsspec = {'wave': self.wave, 'flux': self.flux, 'flux_unc': self.flux_unc}
            cafefit = {'cafefit': {'obsspec': obsspec,
                                   'fitPars': fitPars,
                                   'CompFluxes': CompFluxes,
                                   'CompFluxes_0': CompFluxes_0,
                                   'extComps': extComps,
                                   'e0': e0,
                                   'tau0': tau0,
                                   'gauss': output_gauss,
                                   'drude': output_drude
                                   }
                       }

            # Save output result to .asdf file
            target = AsdfFile(cafefit)
            if output_dirc is None:
                target.write_to('./output_cafefit.asdf')
            else:
                target.write_to(output_dirc)

