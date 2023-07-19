""" Function library for pyCAFE

These functions are used by pyCAFE in fitting, but don't care about 1D/2D. Most of these functions
shouldn't need to be changed to alter how CAFE is doing its fitting
"""

import numpy as np 
import matplotlib.pyplot as plt 
import lmfit as lm # https://dx.doi.org/10.5281/zenodo.11813
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator
from scipy.integrate import simps
from scipy.special import erf
import time
import sys
import warnings
import ast
import datetime
import configparser
from astropy.table import QTable
import astropy.units as u
from astropy.stats import mad_std
from matplotlib.ticker import ScalarFormatter

import CAFE
from CAFE.dustgrainfunc import grain_totemissivity
from CAFE.component_model import pah_drude, gauss_flux, drude_prof, drude_int_fluxes


# import pdb, ipdb

#################################
### Miscellaneous             ###
#################################

def trim_overlapping(bandnames, keep_next):
    
    val_inds = []
    band_unique_names = list(dict.fromkeys(bandnames))
    band_last_inds = []
    for band_name in band_unique_names: band_last_inds.append(np.where(bandnames == band_name)[0][-1])
    band_ind = 0
    i = 0
    while i < len(bandnames):
        if bandnames[i] == band_unique_names[band_ind]:
            val_inds.append(True)
        else:
            if bandnames[i] == band_unique_names[np.minimum(band_ind+1, len(band_unique_names)-1)]:
                if keep_next:
                    val_inds.append(True)
                    band_ind += 1
                else:
                    val_inds.append(False)
            else:
                val_inds.append(False)
        if i in band_last_inds:
            if not keep_next: band_ind += 1
        i += 1
        
    #self.waves = self.waves[val_inds]
    #if self.fluxes.ndim != 1:
    #    self.fluxes = self.fluxes[val_inds,:,:]
    #    self.flux_uncs = self.flux_uncs[val_inds,:,:]
    #    self.masks = self.masks[val_inds,:,:]
    #else:
    #    self.fluxes = self.fluxes[val_inds]
    #    self.flux_uncs = self.flux_uncs[val_inds]
    #    self.masks = self.masks[val_inds]
    #self.bandnames = bandnames[val_inds]
    
    return val_inds
        

def mask_spec(data, x=0, y=0):
    
    if data.masks.ndim != 1:
        mask = data.masks[:,y,x] != 0
        mask[np.isnan(data.fluxes[:,y,x])] = True                
        mask[(data.fluxes[:,y,x] > np.nanmedian(data.fluxes[:,y,x])+mad_std(data.fluxes[:,y,x], ignore_nan=True)*1e3) |\
             #(data.fluxes[:,y,x] < np.nanmedian(data.fluxes[:,y,x])-mad_std(data.fluxes[:,y,x], ignore_nan=True)*1e3) |\
             (data.fluxes[:,y,x] < 0)] = True # | (data.fluxes[:,y,x] < 0)
        flux = data.fluxes[~mask,y,x]
        flux_unc = data.flux_uncs[~mask,y,x]
    else:
        mask = data.masks != 0
        mask[np.isnan(data.fluxes)] = True
        mask[(data.fluxes > np.nanmedian(data.fluxes)+mad_std(data.fluxes, ignore_nan=True)*1e3) |\
             #(data.fluxes < np.nanmedian(data.fluxes)-mad_std(data.fluxes, ignore_nan=True)*1e3) |\
             (data.fluxes < 0)] = True # | (data.fluxes < 0)
        flux = data.fluxes[~mask]
        flux_unc = data.flux_uncs[~mask]
        
    wave = data.waves[~mask]
    bandname = data.bandnames[~mask]
    mask_inds = mask[~mask]
    
    return wave, flux, flux_unc, bandname, mask_inds



# RIGHT NOT THIS FUNCTION IS NOT USED SINCE THE FITTING IS DONE ALL AT ONCE
def calc_weights(wave, pwave, nBins):
    ''' Calculates array of weights to use in continuum fitting

    Ports the functionality of CAFE_WEIGHTS. However, this version explicitly 
    treats spec and phot separately, which the IDL version also does in a way
    that's much harder to follow
    Outputs verified to match IDL version 08/16/20

    Arguments:
    wave -- array of rest-frame spectrum wavelengths, in um
    pwave -- wavelengths of broad-band photometric data points, in um
    nBins -- number of log-spaced bins to divide spectrum into

    Returns: array of weights for each spectral and photometric wavelength point
    '''
    sMin = np.nanmin(wave)
    sMax = np.nanmax(wave)
    nPerDex = wave.size/np.log10(sMax/sMin)
    if len(pwave) > 0:
        pMin = np.nanmin(pwave)
        pMax = np.nanmax(pwave)
        # Gets the overall min and max wavelengths
        wMin = np.minimum(pMin, sMin)
        wMax = np.maximum(pMax, sMax)
    else:
        wMin = sMin
        wMax = sMax

    # Min/max wavelengths for each bin, and weight per bin
    wMinBin = 10**(np.log10(wMin) + np.asarray(range(10))*np.log10(wMax/wMin)/nBins)
    wMaxBin = 10**(np.log10(wMin) + (1. + np.asarray(range(10)))*np.log10(wMax/wMin)/nBins)
    weightPerBin = nPerDex*np.log10(wMaxBin/wMinBin)

    # Number of data points per bin
    nPerBin = []
    eps = 1e-5 # Not sure why this is here but it's in the IDL version
    for i in range(nBins):
        nSpec = np.where(((wave >  (1-eps)*wMinBin[i]) & (wave < wMaxBin[i]*(1+eps))))[0].size
        nPhot = np.where(((pwave >  (1-eps)*wMinBin[i]) & (pwave < wMaxBin[i]*(1+eps))))[0].size
        nPerBin.append(nSpec+nPhot)
    
    # Create weight arrays for each pixel in spec and phot
    sweights = np.zeros(wave.size)
    pweights = np.zeros(pwave.size)
    for i in range(nBins):
        idxSpec = ((wave >  (1-eps)*wMinBin[i]) & (wave < wMaxBin[i]*(1+eps)))
        idxPhot = ((pwave >  (1-eps)*wMinBin[i]) & (pwave < wMaxBin[i]*(1+eps)))
        # If there are points in the bin, assign weight based on number of points
        if nPerBin[i] > 0:
            sweights[idxSpec] += weightPerBin[i]/nPerBin[i]
            pweights[idxPhot] += weightPerBin[i]/nPerBin[i]
        # If there are no points, then distribute the weight to surrounding bins
        else:
            dLow = 0
            while nPerBin[i-dLow] == 0: dLow+=1
            dHigh = 0
            while nPerBin[i+dHigh] == 0: dHigh+=1
            wHigh = 1. - dHigh/(dLow + dHigh)
            wLow =  1. - dLow/(dLow + dHigh)

            idxSLow = ((wave > wMinBin[i-dLow]) & (wave < wMaxBin[i+dLow]))
            idxSHigh = ((wave > wMinBin[i-dHigh]) & (wave < wMaxBin[i+dHigh]))
            idxPLow = ((pwave > wMinBin[i-dLow]) & (pwave < wMaxBin[i+dLow]))
            idxPHigh = ((pwave > wMinBin[i-dHigh]) & (pwave < wMaxBin[i+dHigh]))
            # The += differs from the IDL version which I'm pretty sure is wrong since it overwrites weights
            sweights[idxSLow] += wLow * weightPerBin[i]/nPerBin[i-dLow]
            sweights[idxSHigh] += wHigh * weightPerBin[i]/nPerBin[i+dHigh] 
            pweights[idxPLow] += wLow * weightPerBin[i]/nPerBin[i-dLow]
            pweights[idxPHigh] += wHigh * weightPerBin[i]/nPerBin[i+dHigh] 
    # Normalize and return
    tsize = pweights.size + sweights.size
    tsum = pweights.sum() + sweights.sum()
    pweights*=tsize/tsum
    sweights*=tsize/tsum
    return sweights, pweights


def synphot(wave, flux, sigma, z=0., filters=None, filterPath='tables/filters/'):
    ''' Integrate flux over the specified filters

    Integrate the spectrum over the transmission curve for the specific filter, in 
    order to accurately predict photometric points. Note the (1+z) factors - the 
    spectrum needs to be in observed wavelength for accurate transmission curves
    but the input is in rest wavelength. Flux is in flux density units, so it also
    needs to be scaled.

    Arguments:
    wave  -- the rest wavelength of the spectrum
    flux -- the rest-frame fluxes
    sigma -- the flux uncertainty at each wavelength

    Keyword Arguments:
    z -- Redshift (default 0.0)
    filters -- list of filter names to integrate over
    filterPath -- directory for filter files

    Returns: Dict, containing the effective wavelength, flux measured in the
    filter, flux uncertainty in the filter, and the effective filter width
    '''
    # Load filter transmission curves
    fwaves = []
    ftrans = []
    for filt in filters:
        data = np.genfromtxt(filterPath+'filter.'+filt.lower()+'.txt', comments=';')
        fwaves.append(data[:,0])
        ftrans.append(data[:,1])

    # Convert flux density to wavelength units
    flux*=3e14/wave**2
    sigma*=3e14/wave**2

    logWave = np.log(wave)
    synWave = []
    synFlux = []
    synSigma = []
    synWidth = []
    for i in range(len(filters)):
        wave0 =  fwaves[i]*(1+z) ### Shift spec to observed frame for this
        logWave0 = np.log(wave0)
        flux0 = np.interp(logWave0, logWave, flux)*(1+z) ### due to the units we used
        sigma0 = np.interp(logWave0, logWave, sigma)*(1+z)
        trans = ftrans[i]/simps(wave0*ftrans[i], logWave0)
        fluxTot = simps(wave0*trans*flux0, logWave0)
        sigmaTot = simps(wave0*trans*sigma0, logWave0)
        waveTot = simps(wave0*trans*wave0, logWave0)
        widthTot = 1./np.nanmax(trans)
        scale = waveTot**2/3e14
        fluxTot*=scale
        sigmaTot*=scale
        synWave.append(waveTot)
        synFlux.append(fluxTot)
        synSigma.append(sigmaTot)
        synWidth.append(widthTot)
    return {'wave':synWave, 'flux':synFlux, 'sigma':synSigma, 'width':synWidth}


#################################
### Math functions            ###
#################################
def spline(xnew, xold, yold):
    ''' Wrapper for cubic spline interpolation, to work like np.interp()
    '''
    f = interp1d(xold, yold, kind='cubic')
    return f(xnew)


def intTab(f, h):
    ''' Port of Jam_IntTab - does a 5-point Newton-Cotes formula

    Arguments:
    f -- Array of x-pts to be integrated over
    h -- Array of y-values corresponding to x-pts

    Returns: Integral with respect to f 
    '''
    # Calculate number of extra points at end of integrand
    xSeg = len(f) - 1
    xExt = xSeg % 4 + 1

    # Compute integral with 5-point Newton-Cotes formula
    idx = (np.arange(int(xSeg/4)) + 1)*4
    integrand = np.sum(2*h/45.*(7*(f[idx-4]+f[idx]) + 32*(f[idx-3] + f[idx-1]) + 12*(f[idx-2])), axis=0)
    # Deal with the last couple of points
    if xExt == 1: integrand+=0
    elif xExt == 2: integrand+=0.5*h*(f[xSeg-1]+f[xSeg])
    elif xExt == 3: integrand+=(h/3.)*(f[xSeg-2] + 4*f[xSeg-1] + f[xSeg])
    else: integrand+= (3*h/8.)*(f[xSeg-3] + f[xSeg] + 3*f[xSeg-2]+f[xSeg-1])
    
    if integrand.size == 1: return float(integrand)
    else: return integrand 


##############################
### Goodness of Fit        ###
##############################
def chisquare(params, wave, flux, error, weights, cont_profs, show=True):
    ''' Objective function for fitting

    This is the objective function to minimize in the fitting. Note
    the extra sqrt at the end due to an lmfit convention

    Arguments:
    params -- lm parameters object being minimized
    wave -- rest wavelength of the spectrum
    flux -- measured fluxes at wave points
    error -- flux error estimates
    weights -- array of weights used to compute chi^2
    redchis -- Previous redchi value, used for printing
    cont_profs -- dictionary of arguments to be passed to the model flux computation

    Keyword Arguments:
    show -- whether to print the current chi2 of the fit (default True)

    Returns: weighted mean-square difference between observed flux and the 
    CAFE model from params and cont_profs.
    '''
    if weights is None:
        weights = 1./error**2
    model = get_model_fluxes(params, wave, cont_profs)

    # Note that this is the root of the thing who's sum you want to minimize
    # lmfit is kind of weird like that
    #redchi = (weights*(flux-model)**2).sum()/(wave.size-np.size(params))
    #redchis.append(redchi)
    #print(redchis[-1])
    #if len(redchis) > 1:
    #    if redchis[-1] != redchis[-2]: print('chi^2/DOF:', redchi)
    #if show and redchis[-2] - redchis[-1] > 1e-3:
        # txt = 'chi^2/DOF: ' +  str(np.round(redchi,3)).ljust(6)
        # sys.stdout.write('\r'+txt)
        # sys.stdout.flush()
        #print('chi^2/DOF:', np.round(redchi,3)) 

    #ipdb.set_trace()
    return (flux-model)*np.sqrt(weights)


#def redchi(obs, model, weights, npars=0):
#    ''' Basic weighted reduced-chi-squared calculation
#    '''
#    return np.sum(weights*(obs-model)**2/(np.size(obs-npars)))



def get_feat_pars(params, errors=False, apply_vgrad2waves=True):
    ''' Turns lm parameters into lists for flux computation
    
    Arguments:
    params -- lm Parameters object for lines being fit

    Returns: lists of line profile parameters, separated into gaussian and
             drude line profiles.
    '''
    #p = params.valuesdict()
    pkeys = params.keys()
    lwave = [] ; lgamma = [] ; lpeak = [] ; lname = [] ; ldoub = []
    pwave = [] ; pgamma = [] ; ppeak = [] ; pname = [] ; pcomp = []
    owave = [] ; ogamma = [] ; opeak = [] ; oname = []
    for key in pkeys:
        if key[0] == 'g': # e.g.: g_NeIII_33333B_Wave
            if key[-1] == 'e': # Wave
                fname = key.split('_')[1]
                fwave = key.split('_')[2]
                if errors == False:
                    lwave.append(params[key].value)
                    lgamma.append(params[key.replace('Wave','Gamma')].value)
                    lpeak.append(params[key.replace('Wave','Peak')].value)
                else:
                    lwave.append(params[key].stderr)
                    lgamma.append(params[key.replace('Wave','Gamma')].stderr)
                    lpeak.append(params[key.replace('Wave','Peak')].stderr)
                lname.append(fname+'_'+fwave[:-1])
                if fwave[-1] == 'N':
                    ldoub.append(0)
                    if apply_vgrad2waves == True: lwave[-1] *= (1+params['VGRAD']/2.998e5)
                elif fwave[-1] == 'B':
                    ldoub.append(1)
                else:
                    raise ValueError('The line lable misses the component keyword N/B')
            elif key[-1] == 'a' or key[-1] == 'k': continue
            else:
                raise ValueError('You messed with the feature parameter names')
        
        elif key[0] == 'd':
            if key[-1] == 'e':
                fname = key[:-5]
                if errors == False:
                    pwave.append(params[key].value)
                    pgamma.append(params[fname+'_Gamma'].value)
                    ppeak.append(params[fname+'_Peak'].value)
                else:
                    pwave.append(params[key].stderr)
                    pgamma.append(params[fname+'_Gamma'].stderr)
                    ppeak.append(params[fname+'_Peak'].stderr)
                pname.append(fname[1:])
                pcomp.append(key.split('_')[0][1:])
                if apply_vgrad2waves == True: pwave[-1] *= (1+params['VGRAD']/2.998e5)
            elif key[-1] == 'a' or key[-1] == 'k': continue
            else:
                raise ValueError('You messed with the feature parameter names')

        elif key[0] == 'o':
            if key[-1] == 'e': # Wave
                fname = key.split('_')[1]
                fwave = key.split('_')[2]
                if errors == False:
                    owave.append(params[key].value)
                    ogamma.append(params[key.replace('Wave','Gamma')].value)
                    opeak.append(params[key.replace('Wave','Peak')].value)
                else:
                    owave.append(params[key].stderr)
                    ogamma.append(params[key.replace('Wave','Gamma')].stderr)
                    opeak.append(params[key.replace('Wave','Peak')].stderr)
                oname.append(fname+'_'+fwave)
                if apply_vgrad2waves == True: owave[-1] *= (1+params['VGRAD']/2.998e5)
            elif key[-1] == 'a' or key[-1] == 'k': continue
            else:
                raise ValueError('You messed with the feature parameter names')
        
        else:
            continue

    gauss = [np.asarray(lwave), np.asarray(lgamma), np.asarray(lpeak), lname, np.asarray(ldoub)]
    drude = [np.asarray(pwave), np.asarray(pgamma), np.asarray(ppeak), pname, pcomp]
    gauss_opc = [np.asarray(owave), np.asarray(ogamma), np.asarray(opeak), oname]

    return gauss, drude, gauss_opc


############################
#### Model Computations ####
############################
def get_model_fluxes(params, wave, cont_profs, comps=False):
    ''' Return the model flux

    Computes the model CAFE flux corresponding to pars and cont_profs at 
    the specified wavelengths. This version is used for the fitting,
    see below for an alternative with more plot-friendly options.

    Arguments:
    params -- lm Parameters object with CAFE continuum parameters
    wave -- wavelength points to compute flux
    Cont_Profs -- a dictionary containing:
        waveSED
        wave0
        flux0
        kAbs, kExt
        E_CIR, E_COO, E_CLD, E_WRM, etc
        kIce, kHac, and kCOrv
        various sources
        filters and pwaves
        dofilter
        z

    Returns: array of fluxes at given wavelengths
    '''
    # Get model parameters
    p = params.valuesdict()
    
    gauss, drude, gauss_opc = get_feat_pars(params)

    # Get variations on wavelengh vector
    logWave = np.log(wave)
    waveMod = cont_profs['waveSED']
    logWaveMod = np.log(waveMod)
    if waveMod.shape==wave.shape and np.allclose(wave, waveMod): fixWave = False
    else: fixWave=True
    
    # Calculate total normalized dust opacity at 9.7 um
    kAbsTot = cont_profs['kAbs']['Carb'] + cont_profs['kAbs']['SilAmoTot']
    #kExtTot = cont_profs['kExt']['Carb'] + cont_profs['kAbs']['SilAmoTot'] Substitute kAbs by kExt by TDS
    kExtTot = cont_profs['kExt']['Carb'] + cont_profs['kExt']['SilAmoTot']
    idxMax = ((waveMod > 8.5) & (waveMod < 11.))
    kAbsTot0 = np.nanmax(kAbsTot[idxMax])
    kExtTot0 = np.nanmax(kExtTot[idxMax])
    kAbsTot/=kAbsTot0
    kExtTot/=kExtTot0
    kTot = kAbsTot
    kTot0 = kAbsTot0
    if cont_profs['ExtOrAbs'].upper() == 'ABS': 
        kTotDsk = kAbsTot
    else: 
        kTotDsk = kExtTot 

    # Additional opacity sources
    #tauIce = p['TAU_ICE']*(p['ICE_RAT']*0.25*cont_profs['kIce3']+cont_profs['kIce6']) + p['TAU_HAC']*cont_profs['kHac'] + p['TAU_CORV']*cont_profs['kCOrv']
    tauIce = p['ICE3_TAU'] * (0.25*cont_profs['kIce3']) + p['ICE6_TAU'] * cont_profs['kIce6'] + \
             p['HAC_TAU'] * cont_profs['kHac'] + p['CORV_TAU']*cont_profs['kCOrv'] + \
             p['CO2_TAU'] * cont_profs['kCO2'] + \
             p['CRYSI_233_TAU']*cont_profs['kCrySi_233']

    # tauIce = p['TAU_ICE'] * (p['ICE_RAT'] * 0.25*cont_profs['kIce3'] + cont_profs['kIce6']) + \
    #          p['TAU_HAC'] * cont_profs['kHac'] + p['TAU_CORV'] * cont_profs['kCOrv'] + \
    #          p['TAU_CO2'] * cont_profs['kCO2']

    # The gaussian opacities are moved with VGRAD
    if gauss_opc[0].size > 0:
        tau_gopc = gauss_flux(waveMod, [gauss_opc[0], gauss_opc[1], gauss_opc[2]])
    else:
        tau_gopc = np.zeros(waveMod.size)

    tauIce += tau_gopc

    ### CIR component flux
    if p['CIR_FLX'] > 0:
        # Skipping the if E_CIR = 0 conditional because I don't think it's actually called
        # NOTE - grain_totemissivity is completely untested
        jCIR = grain_totemissivity(waveMod, p['CIR_TMP'], E_T=cont_profs['E_CIR'], FASTTEMP=cont_profs['FASTTEMP'])

        jCIR0 = np.interp(np.log(cont_profs['wave0']['CIR']), logWaveMod, jCIR)
        if np.abs(jCIR0) == 0.: jCIR0 = 1.
        fCIR = p['CIR_FLX'] * cont_profs['flux0']['CIR'] / jCIR0 * jCIR
        fCIR[fCIR < 0] = 0.
        if comps:
            fCIR_0 = np.copy(fCIR)
            #fCIR_Tot = np.trapz(fCIR/waveMod, logWaveMod)
            #fDST_Tot = np.copy(fCIR_Tot)
    else:
        fCIR = np.zeros(waveMod.size)
        if comps:
            jCIR = np.zeros(waveMod.size)
            jCIR0 = 0.
            fCIR_0 = np.zeros(waveMod.size)
            #fCIR_Tot = 0.
            #fDST_Tot = 0.

    ### CLD component flux
    if p['CLD_FLX'] > 0:
        # Skipping if E_CLD = 0
        jCLD = grain_totemissivity(waveMod, p['CLD_TMP'], E_T=cont_profs['E_CLD'], FASTTEMP=cont_profs['FASTTEMP'])

        jCLD0 = np.interp(np.log(cont_profs['wave0']['CLD']), logWaveMod, jCLD)
        if np.abs(jCLD0) == 0.: jCLD0 = 1.
        fCLD = p['CLD_FLX'] * cont_profs['flux0']['CLD'] / jCLD0 * jCLD
        fCLD[jCLD < 0] = 0.
        if comps:
            fCLD_0 = np.copy(fCLD)
            #fCLD_Tot = np.trapz(fCLD/waveMod, logWaveMod)
            #fDST_Tot+=fCLD_Tot
    else:
        fCLD = np.zeros(waveMod.size)
        if comps:
            jCLD = np.zeros(waveMod.size)
            jCLD0 = 0.
            fCLD_0 = np.zeros(waveMod.size)
            #fCLD_Tot = 0.

    ### COO component flux
    if p['COO_FLX'] > 0:
        if p['COO_TAU'] > 0:
            tauScrCOO = (1. - p['COO_MIX'])*(p['COO_TAU'] * kTot + tauIce)
            tauMixCOO = p['COO_MIX'] * (p['COO_TAU'] * kTot + tauIce)
            extScrCOO = np.exp(-tauScrCOO)
            extMixCOO = np.ones(waveMod.size)
            idx = tauMixCOO > 0
            extMixCOO[idx] = (1. - np.exp(-tauMixCOO[idx])) / tauMixCOO[idx]
            extCOO = (1. - p['COO_COV']) + p['COO_COV'] * extScrCOO * extMixCOO
        else:
            extCOO = np.ones(waveMod.size)

        # Again skipping if size(E_COO) = 0 condition
        jCOO_0 = grain_totemissivity(waveMod, p['COO_TMP'], E_T=cont_profs['E_COO'], FASTTEMP=cont_profs['FASTTEMP'])

        jCOO = extCOO * jCOO_0
        jCOO0 = np.interp(np.log(cont_profs['wave0']['COO']), logWaveMod, jCOO)
        if np.abs(jCOO0) == 0.: jCOO0 = 1.
        fCOO = p['COO_FLX'] * cont_profs['flux0']['COO'] / jCOO0 * jCOO
        fCOO[fCOO < 0] = 0.
        if comps:
            fCOO_0 = p['COO_FLX'] * cont_profs['flux0']['COO'] / jCOO0 * jCOO_0
            fCOO_0[fCOO_0 < 0] = 0.
            #fCOO_Tot = np.trapz(fCOO/waveMod, logWaveMod)
            #fDST_Tot+=fCOO_Tot
    else:
        fCOO = np.zeros(waveMod.size)
        if comps:
            extCOO = np.ones(waveMod.size)
            jCOO = np.zeros(waveMod.size)
            jCOO0 = 0.
            fCOO_0 = np.zeros(waveMod.size)
            #fCOO_Tot = 0

    ### WRM component flux
    if p['WRM_FLX'] > 0:
        # The first onion conditional is here, I'm skipping it for now
        if p['WRM_TAU'] > 0:
            tauScrWRM = (1. - p['WRM_MIX'])*(p['WRM_TAU'] * kTot + tauIce)
            tauMixWRM = p['WRM_MIX'] * (p['WRM_TAU'] * kTot + tauIce)
            extScrWRM = np.exp(-tauScrWRM)
            extMixWRM = np.ones(waveMod.size)
            idx = tauMixWRM > 0
            extMixWRM[idx] = (1. - np.exp(-tauMixWRM[idx])) / tauMixWRM[idx]
            extWRM = (1. - p['WRM_COV']) + p['WRM_COV'] * extScrWRM * extMixWRM
        else:
            extWRM = np.ones(waveMod.size)

        # Skipping the size 0 condition again
        jWRM_0 = grain_totemissivity(waveMod,  p['WRM_TMP'], E_T=cont_profs['E_WRM'], FASTTEMP=cont_profs['FASTTEMP'])

        jWRM = extWRM * jWRM_0
        jWRM0 = np.interp(np.log(cont_profs['wave0']['WRM']), logWaveMod, jWRM)
        if np.abs(jWRM0) == 0.: jWRM0 = 1.
        fWRM = p['WRM_FLX'] * cont_profs['flux0']['WRM'] / jWRM0 * jWRM
        fWRM[fWRM < 0] = 0.
        if comps:
            fWRM_0 = p['WRM_FLX'] * cont_profs['flux0']['WRM'] / jWRM0 * jWRM_0
            fWRM_0[fWRM_0 < 0] = 0.
            #fWRM_Tot = np.trapz(fWRM/waveMod, logWaveMod)
            #fDST_Tot+=fWRM_Tot
    else:
        fWRM = np.zeros(waveMod.size)
        if comps:
            extWRM = np.ones(waveMod.size)
            jWRM = np.zeros(waveMod.size)
            jWRM0 = 0.
            fWRM_0 = np.zeros(waveMod.size)
            #fWRM_Tot = 0

    ### HOT component flux
    if p['HOT_FLX'] > 0:
        # Another onion conditional
        if p['HOT_TAU'] > 0:
            tauScrHOT = (1. - p['HOT_MIX'])*(p['HOT_TAU'] * kTot + tauIce)
            tauMixHOT = p['HOT_MIX'] * (p['HOT_TAU'] * kTot + tauIce)
            extScrHOT = np.exp(-tauScrHOT)
            extMixHOT = np.ones(waveMod.size)
            idx = tauMixHOT > 0
            extMixHOT[idx] = (1. - np.exp(-tauMixHOT[idx])) / tauMixHOT[idx]
            extHOT = (1. - p['HOT_COV']) + p['HOT_COV'] * extScrHOT * extMixHOT
        else:
            extHOT = np.ones(waveMod.size)

        # Skipping the size 0 condition again
        jHOT_0 = grain_totemissivity(waveMod,  p['HOT_TMP'], E_T=cont_profs['E_HOT'], FASTTEMP=cont_profs['FASTTEMP'])

        jHOT = extHOT * jHOT_0
        jHOT0 = np.interp(np.log(cont_profs['wave0']['HOT']), logWaveMod, jHOT)
        if np.abs(jHOT0) == 0.: jHOT0 = 1.
        fHOT = p['HOT_FLX'] * cont_profs['flux0']['HOT'] / jHOT0 * jHOT
        fHOT[fHOT < 0] = 0.
        if comps:
            fHOT_0 = p['HOT_FLX'] * cont_profs['flux0']['HOT'] / jHOT0 * jHOT_0
            fHOT_0[fHOT_0 < 0] = 0.
            #fHOT_Tot = np.trapz(fHOT/waveMod, logWaveMod)
            #fDST_Tot+=fHOT_Tot
    else:
        fHOT = np.zeros(waveMod.size)
        if comps:
            extHOT = np.ones(waveMod.size)
            jHOT = np.zeros(waveMod.size)
            jHOT0 = 0.
            fHOT_0 = np.zeros(waveMod.size)
            #fHOT_Tot = 0

    ### PAH component flux
    if p['PAH_FLX'] > 0:
        if p['PAH_TAU'] > 0:
            # tau PAH being tied is dealt with using conditionals in the IDL
            # version - lmfit should let us deal with it in the tie in the 
            # initial parameter definition
            tauScrPAH = (1. - p['PAH_MIX']) * (p['PAH_TAU'] * kTot + tauIce)
            tauMixPAH = p['PAH_MIX'] * (p['PAH_TAU'] * kTot + tauIce)
            extScrPAH = np.exp(-tauScrPAH)
            extMixPAH = np.ones(waveMod.size)
            idx = tauMixPAH > 0
            extMixPAH[idx] = (1. - np.exp(-tauMixPAH[idx])) / tauMixPAH[idx]
            extPAH = (1. - p['PAH_COV']) + p['PAH_COV'] * extScrPAH * extMixPAH
        else:
            extPAH = np.ones(waveMod.size)

        # We move the profiles with VGRAD
        if drude[0].size > 0:
            jPAH_0 = drude_prof(waveMod, [drude[0], drude[1], drude[2]])
            jPAH_0[jPAH_0 < 0] = 0.0
            jPAH = extPAH * jPAH_0
        else:
            jPAH_0 = np.zeros(waveMod.size)
            jPAH = np.zeros(waveMod.size)

        #jPAH0 = np.interp(np.log(cont_profs['wave0']['PAH']), logWaveMod, jPAH)
        #if np.abs(jPAH0) == 0.: jPAH0 = 1.
        # fPAH = p['PAH_FLX'] * cont_profs['flux0']['PAH'] / jPAH0 * jPAH
        ### Skip scaling factors
        fPAH = jPAH
        fPAH[fPAH < 0] = 0.
        if comps:
            fPAH_0 = jPAH_0
            fPAH_0[fPAH_0 < 0] = 0.
            #fPAH_Tot = np.trapz(fPAH/waveMod, logWaveMod)
    else:
        fPAH = np.zeros(waveMod.size)
        if comps:
            extPAH = np.ones(waveMod.size)
            jPAH = np.zeros(waveMod.size)
            #jPAH0 = 0.
            fPAH_0 = np.zeros(waveMod.size)
            #fPAH_Tot = 0.

    ### STR component flux
    if p['STR_FLX'] > 0:
        if p['STR_TAU'] > 0:
            tauScrSTR = (1. - p['STR_MIX']) * (p['STR_TAU'] * kTot + tauIce)
            tauMixSTR = p['STR_MIX'] * (p['STR_TAU'] * kTot + tauIce)
            extScrSTR = np.exp(-tauScrSTR)
            extMixSTR = np.ones(waveMod.size)
            idx = tauMixSTR > 0
            extMixSTR[idx] = (1. - np.exp(-tauMixSTR[idx]))/tauMixSTR[idx]
            extSTR = (1. - p['STR_COV']) + p['STR_COV'] * extScrSTR * extMixSTR
        else:
            extSTR = np.ones(waveMod.size)

        jSTR_0 = cont_profs['sourceSTR']

        jSTR = extSTR * jSTR_0
        jSTR0 = np.interp(np.log(cont_profs['wave0']['STR']), logWaveMod, jSTR)
        if np.abs(jSTR0) == 0.: jSTR0 = 1.
        fSTR = p['STR_FLX'] * cont_profs['flux0']['STR'] / jSTR0 * jSTR
        fSTR[fSTR < 0] = 0.
        if comps:
            fSTR_0 = p['STR_FLX'] * cont_profs['flux0']['STR'] / jSTR0 * jSTR_0
            fSTR_0[fSTR_0 < 0] = 0.
            #fSTR_Tot = np.trapz(fSTR/waveMod, logWaveMod)
    else:
        fSTR = np.zeros(waveMod.size)
        if comps:
            extSTR = np.ones(waveMod.size)
            jSTR = np.zeros(waveMod.size)
            jSTR0 = 0.
            fSTR_0 = np.zeros(waveMod.size)
            #fSTR_Tot = 0.

    ### STB component flux
    if p['STB_FLX'] > 0:
        if p['STB_TAU'] > 0:
            tauScrSTB = (1. - p['STB_MIX']) * (p['STB_TAU'] * kTot + tauIce)
            tauMixSTB = p['STB_MIX'] * (p['STB_TAU'] * kTot + tauIce)
            extScrSTB = np.exp(-tauScrSTB)
            extMixSTB = np.ones(waveMod.size)
            idx = tauMixSTB > 0
            extMixSTB[idx] = (1. - np.exp(-tauMixSTB[idx]))/tauMixSTB[idx]
            extSTB = (1. - p['STB_COV']) + p['STB_COV'] * extScrSTB * extMixSTB
        else:
            extSTB = np.ones(waveMod.size)

        jSTB_0_100 = p['STB_100']*cont_profs['source100Myr']
        jSTB_0_010 = ((1. - p['STB_100']) * p['STB_010'] * cont_profs['source10Myr'])
        jSTB_0_002 = ((1. - p['STB_100']) * (1. - p['STB_010']) * cont_profs['source2Myr'])
        jSTB_0 = jSTB_0_100 + jSTB_0_010 + jSTB_0_002

        jSTB = extSTB * jSTB_0
        jSTB0 = np.interp(np.log(cont_profs['wave0']['STB']), logWaveMod, jSTB)
        if np.abs(jSTB0) == 0.: jSTB0 = 1.
        const = p['STB_FLX'] * cont_profs['flux0']['STB'] / jSTB0
        fSTB = const * jSTB
        fSTB[fSTB < 0] = 0.
        if comps:
            fSTB_100 = const * jSTB_0_100 * extSTB
            fSTB_100[fSTB_100 < 0] = 0.
            fSTB_010 = const * jSTB_0_010 * extSTB
            fSTB_010[fSTB_010 < 0] = 0.
            fSTB_002 = const * jSTB_0_002 * extSTB
            fSTB_002[fSTB_002 < 0] = 0.
            #fSTB = fSTB_100 + fSTB_010 + fSTB_002
            #fSTB_0 = const * jSTB_0
            fSTB_0_100 = const * jSTB_0_100
            fSTB_0_100[fSTB_0_100 < 0] = 0.
            fSTB_0_010 = const * jSTB_0_010
            fSTB_0_010[fSTB_0_010 < 0] = 0.
            fSTB_0_002 = const * jSTB_0_002
            fSTB_0_002[fSTB_0_002 < 0] = 0.
            fSTB_0 = fSTB_0_100 + fSTB_0_010 + fSTB_0_002
            #fSTB_Tot = np.trapz(fSTB/waveMod, logWaveMod)
    else:
        fSTB = np.zeros(waveMod.size)
        if comps:
            extSTB = np.ones(waveMod.size)
            fSTB_100 = np.zeros(waveMod.size)
            fSTB_010 = np.zeros(waveMod.size)
            fSTB_002 = np.zeros(waveMod.size)
            jSTB = np.zeros(waveMod.size)
            jSTB0 = 0.
            fSTB_0 = np.zeros(waveMod.size)
            fSTB_0_100 = np.zeros(waveMod.size)
            fSTB_0_010 = np.zeros(waveMod.size)
            fSTB_0_002 = np.zeros(waveMod.size)
            #fSTB_Tot = 0.

    ### DSK component flux
    # Again don't think the tying conditionals are necessary now
    if p['DSK_FLX'] > 0:
        if p['DSK_TAU'] > 0:
            extDSK = (1. - p['DSK_COV']) + p['DSK_COV'] * np.exp(-p['DSK_TAU'] * kTotDsk - tauIce)
        else:
            extDSK = np.ones(waveMod.size)
            
        jDSK_0 = cont_profs['sourceDSK']
        
        jDSK = extDSK * jDSK_0
        jDSK0 = np.interp(np.log(cont_profs['wave0']['DSK']), logWaveMod, jDSK)
        if np.abs(jDSK0)== 0.: jDSK0 = 1.
        fDSK = p['DSK_FLX'] * cont_profs['flux0']['DSK'] / jDSK0 * jDSK
        fDSK[fDSK < 0] = 0.
        if comps:
            fDSK_0 = p['DSK_FLX'] * cont_profs['flux0']['DSK'] / jDSK0 * jDSK_0
            fDSK_0[fDSK_0 < 0] = 0.
            #fDSK_Tot = np.trapz(fDSK/waveMod, logWaveMod)
    else:
        fDSK = np.zeros(waveMod.size)
        if comps:
            extDSK = np.ones(waveMod.size)
            jDSK = np.zeros(waveMod.size)
            jDSK0 = 0.
            fDSK_0 = np.zeros(waveMod.size)
            #fDSK_Tot = 0.

    ### Line component flux 
    if gauss[0].size > 0:
        # fLIN = gauss_flux(waveMod, gauss)
        # fLIN[fLIN < 0] = 0.0
        # if comps: fLIN_0 = fLIN
        fLIN_0 = gauss_flux(waveMod, [gauss[0], gauss[1], gauss[2]])
        fLIN_0[fLIN_0 < 0] = 0.0
        fLIN = extPAH * fLIN_0
    else:
        #fLIN = np.zeros(waveMod.size)
        fLIN_0 = np.zeros(waveMod.size)
        fLIN = np.zeros(waveMod.size)
        #if comps: fLIN_0 = np.zeros(waveMod.size)

    ### Calculate total flux and spline to input wave
    fluxMod = fCIR + fCLD + fCOO + fWRM + fHOT + fLIN + fPAH + fSTR + fSTB + fDSK

    # print('LIN', fLIN) # This seems to differ slightly from IDL version
    # print('PAH', fPAH) # seems to be consistently about 2 percent off

    if fixWave: 
        # For some reason using my spline function breaks things here
        #f1 = interp1d(logWaveMod, fluxMod)
        #flux = f1(logWave)
        # Instead of interpolating, just chose the overlapping indices/wavelengths
        flux = fluxMod[np.isin(logWaveMod, logWave)]
    else:
        flux = np.copy(fluxMod)
    #ipdb.set_trace()

    ### Integrate over filters
    if cont_profs['DoFilter']:
        photdict = synphot(waveMod*(1+cont_profs['z']), fluxMod, np.zeros(waveMod.size), 
                           z=cont_profs['z'], filters=cont_profs['filters'])
        for i in range(len(cont_profs['pwaves'])):
            flux[np.abs(wave-cont_profs['pwaves'][i]) < 1e-14] = photdict['flux'][i]
    # Some of the long wavelength photometry is a little funny

    if comps:
        
        fCON = fCIR + fCLD + fCOO + fWRM + fHOT + fSTR + fSTB + fDSK
        fDST = fCIR + fCLD + fCOO + fWRM + fHOT
        fSRC = fSTR + fSTB + fDSK
        fFTS = fPAH + fLIN

        CompFluxes = {'wave':waveMod, 'fCIR':fCIR, 'fCLD':fCLD, 'fCOO':fCOO, 'fWRM':fWRM, 'fHOT':fHOT, 
                 'fLIN':fLIN, 'fPAH':fPAH, 'fSTR':fSTR, 'fSTB':fSTB, 'fDSK':fDSK,
                 'fCON':fCON, 'fDST':fDST, 'fSRC':fSRC, 'fFTS': fFTS,
                 'STB_100':fSTB_100, 'STB_010':fSTB_010, 'STB_002':fSTB_002}

        # Model components
        fluxMod0 = fCIR_0 + fCLD_0 + fCOO_0 + fWRM_0 + fHOT_0 + fLIN_0 + fPAH_0 + fSTR_0 + fSTB_0 + fDSK_0
        fCON_0 = fCIR_0 + fCLD_0 + fCOO_0 + fWRM_0 + fHOT_0 + fSTR_0 + fSTB_0 + fDSK_0
        fDST_0 = fCIR_0 + fCLD_0 + fCOO_0 + fWRM_0 + fHOT_0
        fSRC_0 = fSTR_0 + fSTB_0 + fDSK_0
        CompFluxes_0 = {'wave':waveMod, 'fCIR0':fCIR_0, 'fCLD0':fCLD_0, 'fCOO0':fCOO_0, 'fWRM0':fWRM_0, 'fHOT0':fHOT_0, 
                        'fLIN0':fLIN_0, 'fPAH0':fPAH_0, 'fSTR0':fSTR_0, 'fSTB0':fSTB_0, 'fDSK0':fDSK_0,
                        'fCON0':fCON_0, 'fDST0':fDST_0, 'fSRC0':fSRC_0,
                        'STB0_100':fSTB_0_100, 'STB0_010':fSTB_0_010, 'STB0_002':fSTB_0_002}

        # Extinctions
        extComps = {'wave':waveMod, 'extCOO':extCOO, 'extWRM':extWRM, 'extHOT':extHOT, 'extPAH':extPAH, 
                    'extSTR':extSTR, 'extSTB':extSTB, 'extDSK':extDSK,}

        # E0/tau0
        emiss = {'jCIR':jCIR, 'jCLD':jCLD, 'jCOO':jCOO, 'jWRM':jWRM, 'jHOT':jHOT, 'jPAH':jPAH}
        tau0 = {'tau0COO':p['COO_TAU'], 'tau0WRM':p['WRM_TAU'], 'tau0HOT':p['HOT_TAU'], 'tau0PAH':p['PAH_TAU'],
                'tauSTR':p['STR_TAU'], 'tau0STB':p['STB_TAU'], 'tau0DSK':p['DSK_TAU']}

        return CompFluxes, CompFluxes_0, extComps, emiss, tau0

    else:

        return flux



############################
### Error estimation     ###
############################
def cont_err(params, derivs, covar, pkeys=None):
    ''' Calculates continuum uncertainty

    Arguments:
    params -- lm parameters object with fitting errors
    derivs -- numerical derivative for each parameter at its value
    covar -- covariance matrix of parameters

    Keyword Arugments:
    pkeys -- list of parameter names to use in estimating uncertainty (default None)

    Returns: Estimated propegated undertainty in flux
    '''
    if pkeys is None:
        pkeys = []
        for par in params:
            if params[par].vary:
                pkeys.append(par)
        inds = range(len(pkeys))
    else:
        ### Need to identify indicies corresponding to specified keys
        pass

    totuncert = np.zeros(derivs[0].size)
    for i in inds:
        if params[pkeys[i]].vary:
            ### Normal error from parameter
            err = (params[pkeys[i]].stderr * derivs[i])**2
            totuncert += err 
            ### Loop over params to add covariances
            for j in range(len(params)):
                if j != i:
                    err = 2 * np.abs(covar[i][j]**2 * derivs[i] * derivs[j])
                    totuncert += err 

    return np.sqrt(totuncert)

def deriv(func, wave, params, funcargs=None, eps=1e-2):
    '''Takes a numerical derivative of callable with form f(pars, wave, funcargs)

    Arguments:
    func -- callable, function to differentiate
    wave -- wavelength points where we take the derivative
    params -- params we're taking derivatives with respect to
    
    Keyword Arugments:
    funcargs -- Dict of additional parameters to pass to func (default None)
    eps --  Fractional variation to use in differentiation (default 1e-2)

    Returns: list of numerical derivatives for each parameter at each wavelength
    '''
    derivs = []
    pkeys = list(params.valuesdict().keys())
    for i in range(len(pkeys)):
        ### Skip fixed parameters
        if params[pkeys[i]].vary:
            ### Need to make copy of params to manipulate
            parHi = params.copy()
            parHi[pkeys[i]].value = params[pkeys[i]].value+eps*params[pkeys[i]].value
            parLo = params.copy()
            parLo[pkeys[i]].value = params[pkeys[i]].value-eps*params[pkeys[i]].value
            yh = func(parHi, wave, funcargs)
            yl = func(parLo, wave, funcargs)
            derivs.append((yh-yl)/(2*eps*params[pkeys[i]].value))
    return derivs


#################################
### Common plotting functions ###
#################################
def sedplot(wave, flux, sigma, comps, weights=None, npars=1):
    ''' Plot the overall SED and the CAFE fit

    Arguments:
    wave -- rest wavelength of observed spectrum
    flux -- observed flux values
    sigma -- uncertainties in measured fluxes
    comps -- dict of component fluxes

    Keyword Arugments:
    weights -- CAFE weights to use in estimating final chi^2 (default None)
    npars -- number of parameters varied (default 1)

    Returns: Tuple of figure, chi^2 of fit

    '''
    fCir = comps['fCIR']
    fCld = comps['fCLD']
    fCoo = comps['fCOO']
    fWrm = comps['fWRM']
    fHot = comps['fHOT']
    fStb = comps['fSTB']
    fStr = comps['fSTR']
    fDsk = comps['fDSK']
    fLin = comps['fLIN']
    fPAH = comps['fPAH']
    fMod = fCir + fCld + fCoo + fWrm + fHot + fStb + fStr + fDsk + fLin + fPAH
    fCont = fCir + fCld + fCoo + fWrm + fHot + fStb + fStr + fDsk
    wavemod = comps['wave']

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]}, figsize=(8,8), sharex=True)
    ax1.scatter(wave, flux, marker='s', s=6, edgecolor='k', facecolor='none', label='Data')
    ax1.errorbar(wave, flux, yerr=sigma, fmt='none', color='k')
    ax1.plot(wavemod, fCont, color='gray', label='Continuum Fit', linestyle='-')
    ax1.plot(wavemod, fCont+fLin+fPAH, color='#4c956c', label='Total Fit', linewidth='2', zorder=5, alpha=0.95) # green

    alpha = 0.7
    if np.any(fCir > 0):
        ax1.plot(wavemod, fCir, label='Cirrus', c='tab:cyan', alpha=alpha)
    if np.sum(fCld > 0):
        ax1.plot(wavemod, fCld, label='Cold', c='tab:blue', alpha=alpha)
    if np.any(fCoo > 0):
        ax1.plot(wavemod, fCoo, label='Cool', c='#008080', alpha=alpha) # teal
    if np.any(fWrm > 0):
        ax1.plot(wavemod, fWrm, label='Warm', c='tab:orange', alpha=alpha)
    if np.any(fHot > 0):
        ax1.plot(wavemod, fHot, label='Hot', c='#FFD700', alpha=alpha) # gold
    if np.any(fStb > 0): 
        ax1.plot(wavemod, fStb, label='Starburst', c='tab:brown', alpha=alpha)
    if np.any(fStr > 0):
        ax1.plot(wavemod, fStr, label='Stellar', c='#FF4500', alpha=alpha) # orangered
    if np.any(fDsk > 0):
        ax1.plot(wavemod, fDsk, label='AGN', c='tab:red', alpha=alpha)
    if np.any(fLin > 0):
        ax1.plot(wavemod, fCont+fLin, label='Lines', c='#ADD8E6', alpha=alpha, linewidth=0.5) # lightblue

    ax1.legend(loc='lower right')
    ax1.tick_params(direction='in', which='both', length=6, width=1, right=True, top=True)
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(bottom=0.8*np.nanmin(flux), top=1.2*np.nanmax(flux))
    ax1.set_xlim(left=1, right=1e3)
    ax1.set_ylabel(r'$f_\nu$ (Jy)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    interpMod = np.interp(wave, comps['wave'], fMod)
    if weights is not None:
        chiSqrTot = np.sum(weights*(flux-interpMod)**2)/(wave.size - npars)
        #print('Final reduced chi^2:', np.round(chiSqrTot,3))
    ax2.plot(wave, (flux-interpMod)/sigma, color='k')
    ax2.axhline(0., color='k', linestyle='--')
    ax2.tick_params(direction='in', which='both', length=6, width=1, right=True, top=True)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylim(bottom=-7, top=7)
    ax2.set_xlabel(r'$\lambda_{rest}$ $(\mu m)$', fontsize=14)
    ax2.set_ylabel(r'$f^{data}_\nu - f^{tot}_\nu$ $(\sigma)$', fontsize=14)

    ax1.set_title('SED Decomposition', fontsize=16)
    plt.subplots_adjust(hspace=0)
    return fig, chiSqrTot


def cafeplot(wave, flux, sigma, comps, gauss, drude, plot_drude=True, pahext=None):
    ''' Plot the SED and the CAFE fit over the spectrum wavelength range

    Arguments:
    wave -- rest wavelength of observed spectrum
    flux -- observed flux values
    sigma -- uncertainties in measured fluxes
    comps -- dict of component fluxes

    Keyword Arugments:
    weights -- CAFE weights to use in estimating final chi^2 (default None)
    drude -- The collection of ouput parameters of Drude profiles
    plot_drude -- if true, plots individual drude profiles, otherwise plots total
    PAH contribution. (default false)
    pahext -- if not None, applies extinction curve to PAHs

    Returns: Figure
    '''
    fCir = comps['fCIR']
    fCld = comps['fCLD']
    fCoo = comps['fCOO']
    fWrm = comps['fWRM']
    fHot = comps['fHOT']
    fStb = comps['fSTB']
    fStr = comps['fSTR']
    fDsk = comps['fDSK']
    fLin = comps['fLIN']
    fPAH = comps['fPAH']
    fMod = fCir + fCld + fCoo + fWrm + fHot + fStb + fStr + fDsk + fLin + fPAH
    fCont = fCir + fCld + fCoo + fWrm + fHot + fStb + fStr + fDsk
    wavemod = comps['wave']

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3,1]}, figsize=(8,8), sharex=True)
    ax1.scatter(wave, flux, marker='o', s=6, edgecolor='k', facecolor='none', label='Data', alpha=0.9)
    ax1.errorbar(wave, flux, yerr=sigma, fmt='none', color='k', alpha=0.1)
    ax1.plot(wavemod, fCont, color='gray', label='Continuum Fit', linestyle='-', zorder=4, alpha=0.8)
    ax1.plot(wavemod, fCont+fLin+fPAH, color='#4c956c', label='Total Fit', linewidth=1.5, zorder=5, alpha=0.85) # green

    alpha = 0.6
    lw = 0.8
    if np.any(fCir > 0):
        ax1.plot(wavemod, fCir, label='Cirrus', c='tab:cyan', alpha=alpha, linewidth=lw)
    if np.sum(fCld > 0):
        ax1.plot(wavemod, fCld, label='Cold', c='tab:blue', alpha=alpha, linewidth=lw)
    if np.any(fCoo > 0):
        ax1.plot(wavemod, fCoo, label='Cool', c='#008080', alpha=alpha, linewidth=lw) # teal
    if np.any(fWrm > 0):
        ax1.plot(wavemod, fWrm, label='Warm', c='tab:orange', alpha=alpha, linewidth=lw)
    if np.any(fHot > 0):
        ax1.plot(wavemod, fHot, label='Hot', c='#FFD700', alpha=alpha, linewidth=lw) # gold
    if np.any(fStb > 0): 
        ax1.plot(wavemod, fStb, label='Starburst', c='tab:brown', alpha=alpha, linewidth=lw)
    if np.any(fStr > 0):
        ax1.plot(wavemod, fStr, label='Stellar', c='#FF4500', alpha=alpha, linewidth=lw) # orangered
    if np.any(fDsk > 0):
        ax1.plot(wavemod, fDsk, label='AGN', c='tab:red', alpha=alpha, linewidth=lw)
    if np.any(fLin > 0):
        ax1.plot(wavemod, fCont+fLin, label='Lines', c='#1e6091', alpha=alpha, linewidth=lw) # blue

    # Plot lines
    for i in range(len(gauss[0])):
        if pahext is None:
            pahext = np.ones(wavemod.shape)
        # pahext = np.ones(wavemod.shape)
        lflux = gauss_flux(wavemod, [[gauss[0][i]], [gauss[1][i]], [gauss[2][i]]], pahext)
        
        if i == 0:
            ax1.plot(wavemod, lflux+fCont, color='#1e6091', label='_nolegend_', alpha=alpha, linewidth=0.4)
        else:
            ax1.plot(wavemod, lflux+fCont, color='#1e6091', label='_nolegend_', alpha=alpha, linewidth=0.4)
    
    # Plot PAH features
    if plot_drude is True:
        for i in range(len(drude[0])):
            if pahext is None: 
                pahext = np.ones(wavemod.shape)
            # pahext = np.ones(wavemod.shape)
            dflux = drude_prof(wavemod, [[drude[0][i]], [drude[1][i]], [drude[2][i]]], pahext)

            if i == 0:
                ax1.plot(wavemod, dflux+fCont, color='purple', label='PAHs', alpha=alpha, linewidth=0.5)
            else:
                ax1.plot(wavemod, dflux+fCont, color='purple', label='_nolegend_', alpha=alpha, linewidth=0.5)
    elif np.any(fPAH > 0):
        ax1.plot(wavemod, fCont+fPAH, label='PAHs', color='purple', alpha=alpha)

    ax11 = ax1.twinx()
    ax11.plot(wavemod, pahext, linestyle='dashed', color='gray', alpha=0.5, linewidth=0.6)
    ax11.set_ylim(0, 1.1)
    ax11.set_ylabel('Attenuation of Warm dust and PAH components', fontsize=14)
    ax11.tick_params(axis='y', labelsize=10)
    #ax11.tick_params(direction='in', which='both', length=4, width=0.8, right=True)

    # Find the flux min(1 um, 5 um) to set ylim in the plot
    #if (len(flux[(wave > 0.9) & (wave < 1.2)]) != 0) & (len(flux[(wave > 4.8) & (wave < 5.2)]) != 0): 
    #    min_flux_at1= np.nanmean(flux[(wave > 0.9) & (wave < 1.2)])
    #    min_flux_at5 = np.nanmean(flux[(wave > 4.8) & (wave < 5.2)])
    #    
    #    min_flux = np.min([min_flux_at1, min_flux_at5])
    #
    #elif len(flux[(wave > 4.8) & (wave < 5.2)]) != 0: 
    #    min_flux = np.nanmean(flux[(wave > 4.8) & (wave < 5.2)])
    #
    #elif len(flux[(wave > 0.9) & (wave < 1.2)]) != 0:
    #    min_flux = flux[np.nanargmin(wave[(wave > 0.9) & (wave < 1.2)])]
    #
    #else:   
    #    min_flux = flux[np.nanargmin(wave)]

    min_flux = np.nanmin(flux[np.r_[0:100,-100:len(flux)]])
    max_flux = np.nanmax(flux[np.r_[0:100,-100:len(flux)]])

    ax1.legend(loc='lower right')
    ax1.tick_params(direction='in', which='both', length=6, width=1, top=True)
    ax1.tick_params(axis='x', labelsize=0)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(bottom=0.1*np.nanmin(min_flux), top=2.*np.nanmax(max_flux))
    #ax1.set_xlim(left=2.5, right=36)
    ax1.set_xlim(np.nanmin(wave)/1.2, 1.2*np.nanmax(wave))
    ax1.set_ylabel(r'$f_\nu$ (Jy)', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    #ax1.axvline(9.7, linestyle='--', alpha=0.2)

    xlabs = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30]
    ax1.set_xticks(xlabs[(np.where(xlabs > np.nanmin(wave))[0][0]):(np.where(xlabs < np.nanmax(wave))[0][-1]+1)])
    ax1.xaxis.set_major_formatter(ScalarFormatter())

    interpMod = np.interp(wave, comps['wave'], fMod)
    res = (flux-interpMod) / flux * 100 # in percentage
    std = np.nanstd(res)
    ax2.plot(wave, res, color='k', linewidth=1)
    #ax2.plot(wave, (flux-interpMod)/sigma, color='k')
    ax2.axhline(0., color='k', linestyle='--')
    ax2.tick_params(direction='in', which='both', length=6, width=1,  right=True, top=True)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.set_ylim(-4*std, 4*std)
    #ax2.set_ylim(bottom=-4, top=4)
    ax2.set_xlabel(r'$\lambda_{rest}$ $(\mu m)$', fontsize=14)
    #ax2.set_ylabel(r'$f^{data}_\nu - f^{tot}_\nu$ $(\sigma)$', fontsize=14)
    ax2.set_ylabel('Residuals (%)', fontsize=14)

    #ax1.set_zorder(100)

    #ax1.set_title('CAFE Spectrum Decomposition', fontsize=16)
    plt.subplots_adjust(hspace=0)
    
    # Use black as the patch backgound 
    # fig.patch.set_facecolor('k')
    # ax1.xaxis.label.set_color('w')
    # ax1.yaxis.label.set_color('w')
    # ax1.tick_params(direction='out', which='both', axis='both', colors='w')
    # ax11.tick_params(direction='out', which='both', length=4, width=0.8, right=True, colors='w')
    # ax11.yaxis.label.set_color('w')
    # ax2.xaxis.label.set_color('w')
    # ax2.yaxis.label.set_color('w')
    # #ax11.tick_params(axis='both', colors='w')
    # ax2.tick_params(direction='out', which='both', axis='both', colors='w')
    
    return fig


def corrmatrixplot(params, outpath='', obj='', tag=''):
    ''' Plots the correlation matrix for the parameters

    Arguments:
    params -- fit lm parameters object. Fit must have been successful for
    errors to be calculted - note that if you turn off error checking,
    this is not guarenteed, and may cause crashes. 
    
    Keyword Arguments:
    outpath -- directory to save plots to (default '')
    obj -- target name to use in filename (default '')
    tag -- string to append to the filename (default '')

    Returns: None, saves the figure to the global outpath 
    '''
    keys = params.valuesdict().keys()
    correls = []
    varnames = []
    count = 0
    for key in keys:
        if params[key].vary:
            correl = list(params[key].correl.values())
            correl.insert(count, 1.) # Need to manually add in the trace
            correl = np.abs(correl)
            correls.append(correl)
            varnames.append(key)
            count+=1
    correls = np.asarray(correls)

    fig = plt.figure(figsize=(12*np.size(correls[0])/20.,12*np.size(correls[0])/20.))
    im = plt.imshow(correls)
    im.axes.set_xticks(range(count))
    im.axes.set_xticklabels(varnames)
    im.axes.tick_params(axis='x', labelrotation=90 )
    im.axes.set_yticks(range(count))
    im.axes.set_yticklabels(varnames)
    im.axes.tick_params(axis='both', which='both', labelsize=14, length=0)
    fig.colorbar(mappable=im, shrink=0.75)
    im.axes.set_title('Correlation Matrix', fontsize=16)
    plt.savefig(outpath+obj+'_correlations'+tag+'.pdf', bbox_inches='tight')


def fracConPlot(comps, outpath='', obj=''):
    ''' Plots each components' fractional contribution to the SED

    Note that this plot interacts weirdly with the 2D version sometimes, so it's
    not generated by default.

    Arguments:
    comps -- dictionary of component fluxes

    Keyword Arguments:
    outpath -- directory to save plots to (default '')
    obj -- target name to use in filename (default '')

    Returns: None, saves to global outpath

    '''
    fCont =  comps['fCIR'] + comps['fCLD'] + comps['fCOO'] + comps['fWRM'] + comps['fHOT'] + comps['fSTB'] + comps['fSTR'] + comps['fDSK']
    
    for key in comps.keys():
        if key not in ['wave', 'fLIN', 'fPAH', 'fSRC', 'fDST'] and comps[key].sum() > 1e-14:
            plt.loglog(comps['wave'], comps[key]/fCont, label=key)
    plt.fill_betweenx([1e-2, 1], 2.5, x2=38., color='k', alpha=0.3, label='CAFE coverage')
    plt.xlabel(r'$\lambda_{rest}$ $(\mu m)$', fontsize=14)
    plt.ylabel('Fraction of Continuum', fontsize=14)
    plt.xlim(left=1., right=1e3)
    plt.ylim(bottom=1e-2, top=1)
    plt.legend()
    plt.savefig(outpath+obj+'_contfrac.pdf', bbox_inches='tight')


def spatplot(data, name, outpath='', obj='', contour=False, givefig=True):
    ''' Plot array, assuming it's a spatial image

    Plots a spatial map of the array with the specified title. Defaults to image, can be switched
    using the contour keyword. Will eventually want to be able to specify axes in physical
    units, but need information on detector and direction to NCP.

    Arguments:
    data -- 2D spatial array of values
    name -- name of parameter to show in plot title

    Keyword Arguments:
    outpath -- directory to save plots to (default '')
    obj -- target name to use in filename (default '')
    contour -- Whether to make a contour plot (default False)
    givefig -- Whether to return figure or save it. Default false (saves)

    Returns: None, saves figure to output directory 
    '''
    fig = plt.figure()
    if not contour:
        im = plt.imshow(data, origin='lower', extent=(0,len(data),0,len(data[0])))  
    else:
        im = plt.contour(data)
    ax = im.axes
    ax.set_xlabel('Detector x axis pixel')
    ax.set_ylabel('Detector y axis pixel')
    fig.colorbar(mappable=im, shrink=0.75)
    ax.set_title(name+' Spatial Map')
    if not givefig:
        plt.savefig(outpath+obj+'_'+name+'_spatialmap.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def parplot(pars, name, outpath='', obj='', contour=False, givefig=False):
    ''' Plot spatial image of parameter

    Plots a spatial map of the parameter with name in pars. Defaults to image, can be switched
    using the contour keyword. Will eventually want to be able to specify axes in physical
    units, but need information on detector and direction to NCP.

    Arguments:
    pars -- lm parameters object with the parameter to plot
    name -- name of parameter to show

    Keyword Arguments:
    outpath -- directory to save plots to (default '')
    obj -- target name to use in filename (default '')
    contour -- Whether to make a contour plot (default False)
    givefig -- Whether to return figure or save it. Default false (saves)

    Returns: None, saves figure to output directory 
    '''
    data = pars[name]
    ### Collapse the wavelength axis if present - use trapezoid sum to integrate
    if data.ndim == 3:
        wave = pars['wave'][0][0]
        data = np.trapz(data, x=wave)
        # print(data.shape)
    fig = plt.figure()
    if not contour:
        im = plt.imshow(data, origin='lower', extent=(0,len(data),0,len(data[0])))  
    else:
        im = plt.contour(data)
    ax = im.axes
    ax.set_xlabel('Detector x axis pixel')
    ax.set_ylabel('Detector y axis pixel')
    fig.colorbar(mappable=im, shrink=0.75)
    ax.set_title(name.upper()+' Spatial Map')
    if not givefig:
        plt.savefig(outpath+obj+'_'+name+'_spatialmap.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        return fig



def errplot(pars, name, outpath='', obj='', contour=False):
    ''' Plot spatial image of parameter's estimated error

    Plots a spatial map of the parameter error with name in pars. Defaults to image, can be switched
    using the contour keyword. Will eventually want to be able to specify axes in physical
    units, but need information on detector and direction to NCP.

    Arguments:
    pars -- lm parameters object with the parameter to plot
    name -- name of parameter to show

    Keyword Arguments:
    outpath -- directory to save plots to (default '')
    obj -- target name to use in filename (default '')
    contour -- Whether to make a contour plot (default False)

    Returns: None, saves figure to output directory 
    '''
    data = pars[name].stderr
    ### Collapse the wavelength axis if present - use trapezoid sum to integrate
    if data.ndim == 3:
        wave = pars['wave'][0][0]
        data = np.trapz(data, x=wave)
        # print(data.shape)
    fig = plt.figure()
    if not contour:
        im = plt.imshow(data, origin='lower', extent=(0,len(data),0,len(data[0])))  
    else:
        im = plt.contour(data)
    ax = im.axes
    ax.set_xlabel('Detector x axis pixel')
    ax.set_ylabel('Detector y axis pixel')
    fig.colorbar(mappable=im, shrink=0.75)
    ax.set_title(name.upper()+'Error Spatial Map')
    plt.savefig(outpath+obj+'_'+name+'_error_spatialmap.pdf', bbox_inches='tight')
    plt.close(fig)



def check_fit_pars(self, wave, flux_unc, fit_params, params, old_params, errors_exist):

    acceptFit = True
    if errors_exist is False: print('No errors retuned') ; acceptFit = False

    for par in fit_params:
        if par[0] not in ['g', 'd', 'o']:
            # -----------------------------------
            # Check if continuum param hit bounds
            # -----------------------------------
             ### Check for parameters at a boundary or initial value (not onion-related)
            if fit_params[par].vary and par not in ['HOT_WRM', 'WRM_COO']:
                if np.allclose(fit_params[par].max, fit_params[par].value, atol=0., rtol=1e-5): # relative comparison (default rtol=1e-5)
                    print(fit_params[par], 'at upper bound, fixing to ', params[par].max)                    
                    # Change input parameter to upper bound
                    params[par].value = params[par].max
                    # Force the parameter fixed
                    params[par].vary = False
                    #acceptFit = False
                    #logFile.write(par + ' at upper bound\n')
                   
                if np.allclose(fit_params[par].min, fit_params[par].value, atol=1e-5, rtol=0.): # absolute comparison
                    print(fit_params[par], 'at lower bound, fixing to', params[par].min)
                    # Change input parameter to lower bound
                    params[par].value = params[par].min
                    # Force the parameter fixed
                    params[par].vary = False
                    #acceptFit = False
                    #logFile.write(par + ' at lower bound\n')
                    
                    ### If a flux component hits the lower bound, need to also fix
                    ### related parameters or they wind up in an infinite loop
                    if par in ['CIR_FLX', 'CLD_FLX']:
                        params[par[:3]+'_TMP'].vary = False
                    if par in ['COO_FLX', 'WRM_FLX', 'HOT_FLX']:
                        params[par[:3]+'_TMP'].vary = False
                        params[par[:3]+'_TAU'].vary = False
                        params[par[:3]+'_MIX'].vary = False
                        params[par[:3]+'_COV'].vary = False
                    if par in ['STR_FLX', 'STB_FLX']:
                        params[par[:3]+'_TAU'].vary = False
                        params[par[:3]+'_MIX'].vary = False
                        params[par[:3]+'_COV'].vary = False
                    if par in ['DSK_FLX']: # DSK_MIX not exist
                        params[par[:3]+'_TAU'].vary = False
                        params[par[:3]+'_COV'].vary = False
                        
                # Check if the values didn't move from the initial value
                if errors_exist is False:
                    if np.allclose(fit_params[par].value, old_params[par].value, atol=0., rtol=1e-5): # relative comparison
                        print(fit_params[par], 'at initial value',old_params[par].value,'. Fixing value.')
                        params[par].vary = False
                        #acceptFit = False
                        #logFile.write(par + 'at initial value\n')
                    
            ### Deal with onion parameters hitting a bound
            elif fit_params[par].vary and par in ['HOT_WRM', 'WRM_COO']:
                if np.allclose(fit_params[par].min, fit_params[par].value, atol=0., rtol=1e-5) \
                   or np.allclose(fit_params[par].value, self.inpars['CONTINUA INITIAL VALUES AND OPTIONS'][par][0], atol=0., rtol=1e-5): # relative comparisons
                    raise RuntimeError('Onion parameter '+par+' failed, rerunning fit without Onion constrains.')
                    #logFile.write(par+' failed, turning off onion\n')
                    self.inopts['SWITCHES']['ONION'] = False
                    #acceptFit = False

        else:
            # ---------------------------------
            # Check if lines or PAHs hit bounds
            # ---------------------------------
            # Look at line parameters - note we're looking at fit_params, but updating params
            if fit_params[par].vary:
                if np.allclose(fit_params[par].max, fit_params[par].value, atol=0., rtol=1e-5): # relative comparison
                    print(fit_params[par], 'at upper bound, fixing to', fit_params[par].max)
                    #logFile.write(par + ' at bound\n')
                    params[par].vary = False
                    params[par].value = fit_params[par].max
                    ### If one parameter hits an infinite loop, also need to update others
                    if par[-4:] == 'Wave':
                        base = par.split('_Wave')[0]
                        params[base+'_Peak'].vary = False
                        params[base+'_Gamma'].vary = False
                    elif par[-4:] == 'Peak':
                        base = par.split('_Peak')[0]
                        params[base+'_Wave'].vary = False
                        params[base+'_Gamma'].vary = False
                if par[-4:] == 'Peak':
                    if np.allclose(fit_params[par].min, fit_params[par].value, atol=1e-5, rtol=0.): # absolute comparison
                        print(fit_params[par], 'at lower bound, fixing to', fit_params[par].min)
                        #logFile.write(par + ' at bound\n')
                        params[par].vary = False
                        params[par].value = fit_params[par].min
                        ### If one parameter hits an infinite loop, also need to update others
                        base = par.split('_Peak')[0]
                        params[base+'_Wave'].vary = False
                        params[base+'_Gamma'].vary = False                    
                else:
                    if np.allclose(fit_params[par].min, fit_params[par].value, atol=0., rtol=1e-5): # relative comparison
                        print(fit_params[par], 'at lower bound, fixing to', fit_params[par].min)
                        #logFile.write(par + ' at bound\n')
                        params[par].vary = False
                        params[par].value = fit_params[par].min
                        ### If one parameter hits an infinite loop, also need to update others
                        if par[-4:] == 'Wave':
                            base = par.split('_Wave')[0]
                            params[base+'_Peak'].vary = False
                            params[base+'_Gamma'].vary = False

                # Check if the values didn't move from the initial value
                if errors_exist is False:
                    if np.allclose(fit_params[par].value, old_params[par].value, atol=0., rtol=1e-5): # relative comparison
                        print(fit_params[par], 'at initial value',old_params[par].value,'. Fixing value.')
                        params[par].vary = False
                        #acceptFit = False
                        #logFile.write(par + 'at initial value\n')

    ### If errors are calculated, run additional checks but accept the fit
    if errors_exist:
        acceptFit = True
        #conttaupars = list(filter(lambda p: 'TAU' == p[-3:] or 'FLX' == p[-3:], fit_params))
        conttaupars = list(filter(lambda p: 'g' != p[0] and 'd' != p[0] and 'o' != p[0], fit_params))
        for conttaupar in conttaupars:
            if fit_params[conttaupar].vary and fit_params[conttaupar].value > 0. \
               and fit_params[conttaupar].stderr/fit_params[conttaupar].value > self.inopts['FIT OPTIONS']['REL_ERR_C_FEAT']:
                #acceptFit = False
                print(fit_params[conttaupar], 'unconstrained at',fit_params[conttaupar].stderr/fit_params[conttaupar].value*100.,'% error. Fixing value.')
                params[conttaupar].vary = False
                
        # Check for the emission features to be within the accepted relative error limits 
        featpars = list(filter(lambda p: 'g' == p[0] or 'd' == p[0] or 'o' == p[0], fit_params))
        for featpar in featpars:
            badfeatpar = False
            fnames = featpar.split('_')
            if fnames[-1] == 'Peak':
                base = '_'.join(fnames[:-1])
                if fit_params[featpar].value > 0.:
                    if fit_params[base+'_Wave'].stderr / fit_params[base+'_Wave'].value > self.inopts['FIT OPTIONS']['REL_ERR_'+featpar[0].upper()+'_W0']:
                        print(fit_params[base+'_Wave'], 'unconstrained, fixing entire feature') ; badfeatpar=True
                    if fit_params[base+'_Gamma'].stderr / fit_params[base+'_Gamma'].value > self.inopts['FIT OPTIONS']['REL_ERR_'+featpar[0].upper()+'_SIG']:
                        print(fit_params[base+'_Gamma'], 'unconstrained, fixing entire feature') ; badfeatpar=True
                    if fit_params[featpar].stderr / fit_params[featpar].value > self.inopts['FIT OPTIONS']['REL_ERR_'+featpar[0].upper()+'_AMP']:
                        print(fit_params[featpar], 'unconstrained, fixing entire feature') ; badfeatpar=True
                    #for featp in list(filter(lambda p: base in p, gfeatpars)): print(fit_params[featp], ' unconstrained')
                    if badfeatpar is True:
                        #acceptFit = False
                        params[base+'_Wave'].vary = False
                        params[base+'_Gamma'].vary = False
                        params[featpar].vary = False
                    # If the feature peak is lower than the 0.5-sigma uncertainty, set it to 0
                    feat_fluxunc = np.interp(fit_params[base+'_Wave'].value, wave, flux_unc)
                    if fit_params[featpar].value < feat_fluxunc*0.5:
                        print(fit_params[featpar], 'lower than uncertainty', feat_fluxunc, ', fixing to 0.0')
                        params[featpar].value = 0.
                        params[featpar].vary = False
                        
    return acceptFit

