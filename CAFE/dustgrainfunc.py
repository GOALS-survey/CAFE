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

import CAFE
from CAFE.mathfunc import spline, intTab

#from pycafelib_cube import *
from CAFE.component_model import pah_drude, gauss_prof, drude_prof, drude_int_fluxes
from CAFE.sourceSED import planck, sourceSED_ISRF, sourceSED_AGN, sourceSED_SB, sourceSED, load_opacity

import ipdb

#from pycafelib_cube import drude_prof

# class grain_base:
#     # Define grain radii 
#     minRad=1e-3
#     maxRad=10.
#     numRad=81
#     rad = np.logspace(np.log10(minRad),np.log10(maxRad),num=numRad) # radii sampling

#     # Define blackbody matching ambient radiation field
#     T_bb = np.logspace(np.log10(3.), np.log10(1750.), num=30)

#     def __init__(self, rad=None, T_bb=None):

#         self.rad = rad


#     def func1():
#         rad = self.rad

#################################
### Dust Grain Functions      ###
#################################
def grain_radii(numRad=81, minRad=1e-3, maxRad=10.):
    ''' Create array of dust grain sizes

    This is a direct port of jam_grainradii.pro 
    Verified to work same as IDL version 8/18/20

    Keyword Arguments:
    numRad -- the number of output points (default 81)
    minRad -- minimum grain radius in micron (default 1e-3)
    maxRad -- max grain radius in micron (default 10.)

    Returns: Array of grain radii, log-spaced
    '''
    radii = np.geomspace(minRad,maxRad,num=numRad)
    dlogRadii = np.log(radii[1]/radii[0])
    dlogVSG = np.log(minRad/3.5e-4)
    numRadVSG = np.ceil(dlogVSG/dlogRadii)
    numRad += numRadVSG
    minRad = np.exp(np.log(minRad) - numRadVSG*dlogRadii)
    radii = np.geomspace(minRad, maxRad, num=int(np.round(numRad)))
    
    return radii

def grain_crosssection(wave, rad, scaleSIL, tablePath):
#def grain_crosssection(wave, rad, scaleSIL):
    ''' Compute dust grain cross sections

    Direct port of jam_graincrosssection. Note that the python interpolation algorithms
    return slightly different results for extrapolations than their IDL equivalents, which
    may lead to slightly different results in some cases. Doesn't appear to ever be more
    than about 2 percent.
    Verified 8/24/20

    Arguments:
    wave -- array of wavelengths over which to compute cross sections
    rad -- array of dust grain sizes to compute cross sections for
    scaleSIL -- Array to use in scaling silicate features between Draine/OHMc

    Keyword Arguments:
    tablePath -- directory for grain opacity tables

    Returns: tuple of dicts, first for absorption and second for extinction. Each
    dictionary contains grids of wavelength, grain size, and opacities for silicate,
    graphite neutral, and ionized species
    '''

    # Read in Cross section profile from silicate and graphite
    cAbsSilTab = np.loadtxt(tablePath+'c_abs.sil.txt', comments=';') # with shape (241, 91)
    cAbsGraTab = np.loadtxt(tablePath+'c_abs.gra.txt', comments=';')
    cExtSilTab = np.loadtxt(tablePath+'c_ext.sil.txt', comments=';')
    cExtGraTab = np.loadtxt(tablePath+'c_ext.gra.txt', comments=';')

    # The grain cross-sections use their own Tabulated wavelength array,
    # and then they are interpolated to the supplied wavelength array
    waveTab = np.logspace(-3, 3, num=241) # This is hardcoded from the table header
    aTab = grain_radii() # Also taken from table header???

    logWaveTab = np.log(waveTab)
    logaTab = np.log(aTab)

    logWave = np.log(wave)
    loga = np.log(rad)

    ### If necessary, interpolate onto input grid
    ### Note - Python interpolation routines differ slightly from the IDL versions,
    ### can lead to some issues. Difference appears to be larger when extrapolating
    ### off-grid, up to about 4 percent, typically more like 0.05 percent.
    if not (rad.shape==aTab.shape and np.allclose(rad, aTab)):
        log_cAbsSilTab = np.log(cAbsSilTab)
        log_cAbsGraTab = np.log(cAbsGraTab)
        log_cExtSilTab = np.log(cExtSilTab)
        log_cExtGraTab = np.log(cExtGraTab)
        log_cAbsSil = np.empty((logWaveTab.size, loga.size))
        log_cAbsGra = np.empty((logWaveTab.size, loga.size))
        log_cExtSil = np.empty((logWaveTab.size, loga.size))
        log_cExtGra = np.empty((logWaveTab.size, loga.size))
        for i in range(waveTab.size):
            f1 = interp1d(logaTab, log_cAbsSilTab[i], fill_value='extrapolate')
            f2 = interp1d(logaTab, log_cAbsGraTab[i], fill_value='extrapolate')
            f3 = interp1d(logaTab, log_cExtSilTab[i], fill_value='extrapolate')
            f4 = interp1d(logaTab, log_cExtGraTab[i], fill_value='extrapolate')
            log_cAbsSil[i] = f1(loga)
            log_cAbsGra[i] = f2(loga)
            log_cExtSil[i] = f3(loga)
            log_cExtGra[i] = f4(loga)
    else:
        cAbsSil = cAbsSilTab
        cAbsGra = cAbsGraTab
        cExtSil = cExtSilTab
        cExtGra = cExtGraTab

    if not (wave.shape==waveTab.shape and np.allclose(wave, waveTab)):
        # Take logs
        log_cAbsSilTab = np.log(cAbsSil)
        log_cAbsGraTab = np.log(cAbsGra)
        log_cExtSilTab = np.log(cExtSil)
        log_cExtGraTab = np.log(cExtGra)
        log_cAbsSil = np.empty((logWave.size, loga.size))
        log_cAbsGra = np.empty((logWave.size, loga.size))
        log_cExtSil = np.empty((logWave.size, loga.size))
        log_cExtGra = np.empty((logWave.size, loga.size))
        for i in range(rad.size):
            f1 = splrep(logWaveTab, log_cAbsSilTab[:,i])
            f2 = splrep(logWaveTab, log_cAbsGraTab[:,i])
            f3 = splrep(logWaveTab, log_cExtSilTab[:,i])
            f4 = splrep(logWaveTab, log_cExtGraTab[:,i])
            log_cAbsSil[:,i] = splev(logWave, f1)
            log_cAbsGra[:,i] = splev(logWave, f2)
            log_cExtSil[:,i] = splev(logWave, f3)
            log_cExtGra[:,i] = splev(logWave, f4)
        cAbsSil = np.exp(log_cAbsSil)
        cAbsGra = np.exp(log_cAbsGra)
        cExtSil = np.exp(log_cExtSil)
        cExtGra = np.exp(log_cExtGra)

    # Scale silicate opacities with scaling factor at each wavelength
    # Hardcode scaleSIL as ohmc for temporary use <-- Not any more. Now scaleSIL needs to be passed
    #scaleSIL = np.genfromtxt(tablePath+'ohmc_scale_upsampled.txt', comments=';')
    scale = np.interp(wave, scaleSIL[:,0], scaleSIL[:,1], left=1, right=1)
    for i in range(rad.size):
        cAbsSil[:,i] *= scale
        cExtSil[:,i] *= scale

    # Calc PAH absorption cross sections
    cCont, cFeat = pah_crosssection(wave, rad, cAbsGra)
    # Carbonaceous grain weighting func
    q_Gra = 0.01
    a_Xi  = 50.
    xi_PAH = (1. - q_Gra) * (a_Xi*1e-4/rad)**3
    xi_PAH[(a_Xi*1e-4/rad)**3 > 1] = 1. - q_Gra
    xi_PAH = np.tile(xi_PAH, (wave.size,1))
    cAbsGraCarb = (1. - xi_PAH) * cAbsGra
    cExtGraCarb = (1. - xi_PAH) * cExtGra 

    # Carbonaceous cross sections
    cAbsNeuCont = xi_PAH * cCont[0] + cAbsGraCarb 
    cAbsIonCont = xi_PAH * cCont[1] + cAbsGraCarb
    cExtNeuCont = xi_PAH * cCont[0] + cExtGraCarb
    cExtIonCont = xi_PAH * cCont[1] + cExtGraCarb
    cNeuFeat = xi_PAH * cFeat[0]
    cIonFeat = xi_PAH * cFeat[1]

    ### Output structure to replicate IDL version
    cExt = {'wave':wave, 'a':rad, 'Sil':cExtSil, 'Gra':cExtGra, 'CarbNeu':[cExtNeuCont, cNeuFeat], 'CarbIon':[cExtIonCont, cIonFeat]}
    cAbs = {'wave':wave, 'a':rad, 'Sil':cAbsSil, 'Gra':cAbsGra, 'CarbNeu':[cAbsNeuCont, cNeuFeat], 'CarbIon':[cAbsIonCont, cIonFeat]}

    return cAbs, cExt


def grainEQTemp(a, T_bb, sourceType, tablePath, scaleSIL=None, cAbs=None, TTable=None):
    """
    Compute equilibrium temperatures of grains in radiation field as function of size (T_eq)

    Port of jam_graineqtemp.pro. Differences from IDL version due to different inteprolation routines,
    should be ~2 percent or less, with edges being the worst
    Verified to match well enough 9/7/20

    Parameters
    ----------
    a -- array of grain sizes
    T_bb -- array of blackbody temperature of ambient radiation field
    cAbs -- absorption cross-sections for the grains (default None)
    sourceType -- object producing ambient radiation field (default None)
    TTable -- Array of temperatures to compute over (default None)
    scaleSIL -- Scaling between Draine and OHMc silicates (default None)

    Returns
    ------- 
        Dictionary of grain sizes, ambient radiation field temperatures,
    and equilibrium temperatures of silcate and carbonaceous grains. 
    Also returns TTable, which is a holdover from IDL and possibly unnecessary (TTable deleted by TL 4/5)
    """
    # This is added to match functionality of the IDL version. I realize it's nuts.
    # This is only for obtaining opacity curves
    if np.sum(T_bb) == 0:
        TSil = np.squeeze(np.zeros((a.size, T_bb.size)))
        TCarb = np.squeeze(np.zeros((a.size, T_bb.size))) 

        return {'rad':a, 'T_bb':T_bb, 'Sil':TSil, 'Carb':TCarb}, None, None

    # Initial inputs
    T = T_bb if TTable is None else TTable['T']
    
    T_bb = np.asarray(T_bb)
    logT = np.log(T)
    loga = np.log(a)
    wave = np.logspace(-3, 3, num=241)
    logWave = np.log(wave)

    # Get illuminating SED
    if type(sourceType)==str: # Why type(sourceType)==str?? This makes ISRF to be the only case.
        # Load an ISRF sed if unspecified
        #sourceType = 'ISRF'
        if TTable is None: print('Generating profiles for',sourceType,'SED profile')
        sWave, sFlux = sourceSED(wave, sourceType, tablePath)
    else:
        ## Interpolate the specified SED onto the wave grid
        # This functionality is disabled and the user needs to select
        # from the sources available, calling them by their name
        ipdb.set_trace()

        sWave = sourceSED[0]
        sFlux = sourceSED[1]
        if np.sum(sWave-wave) > 1e-14:
            sFlux = np.exp(np.interp(logWave, np.log(sWave), np.log(sFlux)))

    # Calculate absorption cross sections
    if cAbs is None:
        cAbs, _ = grain_crosssection(wave, a, scaleSIL, tablePath)
    cSil = cAbs['Sil']
    cCarb = cAbs['Gra']

    # Eqn 12 from Marshall et al 2007, but with LDH and RHS reversed, solving for T_EQ
    # Integrate over wavelength for each radius and temperatue to calculate:
    # LHS = Int[C_Abs(wave,a)*B_Wave(wave,T), wave]
    dlogWave = np.log(wave[1]/wave[0])
    if TTable is None:
        intSil = np.tile(cSil, (np.size(T),1,1)).transpose([1,0,2])
        intCarb = np.tile(cCarb, (np.size(T),1,1)).transpose([1,0,2])
        bigPlanck = np.tile(planck(wave, T), (a.size,1,1)).transpose([2,1,0])
        intSil*=bigPlanck
        intCarb*=bigPlanck
        bigWave = np.tile(wave, (a.size, np.size(T), 1)).transpose([2,1,0])
        LHS_Sil = intTab(bigWave*intSil, dlogWave)
        LHS_Carb = intTab(bigWave*intCarb, dlogWave)
        TTable = {'a':a, 'T':T, 'Sil':LHS_Sil, 'Carb':LHS_Carb}
    else:
        LHS_Sil = TTable['Sil']
        LHS_Carb = TTable['Carb']
    
    # Integrate over wavelength for each grain size to calculate:
    # RHS = Int[f_Wave(wave) * C_Abs(wave,a), wave] / Int[f_Wave(wave), wave]
    bigsFlux = np.tile(sFlux, (a.size, 1)).transpose()
    intSil = cSil * bigsFlux
    intCarb = cCarb * bigsFlux
    bigWave = np.tile(wave, (a.size, 1)).transpose()
    sFluxTot = intTab(wave*sFlux, dlogWave)
    RHS_Sil = intTab(bigWave*intSil, dlogWave)/sFluxTot
    RHS_Carb = intTab(bigWave*intCarb, dlogWave)/sFluxTot

    # Create grain temp arrays
    logTSil = np.zeros((a.size, T_bb.size))
    logTCarb = np.zeros((a.size, T_bb.size))

    # Loop over each bb temp
    sigmaSB = 5.67e-5
    const = np.log(sigmaSB/np.pi * T_bb**4)
    logRHSSil = np.log(RHS_Sil)
    logRHSCarb = np.log(RHS_Carb)
    logLHSSil = np.log(LHS_Sil)
    logLHSCarb = np.log(LHS_Carb)
    for i in range(T_bb.size):
        if np.size(const) == 1:
            tlogRHSSil = const + logRHSSil
            tlogRHSCarb = const + logRHSCarb
        else:
            tlogRHSSil = const[i] + logRHSSil
            tlogRHSCarb = const[i] + logRHSCarb
        for j in range(a.size):
            f1 = interp1d(logLHSSil[:,j], logT, fill_value='extrapolate')
            f2 = interp1d(logLHSCarb[:,j], logT, fill_value='extrapolate')
            logTSil[j,i] = f1(tlogRHSSil[j])
            logTCarb[j,i] = f2(tlogRHSCarb[j])
    TSil = np.squeeze(np.exp(logTSil))
    TCarb = np.squeeze(np.exp(logTCarb))

    # Return
    if np.size(TSil) == 1: TSil = float(TSil)
    if np.size(TCarb) == 1: TCarb = float(TCarb)

    #return {'rad':a, 'T':T_bb, 'Sil':TSil, 'Carb':TCarb}
    return {'rad':a, 'T':T_bb, 'Sil':TSil, 'Carb':TCarb}, cAbs, TTable # old


def grainSizeDF(rad, T_bb, sourceType, tablePath, model='WD01-RV31', dndaTab=None, cutoff=None, scaleSIL=None, cAbs=None, TTable=None):
    ''' Calculate grain size distribution of graphitic and silicate dusts

    Port of jam_grainsizedf. Appears to match IDL functionality 8/26/20
    Note that testing has only been done for T_bb = 0, issues may crop
    up at different T_bb, espeically with grainEQTemp

    Arguments:
    rad -- array of grain radii in microns
    T_bb -- blackbody matching ambient radiation field
    sourceType -- Type of input SED source to be passed to grainEQT

    Keyword Arguments:
    model -- Which grain size distribution model to use (default 'WD01-RV31')
    dndaTab -- Table of grain size distribution parameters (default None0)
    cutoff -- Whether to truncate the grain size distribution. Options are
    'big', 'small', or None (default)

    Returns: Dict of grain sizes, blackbody temperatures, and size distributions
    for both silicate and carbonaceous grains
    '''
    # Initial setup
    a = grain_radii()
    loga = np.log(a)

    ### Load parameter values
    if dndaTab is None:
        dndaTab = np.genfromtxt(tablePath+'grainsizedf_params.txt', comments=';', dtype='str')
    names = dndaTab[:,0]
    row = 0
    for i in range(len(names)):
        if names[i] == model: 
            row = i
    dndaTab = dndaTab[:,1:].astype('float')
    bC = dndaTab[row][0]
    aMin = dndaTab[row][1]
    aMax = dndaTab[row][2]
    cGra = dndaTab[row][3]
    cSil = dndaTab[row][4]
    alpGra = dndaTab[row][5]
    alpSil = dndaTab[row][6]
    atGra = dndaTab[row][7]
    atSil = dndaTab[row][8]
    acGra = dndaTab[row][9]
    acSil = dndaTab[row][10]
    betGra = dndaTab[row][11]
    betSil = dndaTab[row][12]
    gamMax = dndaTab[row][13]

    # Calculate grain-size DF without large and small grain-size cutoffs
    dndaGra0 = (cGra/a) * np.power(a/atGra, -1.0*alpGra)
    if betGra > 0: dndaGra0*=(1+betGra*a/atGra)
    else: dndaGra0/=(1-betGra*a/atGra)
    dndaSil0 = (cSil/a) * np.power(a/atSil, -1.0*alpSil)
    if betSil > 0: dndaSil0*=(1+betSil*a/atSil) 
    else: dndaSil0/=(1-betSil*a/atSil) 

    # Apply large grain-size cutoff
    dndaGra0[a > atGra]*=np.exp(-1.0 * np.power(((a[a > atGra] - atGra)/acGra),gamMax))
    dndaSil0[a > atSil]*=np.exp(-1.0 * np.power(((a[a > atSil] - atSil)/acSil),gamMax))
    # Apply absolute min and max grain size cutoffs
    dndaGra0[a < 1e-4*aMin] = 0.
    dndaSil0[a < 1e-4*aMin] = 0.
    dndaGra0[a > aMax] = 0.
    dndaSil0[a > aMax] = 0.

    # Calculate VSG distributio function
    mC = 1.99e-23                  # Carbon atmoic mass [g]
    a0i = np.asarray([3.5, 30.])   # Log-normal peak locations [A]
    sigma = 0.4                    # Log-normal width
    bCi = np.asarray([0.75, 0.25]) # Fraction of bC in each peak
    densGra = 2.24                 # Graphide density in [g cm-3]
    cmPerA = 1e-8
    umPerA = 1e-4
    dndaVSG0 = 0.*dndaGra0
    if bC > 0:
        B = 3/np.sqrt(2*np.pi)**3 * np.exp(-4.5*sigma**2) / (densGra*(cmPerA*a0i)**3 * sigma) * \
            (1e-6*bC)*bCi*mC / (1 + erf(3*sigma/np.sqrt(2) + np.sqrt(2)*np.log(a0i/3.5)/sigma))
        dndaVSG0 = (B[0]/a) * np.exp(-0.5*(np.log(a/(umPerA*a0i[0]))/sigma)**2) + \
                   (B[1]/a) * np.exp(-0.5*(np.log(a/(umPerA*a0i[1]))/sigma)**2)
        dndaVSG0[a < 1e-4*aMin] = 0.
        dndaVSG0[a > aMax] = 0.
    # Calculate carbonaceous grain-size DF
    dndaCarb0 = dndaGra0 + dndaVSG0
    # Remove sublimated grains
    tSubSil = 1400.
    sSubCar = 1750.
    
    # Initialize empty grid
    dndaSil = np.zeros((a.size, T_bb.size))
    dndaCarb = np.zeros((a.size, T_bb.size))

    #tEQ, TTable = grainEQTemp(a, T_bb, TTable=TTable)
    ## Create grid with axes dndaSil/dndaCarb and T_bb 
    #for i in range(T_bb.size):
    #    dndaSil[:,i] = dndaSil0
    #    dndaCarb[:,i] = dndaCarb0
    #
    ## No grains beyond sublimation temperature
    #dndaSil[tEQ['Sil'] > tSubSil] = 0.
    #dndaCarb[tEQ['Carb'] > tSubSil] = 0.

    tEQ, cAbs, TTable = grainEQTemp(a, T_bb, sourceType, tablePath, scaleSIL=scaleSIL, cAbs=cAbs, TTable=TTable)

    for i in range(T_bb.size):
        dndaSil[:,i] = dndaSil0
        dndaCarb[:,i] = dndaCarb0
        ### If something goes wrong, check here first
        if T_bb.size == 1:
            dndaSil[tEQ['Sil'] > tSubSil,i] = 0.
            dndaCarb[tEQ['Carb'] > tSubSil,i] = 0.
        else:
            dndaSil[tEQ['Sil'][:,i] > tSubSil,i] = 0.
            dndaCarb[tEQ['Carb'][:,i] > tSubSil,i] = 0.

    # Apply big and small cutoffs
    # Possibly related to the inclusion of VSGs in the Draine models
    if cutoff is not None:
        if cutoff == 'big':
            dndaSil[a < 50e-4] = 0.
            dndaCarb[a < 50e-4] = 0.
        elif cutoff == 'small':
            dndaSil[a > 50e-4] = 0.
            dndaCarb[a > 50e-4] = 0.
        else: 
            raise ValueError

    return {'rad':rad, 'T_bb':T_bb, 'Sil':dndaSil, 'Carb':dndaCarb}

    

def grain_opacity(wave, T_bb, scaleSIL, tablePath,
                  cAbs_wR=None, cExt_wR=None, 
                  dnda=None, noPAH=True, gra=None, cutoff=None, 
                  kExt=True, fstTab=None, ensTab=None):
    ''' Calculate grain opacities at different blackbody temperatures

    Port of JAM_GrainOpacity IDL routine. Tested to reproduce IDL outputs 9/6/20

    Arguments:
    wave -- array of wavelengths over which to calculate the opacity
    T_bb -- blackbody temperature of ambient radiation field
    scaleSIL -- ???

    Keyword Arguments:
    tablePath -- directory where grain opacities are stored
    cAbs_wR -- grain absorption cross sections in reduced wavelengths (wR) (default None)
    cExt_wR -- grain scattering cross section in reduced wavelengths (wR) (default None)
    dnda -- grain size table. (default None)
    gra -- whether to use graphitic grain absorption/extinction (default None)
    noPAH -- whether to include PAH contribution (default True)
    cutoff -- whether/which grain size cutoff to impose (defailt None)
    kExt -- Whether to calculate scattering, in addition to absorption (default True)
    fstTab -- Silicate feature mass-opacities table (default None)
    ensTab --  Silicate feature mass-opacities table (default None)

    Returns: 
        Dictionary of wavelengths, blackbody temperature, and opacties by
        species (further divided into continuum/feature)
    '''
    T_bb = np.asarray(T_bb)
    if wave is None: sys.exit('No wavelength array provided for grain_emissivity calculations.')
    logWave = np.log(wave)
    #waveRed = np.geomspace(wave[0], wave[-1], num=1000) # wavelength upsampling
    waveRed = np.geomspace(min(1.,wave[0]), max(1e3,wave[-1]), num=1000)
    logWaveRed = np.log(waveRed)
    
    a = grain_radii()
    loga = np.log(a)
    dloga = np.log(a[1]/a[0])
    big_a = np.tile(a, (waveRed.size, T_bb.size))

    if cAbs_wR is None:
        cAbs_wR, cExt_wR = grain_crosssection(waveRed, a, scaleSIL, tablePath)

    if gra is None:
        cAbsCarb = cAbs_wR['Gra']
        cExtCarb = cExt_wR['Gra']
    elif noPAH: 
        cAbsCarb = 0.5 * (cAbs_wR['CarbNeu'][0]+cAbs_wR['CarbIon'][0])
        cExtCarb = 0.5 * (cExt_wR['CarbNeu'][0]+cExt_wR['CarbIon'][0])
    else:
        cAbsCarb = 0.5 * (cAbs_wR['CarbNeu'][0]+cAbs_wR['CarbNeu'][1]+cAbs_wR['CarbIon'][0]+cAbs_wR['CarbIon'][1])
        cExtCarb = 0.5 * (cExt_wR['CarbNeu'][0]+cExt_wR['CarbNeu'][1]+cExt_wR['CarbIon'][0]+cExt_wR['CarbIon'][1])

    if dnda is None:
        dnda = grainSizeDF(a, T_bb, None, tablePath, scaleSIL=scaleSIL, cutoff=cutoff) # None is given to imply there is no sourceType

    # Calculate total silicate opacities
    bigDndaSil = np.tile(dnda['Sil'][:,0], (waveRed.size, T_bb.size)) # Note: only consider dnda with T_bb=3 K. 
                                                                      #       Assuming Temp not impacting the grain size distribution 
    intSil = big_a * bigDndaSil
    kAbsSilTot = intTab(intSil.transpose()*cAbs_wR['Sil'].transpose(), dloga)

    if kExt:
        kExtSilTot = intTab(intSil.transpose()*cExt_wR['Sil'].transpose(), dloga)

    # Calculate carbonaceous opacities
    bigDndaCarb = np.tile(dnda['Carb'][:,0], (waveRed.size,T_bb.size)) # Note: only consider dnda with T_bb=3 K. 
    intCarb = big_a * bigDndaCarb
    kAbsCarb = intTab(intCarb.transpose()*cAbsCarb.transpose(), dloga)
    if kExt:
        kExtCarb = intTab(intCarb.transpose()*cExtCarb.transpose(), dloga)

    # Create index of wavelength points to fit silicate continuum to
    wBL = 1.
    wBH = 5.2
    wML = 13.8
    wMH = 14.5
    wAL = 1e2 
    wAH = 1e3 
    idx = ((waveRed > wBL) & (waveRed < wAH))
    idxCon = ((waveRed > wBL) & (waveRed < wBH) | \
              (waveRed > wML) & (waveRed < wMH) | \
              (waveRed > wAL) & (waveRed < wAH))

    # Calculate silicate continuum and amorphous features
    kSilCont = np.zeros((waveRed.size, T_bb.size))
    kAmoFeat = np.zeros((waveRed.size, T_bb.size))
    if idxCon.size > 0:
        for i in range(T_bb.size):
            if np.any(kAbsSilTot > 0.):
                # Calculate silicate continuum
                # This section is made to exactly match IDL, and could probably 
                # be streamlined. Note the use of np.copy(), as otherwise numpy
                # will assign multiple names to the same array object
                kSilTot = np.copy(kAbsSilTot)
                tSil = np.copy(kSilTot)
                spl = splrep(np.log(waveRed[idxCon]), np.log(kSilTot[idxCon]))
                tSil[idx] = np.exp(splev(np.log(waveRed[idx]), spl))
                # Subtract continuum to obtain 10/18 micron features
                tAmo = np.subtract(kSilTot, tSil)
                tAmo[tAmo < 0.] = 0.0
                tAmo[tAmo/tSil < 1e-3] = 0.0
                kAmoFeat[:,i] = tAmo
                # Subtract features from total to obtain final continuum
                kSilCont[:,i] = kSilTot - tAmo
            else:
                kSilCont[:,i] = np.zeros(waveRed.size)
                kAmoFeat[:,i] = np.zeros(waveRed.size)

    kAbsSilCont = np.squeeze(np.copy(kSilCont))
    kAbsAmoFeat = kAbsSilTot - kAbsSilCont
    if kExt:
        kExtAmoFeat = np.copy(kAbsAmoFeat)
        kExtSilCont = kExtSilTot - kExtAmoFeat

    # Obtain crystalline silicate feature mass-opacities
    if fstTab is None:
        fstTab = np.genfromtxt(tablePath+'k_abs.fst.txt', comments=';')
        # fstTab = np.genfromtxt(tablePath+'k_abs_upsampled.fst.txt', comments=';')
        fstWave = fstTab[:,0]
        fstK = fstTab[:,1]
    if ensTab is None:
        ensTab = np.genfromtxt(tablePath+'k_abs.ens.txt', comments=';')
        # ensTab = np.genfromtxt(tablePath+'k_abs_upsampled.ens.txt', comments=';')
        ensWave = ensTab[:,0]
        ensK = ensTab[:,1]

    kPerMassFst = np.zeros(waveRed.shape)
    idx = (waveRed >= np.min(fstWave)) & (waveRed <= np.max(fstWave))
    f1 = interp1d(np.log(fstWave), np.log(fstK), fill_value='extrapolate')
    kPerMassFst[idx] = np.exp(f1(logWaveRed[idx]))
    
    kPerMassEns = np.zeros(waveRed.shape)
    idx = (waveRed >= np.min(ensWave)) & (waveRed <= np.max(ensWave))
    f2 = interp1d(np.log(ensWave), np.log(ensK), fill_value='extrapolate')
    kPerMassEns[idx] = np.exp(f2(logWaveRed[idx]))

    # Calculate crystalline silicate feature opacities
    denSil = 3.5e-12
    bMass = np.tile(4*np.pi/3 * denSil * a**3, (waveRed.size, 1)).transpose()
    cFst0 = np.tile(kPerMassFst, (a.size,1))
    cEns0 = np.tile(kPerMassEns, (a.size,1))
    cFst = bMass*cFst0
    cEns = bMass*cEns0
    kAbsFstFeat = intTab(np.tile(cFst, (T_bb.size,1)), dloga)
    kAbsEnsFeat = intTab(np.tile(cEns, (T_bb.size,1)), dloga)
    if kExt:
        kExtFstFeat = kAbsFstFeat
        kExtEnsFeat = kAbsEnsFeat

    # Interpolate onto input wavegrid and output
    kAbsSilTot  = spline(logWave, logWaveRed, kAbsSilTot)
    kAbsSilCont = spline(logWave, logWaveRed, kAbsSilCont)
    kAbsAmoFeat = spline(logWave, logWaveRed, kAbsAmoFeat)
    kAbsFstFeat = spline(logWave, logWaveRed, kAbsFstFeat)
    kAbsEnsFeat = spline(logWave, logWaveRed, kAbsEnsFeat)
    kAbsCarb = spline(logWave, logWaveRed, kAbsCarb)
    kAbsOut = {'wave':wave, 't_bb':T_bb, 'SilCont':kAbsSilCont,
                'SilAmoTot':kAbsSilTot, 
                'SilAmoFeat':kAbsAmoFeat,
                'SilFstFeat':kAbsFstFeat,
                'SilEnsFeat':kAbsEnsFeat,
                'Carb':kAbsCarb}
    if kExt:
        kExtSilTot  = spline(logWave, logWaveRed, kExtSilTot)
        kExtSilCont = spline(logWave, logWaveRed, kExtSilCont)
        kExtAmoFeat = spline(logWave, logWaveRed, kExtAmoFeat)
        kExtFstFeat = spline(logWave, logWaveRed, kExtFstFeat)
        kExtEnsFeat = spline(logWave, logWaveRed, kExtEnsFeat)
        kExtCarb = spline(logWave, logWaveRed, kExtCarb)
        kExtOut = {'wave':wave, 't_bb':T_bb, 'SilCont':kExtSilCont,
                   'SilAmoTot':kExtSilTot, 
                   'SilAmoFeat':kExtAmoFeat,
                   'SilFstFeat':kExtFstFeat,
                   'SilEnsFeat':kExtEnsFeat,
                   'Carb':kExtCarb}
        return kAbsOut, kExtOut # absorption and extinction=absorption+scattering
    else: 
        return kAbsOut


def grain_emissivity(wave, T_bb, sourceType, scaleSIL, tablePath,
                     fstTab=None, ensTab=None, T_EQ=None):
    ''' Calculates grain emissivity at varying blackbody temperatures

    Port of jam_grainemissivity, verified to match IDL version 9/8/20, with some differences due to 
    Python handling interpolation differently in the carbonaceous grains

    Arguments:
    wave -- array of wavelengths over which to calculate the opacity
    T_bb -- blackbody temperature of ambient radiation field
    sourceType -- source of the ambient radiation field (default None)
    scaleSIL -- array of scaling factors between different silicate features (default None)

    Keyword Arguments:
    fstTab -- silicate feature mass-opacities table (default None)
    ensTab --  silicate feature mass-opacities table (default None)
    T_EQ -- array of grain equilibrium temperatures (default None)
    tablePath -- directory for emissivity tables (defaule 'tables/')

    Returns: Dictionary of wavelengths, blackbody temperature, and emissivities by
    species (further divided into continuum/feature)
    '''    
    
    # Making initial inputs
    if wave is None: 
        sys.exit('No wavelength array provided for grain_emissivity calculations.')
    logWave = np.log(wave)
    #waveRed = np.geomspace(wave[0], wave[-1], num=1000)
    waveRed = np.geomspace(min(1.,wave[0]), max(1e3,wave[-1]), num=1000)
    logWaveRed = np.log(waveRed)

    a = grain_radii()
    loga = np.log(a)
    dloga = np.log(a[1]/a[0])
    biga = np.tile(a, (waveRed.size, T_bb.size, 1)).transpose(2,0,1)

    # Get absorption cross sections
    # Reminder that since waveRed extends beyone 1000 um, the high wavelength
    # portions differs from the IDL output due to differences in interpolation algorithm
    cAbs_wR, _ = grain_crosssection(waveRed, a, scaleSIL, tablePath)

    cCarb = 0.5*(cAbs_wR['CarbNeu'][0] + cAbs_wR['CarbIon'][0]).transpose() 
    cSil = cAbs_wR['Sil'].transpose()
    # ---
    # Obtain grain temperatures
    if T_EQ is None:
        #T_EQ = grainEQTemp(a, T_bb, sourceType, scaleSIL, tablePath) # (TL) remove TTable
        T_EQ, cAbs, TTable = grainEQTemp(a, T_bb, sourceType, tablePath, scaleSIL=scaleSIL) #old
    # ---
    # Obtain grain-size DF
    dnda = grainSizeDF(a, T_bb, sourceType, tablePath, scaleSIL=scaleSIL, cAbs=cAbs, TTable=TTable) #, cutoff='big' Removed by TDS

    # Planck functions - need to figure out transposes
    planckSil = np.zeros((waveRed.size, a.size, T_bb.size))
    planckCarb = np.zeros((waveRed.size, a.size, T_bb.size))
    for i in range(T_bb.size):
        planckSil[:,:,i]  = planck(waveRed, T_EQ['Sil'][:,i], um=False).transpose() # Check units
        planckCarb[:,:,i] = planck(waveRed, T_EQ['Carb'][:,i], um=False).transpose()
    planckSil = planckSil.transpose([1,0,2])
    planckCarb =  planckCarb.transpose([1,0,2])

    # Equation 15 in Marshall et al 2007
    # Calculate Silicate emissivities
    bigDndaSil = np.tile(dnda['Sil'], (waveRed.size,1,1)).transpose([1,0,2])
    eSilTot = intTab(biga*bigDndaSil*planckSil*np.tile(cSil, (T_bb.size,1,1)).transpose(1,2,0), dloga)
    # Calculate carbonaceous emissivities
    bigDndaCarb = np.tile(dnda['Carb'], (waveRed.size,1,1)).transpose([1,0,2])
    eCarb = intTab(biga*bigDndaCarb*planckCarb*np.tile(cCarb, (T_bb.size,1,1)).transpose(1,2,0), dloga)

    # Create index of wavelength points to fit silicate continuum to
    wBL = 1.
    wBH = 5.2
    wAL = 1e2
    wAH = 1e3
    idx = ((waveRed > wBL) & (waveRed < wAH))
    idxCon = (((waveRed > wBL) & (waveRed < wBH)) | \
              ((waveRed > wAL) & (waveRed < wAH)))

    # Calculate silicate continuum and amorphous features
    eSilCont = np.zeros((waveRed.size, T_bb.size))
    eAmoFeat = np.zeros((waveRed.size, T_bb.size))
    for i in range(len(T_bb)):
        if np.any(eSilTot > 0.0):
            # Calculate silicate continuum
            # This section is made to exactly match IDL, and could probably 
            # be streamlined. Note the use of np.copy(), as otherwise numpy
            # will assign multiple names to the same array object
            eSilTotTemp = eSilTot[:,i]
            eSilTotTemp[eSilTotTemp<=0.] = 1e-200 # to prevent underflow funkiness
            eSilContTemp = np.copy(eSilTotTemp)
            spl = splrep(np.log(waveRed[idxCon]), np.log(eSilTotTemp[idxCon]))
            eSilContTemp[idx] = np.exp(splev(np.log(waveRed[idx]), spl))
            # Subtract continuum to obtain 10/18 micron features
            eAmoFeatTemp = np.subtract(eSilTotTemp, eSilContTemp)
            eAmoFeatTemp[np.isnan(eAmoFeatTemp)] = 0.0
            eAmoFeatTemp[eAmoFeatTemp < 0.] = 0.0 ### Issue here

            # idx0 is used to avoid weirdness happening if you divide by 0
            idx0 = ((eSilContTemp == 0) | np.isnan(eSilContTemp))
            eSilContTemp[idx0] = 1.
            eAmoFeatTemp[eAmoFeatTemp/eSilContTemp < 1e-3] = 0.0 
            eAmoFeatTemp[idx0] = 0.

            eAmoFeat[:,i] = eAmoFeatTemp
            # Subtract features from total to obtain final continuum
            eSilCont[:,i] = eSilTotTemp - eAmoFeatTemp
        else:
            eSilCont[:,i] = np.zeros(waveRed.size)
            eAmoFeat[:,i] = np.zeros(waveRed.size)
    
    # Obtain crystalline silicate feature mass-opacities
    if fstTab is None:
        fstTab = np.genfromtxt(tablePath+'k_abs.fst.txt', comments=';')
    ### CHECKME
        # fstTab = np.genfromtxt(tablePath+'k_abs_upsampled.fst.txt', comments=';')
        fstWave = fstTab[:,0]
        fstK = fstTab[:,1]
    if ensTab is None:
        ensTab = np.genfromtxt(tablePath+'k_abs.ens.txt', comments=';')
        # ensTab = np.genfromtxt(tablePath+'k_abs_upsampled.ens.txt', comments=';')
        ensWave = ensTab[:,0]
        ensK = ensTab[:,1]

    kPerMassFst = np.zeros(waveRed.shape)
    idx =(waveRed >= np.min(fstWave)) & (waveRed <= np.max(fstWave))
    f1 = interp1d(np.log(fstWave), np.log(fstK), fill_value='extrapolate')
    kPerMassFst[idx] = np.exp(f1(logWaveRed[idx]))
    
    kPerMassEns = np.zeros(waveRed.shape)
    idx = (waveRed >= np.min(ensWave)) & (waveRed <= np.max(ensWave))
    f2 = interp1d(np.log(ensWave), np.log(ensK), fill_value='extrapolate')
    kPerMassEns[idx] = np.exp(f2(logWaveRed[idx]))
    
    # Calculate crystalline silicate feature emissivities
    denSil = 3.5e-12
    bMass = np.tile(4*np.pi/3 * denSil * a**3, (waveRed.size, 1)).transpose()
    cFst0 = np.tile(kPerMassFst, (a.size,1))
    cEns0 = np.tile(kPerMassEns, (a.size,1))
    cFst = bMass*cFst0
    cEns = bMass*cEns0
    eFstFeat = intTab(np.tile(cFst, (T_bb.size,1,1)).transpose(1,2,0), dloga)
    eEnsFeat = intTab(np.tile(cEns, (T_bb.size,1,1)).transpose(1,2,0), dloga)
    xCry = intTab(np.tile(bMass, (T_bb.size,1)), dloga)
    
    # Restore input wavelength grid and output
    eSilTotOut = np.zeros((wave.size, T_bb.size))
    eSilContOut = np.zeros((wave.size, T_bb.size))
    eAmoFeatOut = np.zeros((wave.size, T_bb.size))
    eFstFeatOut = np.zeros((wave.size, T_bb.size))
    eEnsFeatOut = np.zeros((wave.size, T_bb.size))
    eCarbOut = np.zeros((wave.size, T_bb.size))
    
    for i in range(T_bb.size):
        eSilTotOut[:,i] = spline(logWave, logWaveRed, eSilTot[:,i])
        eSilContOut[:,i] = spline(logWave, logWaveRed, eSilCont[:,i])
        eAmoFeatOut[:,i] = spline(logWave, logWaveRed, eAmoFeat[:,i])
        eFstFeatOut[:,i] = spline(logWave, logWaveRed, eFstFeat[:,i])
        eEnsFeatOut[:,i] = spline(logWave, logWaveRed, eEnsFeat[:,i])
        eCarbOut[:,i] = spline(logWave, logWaveRed, eCarb[:,i])
        
    return {'wave':wave, 't_bb':T_bb, 'SilTot':eSilTotOut, 'SilCont':eSilContOut, 'AmoFeat':eAmoFeatOut,
            'FstFeat':eFstFeatOut, 'EnsFeat':eEnsFeatOut, 'Carb':eCarbOut}


def init_interpolator(E_T=None, fCarb=1.0, fSil=1.0, fCry=0., fEns=0.5):
    if E_T is None:
        E_T = grain_emissivity(wave, None)
    
    E_T_Sil = E_T['SilCont']
    j = fCarb*E_T['Carb'] + \
        fSil*((1-fCry)*E_T['SilTot']) + \
        fCry*((1-fEns)*E_T_Sil + fEns*(E_T_Sil+E_T['EnsFeat']))  
    j[j < 0] = 1e-200

    rgi = RegularGridInterpolator((E_T['wave'], np.log(E_T['t_bb'])), j)

    return rgi


def eval_interpolator(wave, T_bb, rgi):
    return rgi(np.stack((wave, np.log(T_bb)*np.ones(wave.shape)), axis=1))


def grain_totemissivity(wave, T_bb, fCarb=1.0, fSil=1.0, fCry=0., fEns=0.5, E_T=None, FASTTEMP=False, rgi=None):
    ''' Calculates the total grain emissivity from all species

    Replacing jam_graintotemissivity, mostly verified as of 9/26/20, dropped features for jSilDo,
    jCryDo, jAmoDo since I don't think they're being used

    Arguments:
    wave -- array of wavelengths over which to calculate the opacity
    T_bb -- blackbody temperature of ambient radiation field

    Keyword Arguments:
    fCarb -- scaling factor for carbonaceous grains (default 1)
    fSil -- scaling factor for silicate grains (default 1)
    fCry -- scaling factor for crystaline opacities (default 1)
    fEns -- scaling factor for something opacity. (default 0.5)
    E_T -- emissivity dictionary (default 0)
    FASTTEMP -- replaces interpolation over temperature with nearest-
    neighbor. Significantly speed improvement, some loss in accuracy
    in dust-dominated portions of the spectrum

    Returns: Total grain emission at the specified wavelength points

    '''
    # RGI should be obsolete. Consider removal
    if rgi is not None:
        return rgi(np.stack((wave, np.log(T_bb)*np.ones(wave.shape)), axis=1))

    ### Fast mode treats temp as a discrete variable
    if FASTTEMP:
        if E_T is None:
            E_T = grain_emissivity(wave, T_bb)
        logtbb = np.log(T_bb)
        logET = np.log(E_T['t_bb'])
        argt = np.argmin(np.abs(logET - logtbb))
        
        E_T_Sil = E_T['SilCont'][:,argt]
        j = fCarb*E_T['Carb'][:,argt] + \
            fSil*((1-fCry)*E_T['SilTot'][:,argt]) + \
            fCry*((1-fEns)*E_T_Sil + fEns*(E_T_Sil+E_T['EnsFeat'][:,argt]))  
        j[j < 0] = 1e-200

        return j
    else:
        interp = np.interp # This speeds up the function lookups in the loop
        log = np.log
        if E_T is None:
            E_T = grain_emissivity(wave, T_bb)

        # Skipping some things that don't seem to be used for now
        # IDL version maybe does an interpolation here but IDK what it's actually doing so skipping
        # Interpolate onto the temperature of the input source
        logtbb = log(T_bb)
        logET = log(E_T['t_bb'])
        tdiff = np.abs(logET - logtbb)
        arglow, arghigh = np.argsort(tdiff)[:2]
        if arghigh < arglow:
            tmp = arglow
            arglow = arghigh
            arghigh = tmp

        E_T_Sil = E_T['SilCont'][:,arglow:arghigh+1]
        j = fCarb * E_T['Carb'][:,arglow:arghigh+1] + \
            fSil * ((1-fCry) * E_T['SilTot'][:,arglow:arghigh+1]) + \
            fCry * ((1-fEns) * E_T_Sil + fEns * (E_T_Sil + E_T['EnsFeat'][:,arglow:arghigh+1]))
        j[j < 0] = 1e-200
        ### Somehow despite the extra logs this is faster???
        logJ = log(j)
        logtlow = logET[arglow]
        logthigh = logET[arghigh]
        logJlow = logJ[:,0]
        logJhigh = logJ[:,1]

        ## Do the interpolation manually since it's faster
        scale = (logtbb - logtlow)/(logthigh - logtlow)
        jTot = scale * logJhigh + (1-scale) * logJlow
        ### This is to keep a really annoying underflow from printing, it's fine
        with np.errstate(invalid='warn'):
            #print(jTot)
            jTot = np.e**jTot

        ### Minimal function calls
        # logtlow = logET[arglow]
        # logthigh = logET[arghigh]
        # jlow = j[:,0]
        # jhigh = j[:,1]

        # ## Do the interpolation manually since it's faster
        # scale = (logtbb - logtlow)/(logthigh-logtlow)
        # jTot = scale*jhigh + (1-scale)*jlow
        
        ### Use scipy RGI - this is slowest
        # rgi = RegularGridInterpolator((E_T['wave'], logET[arglow:arghigh+1]), j)
        # jTot = rgi(np.stack((E_T['wave'], logtbb*np.ones(E_T['wave'].shape)), axis=1))

        if not (wave.shape==E_T['wave'].shape and np.allclose(wave, E_T['wave'])):
            assert 1==0
            jTot = np.exp(interp(log(wave), logET, log(jTot)))

        return jTot



def pah_crosssection(wave, a, cGra, feat=False):
    ''' Compute PAH cross sections 

    Direct port of IDL jam_pahcrosssection routine. 
    Appears to correspond to IDL version verified 8/24/20

    Arguments:
    wave -- array of wavelengths to compute cross sections
    a -- array of grain sizes to compute cross sections
    cGra -- graphite abuncances, read from a grain size table previously

    Keyword Arguments:
    feat -- whether to include narrow (feature) PAHs as well as broad (default False)

    Returns: Continuum and feature absorption cross sections, each of which is further 
    separated into neutral and ionized species
    '''
    # These are hardcoded in the IDL version, so I'm just turning it into python
    # Columns are wave0, gamma, neu, ion
    
    if not feat:
        drudeC = np.asarray([[7.220e-2, 0.195, 7.97e7, 7.97e7],
                             [2.175e-1, 0.217, 1.23e7, 1.23e7],
                             [1.050e+0, 0.055, 0.00e0, 2.00e4], 
                             [1.260e+0, 0.110, 0.00e0, 7.8e-2], 
                             [1.905e+0, 0.090, 0.00e0, -146.5], 
                             [1.500e+1, 0.800, 5.00e1, 5.00e1]])
        peakNeuC = 2./np.pi * drudeC[:,0]/1e4 * 1e-20 * drudeC[:,2] / drudeC[:,1]
        peakIonC = 2./np.pi * drudeC[:,0]/1e4 * 1e-20 * drudeC[:,3] / drudeC[:,1]
    drudeF = np.asarray([[3.300e+0, 0.012, 394., 89.4],
                         [5.250e+0, 0.030, 2.5, 20.],
                         [5.700e+0, 0.040, 7.5, 60.],
                         [6.220e+0, 0.0284, 29.4, 236.],
                         [6.690e+0, 0.070, 5.88, 47.2],
                         [7.417e+0, 0.126, 15.8, 142.],
                         [7.598e+0, 0.044, 23.8, 214.],
                         [7.850e+0, 0.053, 21.3, 192.],
                         [8.330e+0, 0.052, 6.94, 48.4],
                         [8.610e+0, 0.039, 27.8, 194.],
                         [1.123e+1, 0.010, 12.8, 12.],
                         [1.130e+1, 0.029, 58.4, 54.7],
                         [1.199e+1, 0.050, 24.2, 20.5],
                         [1.261e+1, 0.0435, 34.8, 31.],
                         [1.360e+1, 0.020, 0.5, 0.5],
                         [1.419e+1, 0.025, 0.5, 0.5],
                         [1.590e+1, 0.020, 0.5, 0.5],
                         [1.6447e+1, 0.014, 0.75, 0.75],
                         [1.7038e+1, 0.065, 3.08, 3.08],
                         [1.7377e+1, 0.012, 0.28, 0.28],
                         [1.7873e+1, 0.016, 0.14, 0.14]])
    # Number of C atoms in each PAH
    nC = ((1e4/1.286)*a)**3 # density(gra) = 2.24 g cm-3
    
    # Hydrogenation (H/C) for each PAH
    HtoC = np.zeros(nC.size)
    HtoC[nC < 25.] = 0.5
    HtoC[((nC > 25.) & (nC < 100.))] = 0.5/np.sqrt(nC[((nC > 25.) & (nC < 100.))])/25.
    HtoC[nC > 100.] = 0.25

    # Convert to a wavenumber vector
    x = 1./wave

    cContNeu = np.zeros((wave.size, a.size))
    cContIon = np.zeros((wave.size, a.size))
    cFeatNeu = np.zeros((wave.size, a.size))
    cFeatIon = np.zeros((wave.size, a.size))
    bigNC = np.tile(nC, (wave.size,1))
    # Calculate C_PAH

    ### x > 17.25 ###
    idx = x > 17.25
    cContNeu[idx] = cGra[idx]
    cContIon[idx] = cGra[idx]
    ### 15 < x < 17.25 ###
    idx = ((x > 15) & (x < 17.25))
    if idx.size > 0:
        cont = (bigNC[idx] * np.tile((126.-6.4934*x[idx])*1e-18, (a.size, 1)).transpose())
        cContNeu[idx] = cont
        cContIon[idx] = cont
    ### 10 < x < 15 ###
    idx = ((x > 10) & (x < 15))
    if idx.size > 0:
        drudeFluxNeu = drude_prof(wave[idx], [[drudeC[0][0]], [drudeC[0][1]], [peakNeuC[0]]])
        drudeFluxIon = drude_prof(wave[idx], [[drudeC[0][0]], [drudeC[0][1]], [peakIonC[0]]]) 
        cont = (-3. + 1.35*x[idx])*1e-18
        cContNeu[idx] = bigNC[idx] * np.tile(drudeFluxNeu + cont, (a.size, 1)).transpose()
        cContIon[idx] = bigNC[idx] * np.tile(drudeFluxIon + cont, (a.size, 1)).transpose()
    ### 7.7 < x < 10 ###
    idx = ((x > 7.7) & (x < 10))
    cont = bigNC[idx] * np.tile((66.302 - 24.367*x[idx] + 2.95*x[idx]**2 - 0.1057*x[idx]**3) * 1e-18, (a.size,1)).transpose()
    cContNeu[idx] = cont
    cContIon[idx] = cont
    ### 5.9 < x < 7.7 ###
    idx = ((x > 5.9) & (x < 7.7))
    drudeFluxNeu = drude_prof(wave[idx], [[drudeC[1][0]], [drudeC[1][1]], [peakNeuC[1]]])
    drudeFluxIon = drude_prof(wave[idx], [[drudeC[1][0]], [drudeC[1][1]], [peakIonC[1]]]) 
    cont = (1.8687 + 0.1905*x[idx] + 0.4175*(x[idx]-5.9)**2 + 0.0437*(x[idx]-5.9)**3) * 1e-18
    cContNeu[idx] = bigNC[idx] * np.tile(cont + drudeFluxNeu, (a.size, 1)).transpose()
    cContIon[idx] = bigNC[idx] * np.tile(cont + drudeFluxIon, (a.size, 1)).transpose()
    ### 3.3 < x < 5.9 ###
    idx = ((x > 3.3) & (x < 5.9))
    drudeFluxNeu = drude_prof(wave[idx], [[drudeC[1][0]], [drudeC[1][1]], [peakNeuC[1]]])
    drudeFluxIon = drude_prof(wave[idx], [[drudeC[1][0]], [drudeC[1][1]], [peakIonC[1]]])
    cont = (1.8687 + 0.1905*x[idx]) * 1e-18 
    cContNeu[idx] = bigNC[idx] * np.tile(cont + drudeFluxNeu, (a.size, 1)).transpose()
    cContIon[idx] = bigNC[idx] * np.tile(cont + drudeFluxIon, (a.size, 1)).transpose()


    ### x < 3.3 ###
    idx = x < 3.3
    if idx.size > 0:
        for i in range(a.size):
            sigNeuF = 1e-20*drudeF[:,2]
            sigIonF = 1e-20*drudeF[:,3]
            sigNeuF[0] *= HtoC[i]
            sigIonF[0] *= HtoC[i]
            sigNeuF[8:14] *= HtoC[i]
            sigIonF[8:14] *= HtoC[i]
            peakNeuF = (2./np.pi) * drudeF[:,0]/1e4 * sigNeuF / drudeF[:,1]
            peakIonF = (2./np.pi) * drudeF[:,0]/1e4 * sigIonF / drudeF[:,1]
            drudeFluxNeu = drude_prof(wave[idx], [[drudeF[:,0], drudeF[:,1], peakNeuF]])
            drudeFluxIon = drude_prof(wave[idx], [[drudeF[:,0], drudeF[:,1], peakIonF]])
            cFeatNeu[idx, i] = nC[i] * drudeFluxNeu
            cFeatIon[idx, i] = nC[i] * drudeFluxIon
            if not feat: 
                cCont = 34.58 * 10**(-1.0*(18. + (3.431/x[idx])))
                m = 0.4*nC[i] if nC[i] > 40. else 0.3*nC[i]
                waveCutNeu = 1./((3.804/np.sqrt(m)) + 1.052)
                waveCutIon = 1./((2.282/np.sqrt(m)) + 0.889)
                yNeu = waveCutNeu / wave[idx]
                yIon = waveCutIon / wave[idx]
                cutNeu = (np.arctan(1e3*(yNeu-1.)**3/yNeu) / np.pi) + 0.5
                cutIon = (np.arctan(1e3*(yIon-1.)**3/yIon) / np.pi) + 0.5
                drudeFluxNeu = drude_prof(wave[idx], [[drudeC[5][0]], [drudeC[5][1]], [peakNeuC[5]]])
                drudeFluxIon = drude_prof(wave[idx], [[drudeC[5][0]], [drudeC[5][1]], [peakIonC[5]]])
                cContNeu[idx,i] = nC[i] * (cutNeu*cCont + drudeFluxNeu)
                cContIon[idx,i] = nC[i] * (cutIon*cCont + drudeFluxIon)

    # Near-IR term
    if not feat:
        drudeFluxNeu = drude_prof(wave, [drudeC[:,0][2:5], drudeC[:,1][2:5], peakNeuC[2:5]])
        drudeFluxIon = drude_prof(wave, [drudeC[:,0][2:5], drudeC[:,1][2:5], peakIonC[2:5]])
        cCont = 3.5 * 10**(-19. - (1.45/x)) * np.exp(-0.1*x**2)
        cContNeu += bigNC * np.tile(drudeFluxNeu, (a.size, 1)).transpose()
        cContIon += bigNC * np.tile(drudeFluxIon + cCont, (a.size, 1)).transpose()

    cCont = (cContNeu, cContIon)
    cFeat = (cFeatNeu, cFeatIon)

    return cCont, cFeat
