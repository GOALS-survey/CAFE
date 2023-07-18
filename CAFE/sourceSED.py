import numpy as np 
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator
from astropy.table import Table
import warnings

import CAFE
from CAFE.mathfunc import spline, intTab

# import ipdb

#################################
### SourceSED/Opacities       ###
#################################
def planck(wave, T, um=True, Hz=False):
    ''' Compute a Planck function

    Planck function from jam_planck.pro, verified 8/26/20

    Arguments:
    wave -- wavelength grid for blackbody
    T - blackbody temperature(s)

    Keyword Arguments:
    um - compute flux density in um (default True)
    Hz - compute flux density with Hz (default False)

    Returns: Array of blackbody spectra for each temperature over
    the given wavelength range
    '''
    c1 = 3.972895e19 # = 2 * h * c = [Jy sr^-1 um^3]
    c2 = 1.4387731e4 # = h * c / k = [K um] 
    if um and Hz:
        raise ValueError
    if um:
        cUm = 1.1918685e11
        arg1 = cUm/wave**5
    elif Hz:
        cHz = 3.972895e-4
        arg1 = cHz/wave**3
    else:
        arg1 = c1/wave**3
    arg2 = c2/wave 
    Bs = []
    T = np.asarray(T)
    ### This can give an overflow warning but it's harmless, just sets it to zero
    warnings.filterwarnings('ignore', message='overflow Encountered in exp')
    for i in range(T.size):
        if T.size == 1:
            Bs.append(arg1/(np.exp(arg2/T) - 1))
        else:
            Bs.append(arg1/(np.exp(arg2/T[i]) - 1))
    if len(Bs) == 1: Bs = Bs[0]
    warnings.resetwarnings()
    return np.asarray(Bs)


def sourceSED_ISRF(wave, T_bb=None):
    ''' Compute an ISRF SED

    Port of Jam_SourceSED_ISRF from jam_sourcesed.pro, verified 8/26/20

    Arguments:
    wave -- array to compute SED over

    Keyword Arguments:
    T_bb -- Blackbody temperature to scale SED to match (default None)

    Returns: Array of fluxes at specified wavelengths, in erg s-1 cm-2 um-1
    '''
    # Calculate UV part
    uWave = np.zeros(wave.shape)
    uWave[((wave > 0.0912) & (wave < 0.11))] = 38.57/3e10 * np.power(wave[((wave > 0.0912) & (wave < 0.11))], 3.4172)
    uWave[((wave > 0.11) & (wave < 0.134))] = 2.045e-2/3e10
    uWave[((wave > 0.134) & (wave < 0.246))] = 7.115e-4/3e10 * np.power(wave[((wave > 0.134) & (wave < 0.246))], -1.6678)

    # Calculate BB part
    W = 4*np.pi * np.asarray([1e-14, 1e-13, 4e-13]) / 3e10
    T = np.asarray([7.5e3, 4e3, 3e3])
    pf = planck(wave[wave>0.246], T)
    uWave[wave>0.246] = W[0]*pf[0] + W[1]*pf[1] + W[2]*pf[2]
    if T_bb is not None:
        sigma_SB = 5.67e-5
        uWaveOld = uWave
        uTotOld = intTab(wave*uWaveOld, np.log(wave))
        tBBOld = np.power(3e10 / 4. / sigma_SB * uTotOld, 0.25)
        uWave*=(T_bb/tBBOld)**4

    return 3e10*uWave


def sourceSED_AGN(wave, lTot=1e11, r=None, T_bb=None):
    ''' Compute an AGN SED

    Port of the corresponding function in jam_sourceSED

    Arguments:
    wave -- array to compute SED over

    Keyword Arguments:
    lTot -- Total luminosity of the AGN
    r -- Distance from source to compute SED
    T_bb -- Blackbody temperature to scale SED to match (default None)

    Returns: Array of fluxes at specified wavelengths, in erg s-1 cm-2 um-1
    '''
    # 1e-3 < wave/um < 5e-2
    lFreq = (wave/1e-3)**3
    # 5e-2 < wave/um < 0.1216
    idx = wave > 5e-2
    lFreq[idx] = np.interp(5e-2, wave, lFreq)* np.power(wave[idx]/5e-2, 1.8)
    # 0.1216 < wave/um < 10
    idx = wave > 0.1216
    lFreq[idx] = np.interp(0.1216, wave, lFreq)*np.power(wave[idx]/0.1216, 0.46)
    # 10 < wave/um < 1e3
    bb = planck(wave, 1e3, um=False, Hz=True)
    idx = wave > 10.
    lFreq[idx] = np.interp(10., wave, lFreq)*(bb[idx]/np.interp(10., wave, bb))
    
    # convert to L Wave
    lWave = lFreq/wave**2
    # normalize L Wave
    lWave/=intTab(wave*lWave, np.log(wave[1]/wave[0]))

    # Calculate fWave a distance R away from LTot source
    if r is None or T_bb is None:
        r = 1.0 # pc
    cgsPerLSun = 3.827e33
    cmPerPc = 3.086e18
    fWave = cgsPerLSun*lTot*lWave/(4*np.pi*(cmPerPc*r)**2)

    # Scale to Tbb
    if T_bb is not None:
        sigmaSB = 5.67e-5
        fWaveOld = np.copy(fWave)
        fTotOld = intTab(wave*fWaveOld, np.log(wave[1]/wave[0]))
        T_bb_old = np.pow(fTotOld/(2*sigmaSB), 0.25)
        scale = (T_bb/T_bb_old)**4
        fWave*=scale
        fTot = fTotOld*scale
        r/=np.sqrt(scale)

    return fWave


def sourceSED_SB(wave, age, tablePath, nebular=False):
    ''' Compute a starburst SED

    Port of the coresponding function in jam_sourceSED

    Arguments:
    wave -- array to compute SED over
    age -- whcih sb99 grid age to use

    Keyword Arguments:
    tablePath -- location of the sb99 tables (default 'tables/')
    nebular -- whether to use flux or nebular (default False, flux)

    Returns: Array of fluxes at specified wavelengths, in erg s-1 cm-2 um-1
    '''
    # Load inputs from table
    if age not in ['2', '10', '100']:
        raise ValueError('Invalid starburst age')
    fname = 'sb99-'+age+'myr.txt'
    data =  np.genfromtxt(tablePath+fname, comments=';')
    waveTab = data[:,1]
    if nebular: fWave = 10**data[:,2]
    else: fWave = 10**data[:,3]
    waveTab/=1e4
    fWave*=1e4

    # Interpolate onto input wavelength grid
    if not (wave.shape==waveTab.shape and np.allclose(wave, waveTab)):
        f1 = interp1d(np.log(waveTab), fWave, fill_value='extrapolate')
        fWave = f1(np.log(wave))
    
    # Cut off long wavelength flux from 2Myr
    if age=='2':
        fWave[wave>70.]*=np.exp(-1.0*(wave[wave>70.]-70.)/70.)

    # Calculate total luminosity - doesn't appear to be used in CAFE?
    # lTot = intTab(wave*fWave, np.log(wave[1]/wave[0]))

    # Scale to T_bb - I don't think this is used in CAFE?
    return fWave


def sourceSED(wave, source, tablePath, norm=False, Jy=False):
    ''' Returns SED from specified source 
    
    Port of the corespondig function in jam_sourceSED, wrapper 
    for the above set of functions

    Arguments:
    wave -- array of wavelengths to compute SED

    Keyword Arguments:
    source -- type of SED to return (default ISRF)
    Jy -- return flux density in Jy or um-1 (default False/um-1)
    tablePath -- path to SED templates (default 'tables/')

    Returns -- Arrays of wavelength and corresponding flux densities
    '''
    fullWave=np.logspace(-3, 3, num=1000)

    if source == 'ISRF':
        f_Wave = sourceSED_ISRF(fullWave)
    elif source == 'AGN':
        f_Wave = sourceSED_AGN(fullWave)
    elif source == 'SB2Myr':
        f_Wave = sourceSED_SB(fullWave, '2', tablePath)
    elif source == 'SB10Myr':
        f_Wave = sourceSED_SB(fullWave, '10', tablePath)
    elif source == 'SB100Myr':
        f_Wave = sourceSED_SB(fullWave, '100', tablePath)

    if norm:
        f_Wave/=np.trapz(f_Wave*fullWave, np.log(fullWave))

    f_Wave = np.interp(np.log(wave), np.log(fullWave), f_Wave)
    
    if Jy:
        f_Wave*=((1e23/3e14)*wave**2)

    return wave, f_Wave


def load_opacity(wave, fname):
    ''' Loads the opacity table in the format used by the CAFE tables.

    Arguments:
    wave -- wavelegnth range output should be resampled to 
    fname -- filename for table to read

    Returns: Second column from file fname, resampled to wave
    '''
    if fname.split('.')[-1] == 'txt':
        tab = np.loadtxt(fname, comments=';')
        tWave = tab[:,0]
        tKap = tab[:,1]
    if fname.split('.')[-1] == 'ecsv':
        tab = Table.read(fname)
        tWave = tab['wav']
        tKap = tab['tau']

    tau = np.interp(np.log(wave), np.log(tWave), tKap, left=0.0, right=0.0)
    
    return tau

    # tab = np.loadtxt(fname, comments=';')
    # tWave = tab[:,0]
    # tKap = tab[:,1]
    # tau = np.interp(np.log(wave), np.log(tWave), tKap, left=0.0, right=0.0)
    # return tau
