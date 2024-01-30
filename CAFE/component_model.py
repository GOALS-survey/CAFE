import numpy as np 
import astropy.units as u

#import ipdb

#################################
### Feature functions         ###
#################################
def pah_drude(path='tables/'):
    ''' Make initial PAH profile structure

    Makes a first guess at PAH feature amplitudes, presumably based on the 
    references Smith et al. 2006) and Brandl et al. (2006). This is a direct
    port of the jam_pahdrude.pro IDL routine used by original CAFE

    Keyword Arguments:
    path -- the path to look for the pah template file

    Returns: Dict, consisting of inital wavelength, with, peak, and line 
    complex IDs for the PAHs in the file
    '''
    data = np.genfromtxt(path+'pah_template.txt', comments=';')
    wave0 = data[:,0]
    gam = data[:,1]
    peak = data[:,2]
    comp = data[:,3]
    idx3 = comp==0
    idx6 = comp==3

    # Copied from IDL
    # Set feature peaks to the values obtained using PAHFIT (Smith et
    # al. 2006) to fit the mean starburst spectrum of Brandl et al. (2006).
    #         PEAK           WAVE   N  COMPLEX
    """
    peak = np.asarray([0.00000000, #$ ;;  3.30   0   0
            0.00000000, #$ ;;  3.40   26  0
            0.06159884, #$ ;;  5.27   1   1
            0.07050930, #$ ;;  5.70   2   2
            0.77360526, #$ ;;  6.22   3   3
        #   0.10000000, $ ;;  6.05   .   19
        #   0.70000000, $ ;;  6.22   3   3
        #   0.40000000, $ ;;  6.29   .   3
        #   0.15000000, $ ;;  6.30   .   3
            0.09797205, #$ ;;  6.69   4   4
            0.23897256, #$ ;;  7.42   5   5
            0.89279697, #$ ;;  7.60   6   5
            0.86156776, #$ ;;  7.85   7   5
            0.25013986, #$ ;;  8.33   8   6
            0.64566928, #$ ;;  8.61   9   7
            0.02442993, #$ ;; 10.68  10   8 ;;
        #   0.20000000, $ ;; 11.00   .   9
            1.29167980, #$ ;; 11.23  11   9
            1.04019180, #$ ;; 11.33  12   9
            0.32429740, #$ ;; 11.99  13  10
            0.84859132, #$ ;; 12.62  14  11
            0.19136069, #$ ;; 12.69  15  11
            0.21897772, #$ ;; 13.48  16  12
            0.01000000, #$ ;; 14.04  17  13 ;; <- Actually zero peak in fit. ;;
            0.17583134, #$ ;; 14.19  18  14
            0.01376596, #$ ;; 15.90  19  15 ;;
            0.42942265, #$ ;; 16.45  20  16
            0.63617951, #$ ;; 17.04  21  16
            0.29235229, #$ ;; 17.375 22  16
            0.25877291, #$ ;; 17.87  23  16
            0.31397021, #$ ;; 18.92  24  17
            0.85513299]) #  ;; 33.10  25  18
    """
    #p_3_6_12112 = 0.064
    #peak[idx3] = (gam[idx6]/gam[idx3])*p_3_6_12112 / (wave0[idx6]/wave0[idx3])*peak[idx6]
    return {'wave0':wave0, 'gamma':gam, 'peak':peak, 'complex':comp}


def gauss_flux(wave, gauss, ext=None):
    ''' Compute the flux from a gaussian profile

    Arguments:
    wave -- array of wavelengths to compute fluxes for
    gauss -- list of profile parameters for each line [wave0, sigma, peak],...]

    Returns: Array of fluxes corresponding to initial wavelength array
    '''
    A0 = np.asarray(gauss[2]) # Peak
    A1 = np.asarray(gauss[0]) # wave0
    gam = np.asarray(gauss[1]) # width in gamma

    #gam[gam<1e-5] = 1e-5  ### Minimum FWHM (um) - error avoidance
    #A0[A0<1e-14] = 1e-14  ### Avoid zeros/underflows
    A2 = A1*gam / 2.35482 ### From FWHM to sigma
    flux = np.zeros(wave.size)

    ### Make a reference gaussian
    gausspts = np.arange(-5, 5, 2e-1)
    gausspts = np.broadcast_to(gausspts, (A0.size, gausspts.size)).transpose()
    unitgauss = np.exp(gausspts**2/-2.)

    ### Shift the gaussian to match the line profile
    ### This seems convoluted, but is actually much faster than exponentiating
    ### for each line we want to use
    pts = (A2*gausspts+A1).transpose()
    line = (unitgauss*A0).transpose()
    for i in range(A1.size):
        flux+=np.interp(wave, pts[i], line[i])

    if ext is not None: #np.len(ext) == np.len(pah_prof):
        flux = flux * ext

    return flux


def drude_prof(wave, drude, ext=None):
    """
    Output the comblined PAH profiles with a set of parameters (wave, width(gamma), peak, complex).

    Port of jam_drudeflux.pro IDL routine. Drude is a list of 3-elements 
    [wave0, gamma, peak] for each line 
    Verified 8/24/20
    This is the line profile used for PAHs, see Draine's ISM book for a better 
    explanation

    Parameters
    ----------
    wave -- array of wavelengths to compute fluxes for
    drude -- list of profile paramerers for each line [[waves],[widths(gamma)],[peaks],[complexes]]

    Returns
    -------
    Profile of the summation of all the PAH features
    """
    drude_arr = np.asarray(drude[:3]).transpose() # get drude arr with 3 cols: wave, width, peak

    # Turn drude into a dict for readability 
    lam = drude_arr[:,0]
    wid = drude_arr[:,1]
    peak = drude_arr[:,2]

    # The below code can be simplified as long as the dimension of drude is 2
    # if drude.ndim == 1:
    #     A0 = [drude[1]**2 * drude[2]]
    #     A1 = [drude[0]]
    #     A2 = [drude[1]]
    #     drude = [drude]
    # else:
    #     A0 = drude[:,1]**2 * drude[:,2]
    #     A1 = drude[:,0]
    #     A2 = drude[:,1]

    pah_prof = np.zeros(wave.size)
    for i in range(len(drude_arr)): # add each line
        pah_prof += peak[i] * wid[i]**2 / ((wave/lam[i] - lam[i]/wave)**2 + wid[i]**2)
        #pah_prof +=  A0[i] / (((wave/A1[i]) - (A1[i]/wave))**2 + A2[i]**2)
    # Errors get computed here when we need them    

    if ext is not None: #np.len(ext) == np.len(pah_prof):
        pah_prof *= ext

    return pah_prof


# def drude_extinct(wave, drude, ext):
#     ''' Analagous to drude flux, but also applies extinction
    
#     Arguments:

#     Returns: 

#     '''
#     ### Get intrinsic flux
#     flux = drude_prof(wave, drude)

#     ### Apply the extinction
#     flux*=ext 
    
#     return flux


def drude_int_fluxes(wave, drude, ext=None, scale=1.0, flxunits=u.Jy, wvunits=u.um):
    ''' Computes integrated flux of each line in drude, with optional extinction

    Arguments:
    wave -- array of wavelengths to compute fluxes
    drude -- list of profile paramerers for each line [[waves],[widths],[peaks],[complexes]]

    Keyword Arguments:
    ext -- extinction curve to apply to flux, with same shape as wave. Default None (no extinction)
    scale -- PAH contribution relative to continuum at refwave. Default 1.0 (use whatever the fitter gives!)
    refwave -- PAH reference wavelength, default 6.22um

    Returns: array of integrated fluxes for each PAH in the drude table

    '''
    if ext is None:
        ext = np.ones(wave.shape)
    int_fluxes = np.zeros(len(drude[0]))
    totflux = np.zeros(wave.shape)
    wave = wave*wvunits
    for i in range(len(int_fluxes)):
        ### Get parameters for specific line
        #lpars = [drude[0][i], drude[1][i], drude[2][i], drude[3][i]]
        lpars = [[drude[0][i]], [drude[1][i]], [drude[2][i]], [drude[3][i]]]
        ### Get extincted flux
        flux = drude_prof(wave.value, lpars, ext=ext)
        totflux+=flux
        ### Units
        flux = (flux*flxunits).to(u.W/u.m**2/wvunits, equivalencies=u.spectral_density(wave))
        ### Integrate
        int_fluxes[i] = np.trapz(flux.value, wave.value)

    return int_fluxes*(u.W/u.m**2)
