import numpy as np 
from scipy.interpolate import interp1d, splrep, splev, RegularGridInterpolator

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
