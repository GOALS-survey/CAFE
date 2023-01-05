#!/usr/bin/env python3

import pdb
import numpy as np
from lmfit import minimize

class mylmfit2dfun():
    """
    peggy is a wrapper for fitting models to data and trying to perform
    model comparison/selection.
    """
    
    def __init__(self,
                 xy,
                 z,
                 params,
                 xyerr=None,
                 zerr=None,
                 zweight=None,
                 **kwargs):
        """
        Parameters:
            x: independent variable
            y: dependent variable
            xerr/yerr: Uncertainties on the independent variables
            zerr: Uncertainties on the dependent variable
            mymodel: function for the model

        kwarg parameters are:
            mymethod: method for minimization
            mylnlike: function for computing the likelihooda (if omitted,
                      Gaussian likelihood is assumed)
            mylnpriors: function for computing priors (if omitted, no priors
                       are used)
        """

        self.model = gauss2d1gfun
            
        # store data for fitting
        assert len(xy[0]) == len(z), "x, y and z arrays must have the same length."
        self.xy = xy
        self.z = z
        # Arrays are not passed in 2D because the minimizers are not capable to handle NaNs.
        # NaNs need to be removed and then the arrays flattened in order to pass them
        # since it is not possible to keep a 2D strucutre that do not contain all the elements.
        
        if zweight is not None:
            assert len(z) == len(zweight), "z and zweight arrays must have the same length."
            self.zweight = zweight
            self.zerr = 1./np.sqrt(zweight)
        elif zerr is not None:
            assert len(z) == len(zerr), "z and zerr arrays must have the same length."
            self.zerr = zerr
            self.zweight = 1./zerr**2
        else:
            raise ValueError("Uncertainties must be provided for dependent variable.\n")

        self.params = params
        self.ndim = len(self.params)
        
        # assign user-defined likelihood if provided
        self.lnlike = kwargs['mylnlike'] if 'mylnlike' in kwargs.keys() else self.gausslike

        # assign user-defined prior if provided
        self.lnprior = kwargs['mylnpriors'] if 'mylnpriors' in kwargs.keys() else lambda p: 0.
        

    def lmfit(self, **kwargs):

        # assign minimization method if provided
        self.method = kwargs['mymethod'] if 'mymethod' in kwargs.keys() else 'leastsq'        

        nlnl = lambda p, **kwargs: -self.lnlike(p, **kwargs)
        lmfitres = minimize(nlnl, self.params, scale_covar=False, method=self.method, nan_policy='omit', kws=kwargs)
        if not lmfitres:
            raise RuntimeError("No fits were successful.\n")
        
        self.lmfitres = lmfitres        
        
        self.pars = [lmfitres.params[par].value for par in lmfitres.params.keys()]
        self.pars_unc = [lmfitres.params[par].stderr for par in lmfitres.params.keys()]


    def gausslike(self, theta, **kwargs):
        """
        Default Gaussian likelihood
        """

        if self.method == 'leastsq':
            return -(self.z - self.model(self.xy, theta, **kwargs)) * np.sqrt(self.zweight)
        return -0.5 * np.sum((self.z - self.model(self.xy, theta, **kwargs))**2 * self.zweight) + np.sum(np.log(2.*np.pi*np.sqrt(1./self.zweight)))


# Generic definition of a 2D Gaussian function
def gauss2d1gfun(xy, pars, tilt=False, power=False):
    
    npars = len(pars)
    #parsi = np.zeros(npars)
    parsi = np.asarray([])
    if isinstance(pars, dict):
        parvals = pars.valuesdict()
        for i in range(npars): parsi = np.concatenate((parsi, [parvals['par'+str(i)]]))
    else:
        for i in range(npars): parsi = np.concatenate((parsi, [pars[i]]))

    # Tilt angle is defined in mathematical terms: Clockwise starting on the positive x-axis.
    if tilt:
        func = parsi[0]*np.exp(-0.5*( ((xy[0]-parsi[1])*np.cos(parsi[5])/parsi[3] - (xy[1]-parsi[2])*np.sin(parsi[5])/parsi[3])**power + \
                                     ((xy[0]-parsi[1])*np.sin(parsi[5])/parsi[4] + (xy[1]-parsi[2])*np.cos(parsi[5])/parsi[4])**power)) if power else \
                                     parsi[0]*np.exp(-.5*( ((xy[0]-parsi[1])*np.cos(parsi[5])/parsi[3] - (xy[1]-parsi[2])*np.sin(parsi[5])/parsi[3])**2. + \
                                    ((xy[0]-parsi[1])*np.sin(parsi[5])/parsi[4] + (xy[1]-parsi[2])*np.cos(parsi[5])/parsi[4])**2.))
    else:
        func = parsi[0] * np.exp(-0.5 * ((xy[0]-parsi[1])**power + (xy[1]-parsi[2])**power) / parsi[3]**power) if power else \
               parsi[0] * np.exp(-0.5 * ((xy[0]-parsi[1])**2. + (xy[1]-parsi[2])**2.) / parsi[3]**2.)

    if tilt:
        if npars > 6:
            np.add(func, parsi[6])
            if npars > 7:
                for i in range(7,npars): np.add(func, parsi[i] * xy[np.mod(i-7,2)]**((i-7)//2+1))
    else:
        if npars > 4:
            np.add(func, parsi[4])
            if npars > 5:
                for i in range(5,npars): np.add(func, parsi[i] * xy[np.mod(i-5,2)]**((i-5)//2+1))

    return func
