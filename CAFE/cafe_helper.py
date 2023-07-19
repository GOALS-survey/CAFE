import numpy as np
import lmfit as lm # https://dx.doi.org/10.5281/zenodo.11813
import configparser
import ast
from astropy.io import fits
from astropy.table import Table
import site

import CAFE
from CAFE.component_model import gauss_flux, drude_prof, pah_drude
from CAFE.dustgrainfunc import *
from CAFE.cafe_io import *

cafeio = cafe_io()


class CAFE_param_generator:

    def __init__(self, spec, inparfile, optfile, cafe_path=None):

        # The data is read from a spectrum1D object
        self.wave = spec.spectral_axis.value
        self.flux = spec.flux.value
        self.flux_unc = spec.uncertainty.quantity.value
        self.z = spec.redshift.value
        
        # Read in and opt files into dictionaries
        inpars = cafeio.read_inifile(inparfile)
        inopts = cafeio.read_inifile(optfile)

        self.inpars = inpars
        self.inopts = inopts
        tablePath, _ = cafeio.init_paths(inopts, cafe_path=cafe_path)
        self.tablePath = tablePath
        

    def make_parobj(self, get_all=False, parobj_update=False, init4fit=False):
        """
        Build the Parameters object for lmfit based on the input spectrum.
        This function combines the Parameter obj output from "make_cont_pars"
        and "make_feat_pars".

        Parameters
        ----------
            obswave:
                wavelength in the rest frame
        """
        inpars = self.inpars
        inopts = self.inopts

        # Make continuum parameter object from input file
        try:
            params = self.make_cont_pars(inpars['CONTINUA INITIAL VALUES AND OPTIONS'], parobj_update=parobj_update, init4fit=init4fit, Onion=inopts['SWITCHES']['ONION'])
        except:
            raise IOError('Input parameter file not found')
        # Note that if a parameter object (= parobj_update) is provided, the continuum parameters will still have the original limits as specified in the input file
        # However, the parameter .value's will be replaced by those in the parameter object

        # Make feature parameters object
        if 'RESOLUTIONS' in inpars['MODULES & TABLES']:
            instnames = inpars['MODULES & TABLES']['RESOLUTIONS']
        else: raise IOError('No spectral modules given')
        
        if parobj_update is False:
            # Initialize the feature waves, gammas and peaks (and names for lines or complexes for PAHs) from input file
            # Note that these are just individual features, and do not include broad components or the parameters themselves
            gauss, drude, gauss_opc = self.init_feats(self.wave, self.flux, self.flux_unc, self.z, self.tablePath, instnames, get_all=get_all,
                                                      atomic_table=inpars['MODULES & TABLES']['ATOMIC_INPUT'],
                                                      molecular_table=inpars['MODULES & TABLES']['MOLECULAR_INPUT'],
                                                      hrecomb_table=inpars['MODULES & TABLES']['HRECOMB_INPUT'],
                                                      pah_table=inpars['MODULES & TABLES']['PAH_INPUT'],
                                                      gopacity_table=inpars['MODULES & TABLES']['GOPACITIES_INPUT'])
            
        else:
            # Read the feature waves, gammas and peaks from the provided parameter file
            gauss, drude, gauss_opc = self.get_feats(params=parobj_update, apply_vgrad2waves=True)
            
        if not inopts['FIT OPTIONS']['FITLINS']:
            gauss = [[], [], [], [], []]
        if not inopts['FIT OPTIONS']['FITPAHS']:
            drude = [[], [], [], []]
        if not inopts['FIT OPTIONS']['FITOPCS']:
            gauss_opc = [[], [], [], [], []]
            
        # Transform features into LMFIT parameters
        # Contrary to the continuum parameters, the feature parameters are constructed in an extra step
        # Note that make_feat_pars doesn't have parobj_update keyword because the .value's have been injected before with parcube2parobj()
        # and read with get_feats(). make_feat_pars() basically rebuilds the vary, lims and args
        feat_params = self.make_feat_pars(inpars['PAH & LINE OPTIONS'], gauss, drude, gauss_opc, get_all=get_all, parobj_update=parobj_update, init4fit=init4fit)
        for key in feat_params: params.add(feat_params[key])
        
        # Params is a dictionary with all the initialized LMFIT parameters
        if get_all == True and len(feat_params) != (len(gauss[0])+len(drude[0])+len(gauss_opc[0])+int(np.sum(gauss[4])))*3+1:
            raise ValueError('There has been a rejection during the creation of the parameters, which should not have happened')
        print('Parameter object has',int((len(feat_params)-(len(drude[0])*3+len(gauss_opc[0]*3)))/3),'lines,',len(drude[0]),'PAHs,',len(gauss_opc[0]),'extra opacity features, and',len(params)-len(feat_params),'continuum parameters')
        
        
        return params


    @staticmethod
    def init_feats(wave, flux, flux_unc, z,
                   tablePath,
                   instnames,
                   get_all=False,
                   atomic_table=None, 
                   molecular_table=None, 
                   hrecomb_table=None,
                   pah_table=None,
                   gopacity_table=None):
        ''' Initialize line parameters with an iterative estimate

        Drude peaks are a bit different (few ppt) compared with IDL version but I think it's
        fine, verified 9/10/20. This is the first part of the jam_fitfeatures.pro IDL routine.

        Arguments:
        wave -- rest wavelengths to compute spectrum at
        flux -- measured fluxes at those wavelengths
        flux_unc -- uncertainty in measured flux

        Keyword Arguments:
        minWave -- minimum wavelength cutoff to apply to lines (default None)
        maxWave -- max wavelength cutoff to apply to lines (default None)
        ref_pah_wave -- Reference wavelength for PAH features (default 6.22)
        extPAH -- extinction to apply to PAHs (default 1)
        tablePath -- Directory with line parameter tables (default 'tables/')
        atomic_table -- List of input atomic features if set (default None)
        molecular_table -- List of input molecular features if set (default None)
        hrecomb_table -- List of input hydrogen recombination features if set (default None)
        pah_table -- List of input PAH features if set (default None)
        gopacity_table -- List of input gaussian opacity features if set (default None)

        Returns: Initial guesses for gaussian and drude line profile parameters for
        lines in the wavelength range
        '''

        # Read the instrument wavelength coverages as DataFrames
        inst_df = cafeio.read_inst(instnames, tablePath)

        minWave = np.nanmin(wave)
        maxWave = np.nanmax(wave)

        #for i in range(len(inst_df)):
        #    if (minWave*(1+z) > inst_df.iloc[i].wMin) or (maxWave*(1+z) < inst_df.iloc[i].wMax): print(inst_df.inst[i]+' is defined in .ini file but either there is no associated cube or the module definition extends beyond the spectrum waves.')
        #if (minWave*(1+z) < inst_df.iloc[0].wMin): print(inst_df.inst[0]+' is defined in .ini file but the spectrum waves extends beyond the module definition.')
        #if (maxWave*(1+z) > inst_df.iloc[-1].wMax): print(inst_df.inst[-1]+' is defined in .ini file but the spectrum waves extends beyond the module definition.')

        # ---------------------
        # Get atomic line table
        # ---------------------
        data = np.genfromtxt(tablePath+atomic_table, comments=';', dtype='str')

        aNames = data[:,0]
        aWave0 = data[:,1].astype(float)
        aMask = data[:,3].astype(float)
        aDoub = data[:,4].astype(float)

        # Note that the lines that are assigned to a module/cube are eliminated
        aNames_kept = [] ; aWave0_kept = [] ; aDoub_kept = [] ; aGam = []
        for i in range(len(inst_df)):
            idx = ((aMask == 0) & (aWave0*(1+z) > inst_df.iloc[i].wMin) & (aWave0*(1+z) < inst_df.iloc[i].wMax) \
                   & (aWave0 >= minWave) & (aWave0 <= maxWave))
            aGam = np.concatenate((aGam, 1./(inst_df.iloc[i].rSlope * aWave0[idx]*(1+z) + inst_df.iloc[i].rBias)))
            aNames_kept, aNames = np.concatenate((aNames_kept, aNames[idx])), aNames[~idx]
            aWave0_kept, aWave0 = np.concatenate((aWave0_kept, aWave0[idx])), aWave0[~idx]
            aDoub_kept, aDoub = np.concatenate((aDoub_kept, aDoub[idx])), aDoub[~idx]
            aMask = np.delete(aMask, idx, 0)

        for i, name in enumerate(aNames_kept): aNames_kept[i] = name.replace(')', '').replace('(', '').replace('-', '').replace(')', '')+'_'+str(round(aWave0_kept[i]*1e4))

        # ------------------------
        # Get molecular line table
        # ------------------------
        data = np.genfromtxt(tablePath+molecular_table, comments=';', dtype='str')
        
        mNames = data[:,0]
        mWave0 = data[:,1].astype(float)
        mMask = data[:,2].astype(float)
        mDoub = data[:,3].astype(float)

        mNames_kept = [] ; mWave0_kept = [] ; mDoub_kept = [] ; mGam = []
        for i in range(len(inst_df)):
            idx = ((mMask == 0) & (mWave0*(1+z) > inst_df.iloc[i].wMin) & (mWave0*(1+z) < inst_df.iloc[i].wMax) \
                   & (mWave0 >= minWave) & (mWave0 <= maxWave))
            mGam = np.concatenate((mGam, 1./(inst_df.iloc[i].rSlope * mWave0[idx]*(1+z) + inst_df.iloc[i].rBias)))
            mNames_kept, mNames = np.concatenate((mNames_kept, mNames[idx])), mNames[~idx]
            mWave0_kept, mWave0 = np.concatenate((mWave0_kept, mWave0[idx])), mWave0[~idx]
            mDoub_kept, mDoub = np.concatenate((mDoub_kept, mDoub[idx])), mDoub[~idx]
            mMask = np.delete(mMask, idx, 0)

        for i, name in enumerate(mNames_kept): mNames_kept[i] = name.replace(')', '').replace('(', '').replace('-', '').replace(')', '')+'_'+str(round(mWave0_kept[i]*1e4))
        
        # ----------------------------
        # Get recombination line table
        # ----------------------------
        data = np.genfromtxt(tablePath+hrecomb_table, comments=';', dtype='str')

        hNames = data[:,0]
        hWave0 = data[:,1].astype(float)
        hMask = data[:,2].astype(float)
        hDoub = data[:,3].astype(float)
        
        hNames_kept = [] ;  hWave0_kept = [] ; hDoub_kept = [] ; hGam = []
        for i in range(len(inst_df)):
            idx = ((hMask == 0) & (hWave0*(1+z) > inst_df.iloc[i].wMin) & (hWave0*(1+z) < inst_df.iloc[i].wMax) \
                   & (hWave0 >= minWave) & (hWave0 <= maxWave))
            hGam = np.concatenate((hGam, 1./(inst_df.iloc[i].rSlope * hWave0[idx]*(1+z) + inst_df.iloc[i].rBias)))
            hNames_kept, hNames = np.concatenate((hNames_kept, hNames[idx])), hNames[~idx]
            hWave0_kept, hWave0 = np.concatenate((hWave0_kept, hWave0[idx])), hWave0[~idx]
            hDoub_kept, hDoub = np.concatenate((hDoub_kept, hDoub[idx])), hDoub[~idx]
            hMask = np.delete(hMask, idx, 0)
        
        for i, name in enumerate(hNames_kept): hNames_kept[i] = name.replace(')', '').replace('(', '').replace('-', '').replace(')', '')+'_'+str(round(hWave0_kept[i]*1e4))

        # -----------------
        # Create total list
        # -----------------
        gam = np.concatenate((aGam, mGam, hGam))
        names = np.concatenate((aNames_kept, mNames_kept, hNames_kept))
        wave0 = np.concatenate((aWave0_kept, mWave0_kept, hWave0_kept))
        doubs = np.concatenate((aDoub_kept, mDoub_kept, hDoub_kept))
        inds = np.argsort(wave0)
        gam = gam[inds]
        names = names[inds]
        wave0 = wave0[inds]
        doubs = doubs[inds]
        
        # --------------------------
        # Calculate peaks
        # --------------------------
        fwhm = gam*wave0
        # We estimate the underlying continuum under the lines at +/- 200 km/s because now with JWST the lines are resolved
        waveMinMax = [wave0 * (1. - 300./2.998e5), wave0 * (1. + 300./2.998e5)]
        
        peaks = np.zeros(wave0.shape)
        if get_all == False:
            # Identify gauss features that are absent and set their peak as 0
            nrep = 5
            featflux = np.copy(flux)
            for i in range(nrep):
                fluxMinMax = np.interp(waveMinMax, wave, featflux) # linear interpolation at waveMin
                cont = 0.5 * (fluxMinMax[0] + fluxMinMax[1])
                peak = np.interp(wave0, wave, featflux) - cont
                peaks += peak*2.
                #featflux -= gauss_flux(wave, gauss['wave'], gauss['gam'], gauss['peak'])
                featflux -= gauss_flux(wave, [wave0, gam, peak])
            peaks[peaks < 0.0] = 0.0
            
        gauss = [np.asarray(wave0), np.asarray(gam), np.asarray(peaks), names, np.asarray(doubs)]
        flux_left = flux - gauss_flux(wave, gauss) # Not sure if this is quite right

        #if np.any(flux < 0): ipdb.set_trace()

        if get_all == True:
            if wave0.size != aWave0_kept.size+mWave0_kept.size+hWave0_kept.size: raise ValueError('Not all possible gauss features kept even when get_all=True')
        

        # -------------
        # Get PAH table
        # -------------
        pahTab = np.genfromtxt(tablePath+pah_table, comments=';')
        wave0PAH = pahTab[:,0]
        gamPAH = pahTab[:,1]
        peakPAH = pahTab[:,2]
        compPAH = pahTab[:,3]

        # Remove PAH features outside the observe wavelength range
        idx = (wave0PAH >= minWave) & (wave0PAH <= maxWave)
        wave0PAH = wave0PAH[idx]
        gamPAH = gamPAH[idx]
        peakPAH = peakPAH[idx]
        compPAH = compPAH[idx]
        
        # The goal here is to get a rough guess on what the input drude peaks should be.
        # Set the order when designating pah_ref_wave.
        pah_inspection_order = [6.22, 11.33, 7.6, 3.3]        
        
        ref_pah_peak = 0.
        for ref_pah_wave in pah_inspection_order:
            # Check wheather reference wavelength is within the input spectral range
            # Note: Cannot deal with spectrum with gaps
            if (ref_pah_wave >= minWave) and (ref_pah_wave <= maxWave):
                # Get index of the ref PAH
                idx = (np.abs(wave0PAH - ref_pah_wave)).argmin() # PAH band it's closest to the input PAH wavelength
                # Get gamma of the selected reference PAH
                fwhm = gamPAH[idx] * wave0PAH[idx]
                # We estimate the underlying continuum under the PAHs in terms of velocity, at +/- 10000 km/s
                waveMinMax = [wave0PAH[idx] * (1. - 10000./2.998e5), wave0PAH[idx] * (1. + 10000./2.998e5)]
                fluxMinMax = np.interp(waveMinMax, wave, flux_left)
                cont = 0.5 * (fluxMinMax[0] + fluxMinMax[1])
                ref_pah_peak = np.interp(wave0PAH[idx], wave, flux_left) - cont
                peakPAH_guess = peakPAH / peakPAH[idx] * ref_pah_peak

                break

        # If none of the listed PAHs are in the wavelength range
        if ref_pah_peak == 0.:
            #ref_pah_peak = np.interp(pah_inspection_order[0], wave, flux) * 1e-2
            #peakPAH_guess = peakPAH / peakPAH[0] * ref_pah_peak
            raise ValueError("Input spectrum does not contain any PAH band available to perform the PAH component scaling.")

        namePAH = []
        for i in range(len(wave0PAH)): namePAH.append(str(int(compPAH[i]))+'_PAH'+str(round(wave0PAH[i]*1e1))+'_'+str(round(wave0PAH[i]*1e4)))

        drude = [np.asarray(wave0PAH), np.asarray(gamPAH), np.asarray(peakPAH_guess), namePAH, compPAH]


        # ----------------
        # Get gaussian opacity features
        # ----------------
        opcTab = Table.read(tablePath+'/opacity/'+gopacity_table)

        oNames = opcTab['name'].value
        oWave0 = opcTab['wave'].value
        oGam = opcTab['gamma'].value
        oPeak = opcTab['peak'].value
        oMask = opcTab['mask'].value
        
        oNames_kept = [] ; oWave0_kept = [] ; oDoub_kept = [] ; oGam_kept = [] ; oPeak_kept = []
        for i in range(len(inst_df)):
            idx = ((oMask == 0) & (oWave0*(1+z) > inst_df.iloc[i].wMin) & (oWave0*(1+z) < inst_df.iloc[i].wMax) \
                   & (oWave0 >= minWave) & (oWave0 <= maxWave))
            oGam_kept, oGam = np.concatenate((oGam_kept, oGam[idx])), oGam[~idx]
            oNames_kept, oNames = np.concatenate((oNames_kept, oNames[idx])), oNames[~idx]
            oWave0_kept, oWave0 = np.concatenate((oWave0_kept, oWave0[idx])), oWave0[~idx]
            oPeak_kept, oPeak = np.concatenate((oPeak_kept, oPeak[idx])), oPeak[~idx]
            oMask = np.delete(oMask, idx, 0)
 
        for i, name in enumerate(oNames_kept): oNames_kept[i] = name.replace(')', '').replace('(', '').replace('-', '').replace(')', '')+'_'+str(round(oWave0_kept[i]*1e4))

        # Now the input values are not used at all. 
        #gauss_opc = [opcTab['wave'].value, opcTab['gamma'].value, opcTab['peak'].value, opcTab['name'].value]
        gauss_opc = [np.asarray(oWave0_kept), np.asarray(oGam_kept), np.asarray(oPeak_kept), oNames_kept]
        

        # RETURN THE GAUSS, DRUDE AND GAUSS OPACITY PARAMETERS
        return gauss, drude, gauss_opc
   

    @staticmethod
    def get_feats(params, errors=False, apply_vgrad2waves=False):
        ''' Turns lm parameters into  for flux computation
    
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
                    if fwave[-1] == 'B':
                        continue
                    else:
                        if errors is False:
                            lwave.append(params[key].value)
                            lgamma.append(params[key.replace('Wave','Gamma')].value)
                            lpeak.append(params[key.replace('Wave','Peak')].value)
                        else:
                            lwave.append(params[key].stderr)
                            lgamma.append(params[key.replace('Wave','Gamma')].stderr)
                            lpeak.append(params[key.replace('Wave','Peak')].stderr)
                        lname.append(fname+'_'+fwave[0:-1])
                        ldoub.append(1) if 'g_'+fname+'_'+fwave[0:-1]+'B_Wave' in pkeys else ldoub.append(0)
                        if apply_vgrad2waves is True: lwave[-1] *= (1+params['VGRAD']/2.998e5)
                elif key[-1] == 'a' or key[-1] == 'k': continue
                else:
                    raise ValueError('You messed with the feature parameter names')
                
            elif key[0] == 'd':
                if key[-1] == 'e':
                    fname = key[:-5]
                    if errors is False:
                        pwave.append(params[key].value)
                        pgamma.append(params[fname+'_Gamma'].value)
                        ppeak.append(params[fname+'_Peak'].value)
                    else:
                        pwave.append(params[key].stderr)
                        pgamma.append(params[fname+'_Gamma'].stderr)
                        ppeak.append(params[fname+'_Peak'].stderr)
                    pname.append(fname[1:])
                    pcomp.append(key.split('_')[0][1:])
                    if apply_vgrad2waves is True: pwave[-1] *= (1+params['VGRAD']/2.998e5)
                elif key[-1] == 'a' or key[-1] == 'k': continue
                else:
                    raise ValueError('You messed with the feature parameter names')

            elif key[0] == 'o':
                if key[-1] == 'e':
                    fname = key.split('_')[1]
                    fwave = key.split('_')[2]
                    if errors is False:
                        owave.append(params[key].value)
                        ogamma.append(params[key.replace('Wave','Gamma')].value)
                        opeak.append(params[key.replace('Wave','Peak')].value)
                    else:
                        owave.append(params[key].stderr)
                        ogamma.append(params[key.replace('Wave','Gamma')].stderr)
                        opeak.append(params[key.replace('Wave','Peak')].stderr)
                    oname.append(fname+'_'+fwave)
                    if apply_vgrad2waves is True: owave[-1] *= (1+params['VGRAD']/2.998e5)
                elif key[-1] == 'a' or key[-1] == 'k': continue
                else:
                    raise ValueError('You messed with the feature parameter names')
            else:
                continue
            
        gauss = [np.asarray(lwave), np.asarray(lgamma), np.asarray(lpeak), lname, np.asarray(ldoub)]
        drude = [np.asarray(pwave), np.asarray(pgamma), np.asarray(ppeak), pname, pcomp]
        gauss_opc = [np.asarray(owave), np.asarray(ogamma), np.asarray(opeak), oname]
        
        return gauss, drude, gauss_opc


    @staticmethod
    def make_cont_pars(inpars, parobj_update=False, init4fit=False, Onion=False):
        '''Makes the parameters structure for the continuum fit from the input dictionary

        Arguments:
        inpars -- dictionary of initial parameters/settings read from the runfile

        Returns: lm Parameters object for the CAFE continuum parameters
        '''
        params = lm.Parameters()
        npars = 0
        pkeys = []
        for key in inpars.keys():
            ### Just value and fit flag
            if len(inpars[key]) == 2:
                params.add(key, value=inpars[key][0], vary=bool(inpars[key][1]), min=-np.inf, max=np.inf)
                
            ### Min/max - must specify both (use np.inf if you only want one)
            elif len(inpars[key]) == 4 or (len(inpars[key])==5 and inpars[key][-1] is None):
                params.add(key, value=inpars[key][0], vary=bool(inpars[key][1]), min=inpars[key][2], max=inpars[key][3])
                
            ### Tie parameters - must also set bounds (can be inf)
            elif len(inpars[key]) == 5:
                try:
                    params.add(key, value=inpars[key][0], vary=bool(inpars[key][1]), min=inpars[key][2], max=inpars[key][3], expr=inpars[key][4])
                except NameError:
                    ### This happens when reloading a fit that had onion on, it confuses the expr parser
                    params.add(key, value=inpars[key][0], vary=bool(inpars[key][1]), min=inpars[key][2], max=inpars[key][3])
                    
            if parobj_update:
                params[key].value = parobj_update[key].value
                # If the parameter has been fixed to 0 by the fitter and the parameters are needed as initializers for fitting
                if params[key].value == 0 and params[key].vary != parobj_update[key].vary and init4fit != False:
                    params[key].value = 1e-3
                

        ### Force TAU_HOT > TAU_WRM > TAU_COO
        if Onion:
            params.add('HOT_WRM', value=inpars['HOT_TAU'][0]/inpars['WRM_TAU'][0], vary=True, min=1.0, max=np.inf)
            #params['HOT_TAU'] = lm.Parameter(name='HOT_TAU', value=inpars['HOT_TAU'][0], expr='HOT_WRM * WRM_TAU')
            params['HOT_TAU'].set(expr='HOT_WRM * WRM_TAU')
            if parobj_update: params['HOT_WRM'].value = parobj_update['HOT_WRM'].value
            
            params.add('WRM_COO', value=inpars['WRM_TAU'][0]/inpars['COO_TAU'][0], vary=True, min=1.0, max=np.inf)
            #params['COO_TAU'] = lm.Parameter(name='COO_TAU', value=inpars['COO_TAU'][0], expr='WRM_TAU / WRM_COO')
            params['COO_TAU'].set(expr='WRM_TAU / WRM_COO')
            if parobj_update: params['WRM_COO'].value = parobj_update['WRM_COO'].value


        return params


    @staticmethod
    def make_feat_pars(inpars, gauss, drude, gauss_opc, get_all=False, parobj_update=False, init4fit=False):
        ''' Turns lists of initial Gaussian and Drude profiles into a parameters object for the fitter.

        Arguments:
        inpars -- dictionary of input parameters read from inpars['PAH & LINE OPTIONS']
        gauss -- first guess at gaussian line profile parameters
        drude -- first guess at drude line profile parameters

        Returns: lm Parameters object for the CAFE feature parameters

        '''
        params = lm.Parameters()
        ### Dictionary of unique species (including ionization states)
        #states = {}
        ### dictionary of fixed-intensity ratio doublets - scale redder line from bluer
        ### Currently using ratio of A_ul values from NIST for OI6302 and NeIII3968. Check other lines,
        ### and check IR lists for previously unresolved doublets
        doublets = {'OIII_5008':['OIII_4960', 2.994], 'NII_6585':['NII_6550', 2.959], 'OI_6366':['OI_6302', 0.32],
                   'NeIII_3969':['NeIII_3870', 0.31], }
        complab = ['N','B']  # narrow and broad
        
        
        # Line features
        for i in range(gauss[0].size):
            ### May eventually want to replace hardcoded values
            if gauss[2][i] > 1e-7 or get_all == True or parobj_update != False:
                for j in range(int(gauss[4][i])+1):
                    
                    #name = gauss[3][i].replace('(', '').replace('-', '')
                    #name += '_'+str(round(gauss[0][i]*1e4))
                    name = gauss[3][i]

                    ### Do wavelengths ###
                    if inpars['EPSWAVE0_LIN_'+complab[j]] > 0:
                        maxW = gauss[0][i]*(1.+inpars['EPSWAVE0_LIN_'+complab[j]]/2.998e5) # EPSWAVE0_LIN in [km/s]
                        minW = gauss[0][i]*(1.-inpars['EPSWAVE0_LIN_'+complab[j]]/2.998e5)
                    else:
                        maxW =  np.inf
                        minW = -np.inf
                    params.add('g_'+name+complab[j]+'_Wave', value=gauss[0][i], vary=inpars['FITWAVE0_LIN_'+complab[j]], min=minW, max=maxW)
                    if parobj_update:
                        pass
                        # Wavelengths are different because the VGRAD has been applied to the feature wavelengths 
                        #if params['g_'+name+complab[j]+'_Wave'].value != parobj_update['g_'+name+complab[j]+'_Wave'].value: ipdb.set_trace()
                        #if 'g_'+name+complab[j]+'_Wave' in parobj_update.keys():
                        #params['g_'+name+complab[j]+'_Wave'].value = parobj_update['g_'+name+complab[j]+'_Wave'].value
                    
                    ### Do widths (gammas) ###
                    if inpars['EPSGAMMA_LIN_'+complab[j]] > 0:
                        maxG = np.max([gauss[1][i]*(1.+0.05), inpars['EPSGAMMA_LIN_'+complab[j]]/2.998e5]) # EPSGAMMA_LIN in [km/s]
                        minG = gauss[1][i]*(1.-0.05) # Minimum allowed is set to 1/1.05 the resolution (gamma)
                    else:
                        maxG = np.inf
                        minG = gauss[1][i]*(1.-0.05) # Minimum allowed is set to 1/1.05 the resolution (gamma)
                    params.add('g_'+name+complab[j]+'_Gamma', value=gauss[1][i]*(2*j+1), vary=inpars['FITGAMMA_LIN_'+complab[j]], min=minG, max=maxG) # The i'th component is initialized to have a width 2*j+1 * instrumental FWHM
                    if parobj_update:
                        if j == 0 and params['g_'+name+complab[j]+'_Gamma'].value != parobj_update['g_'+name+complab[j]+'_Gamma'].value: ipdb.set_trace()
                        #if 'g_'+name+complab[j]+'_Gamma' in parobj_update.keys():
                        #params['g_'+name+complab[j]+'_Gamma'].value = parobj_update['g_'+name+complab[j]+'_Gamma'].value
                    
                    ### Do amplitudes ###
                    if inpars['EPSPEAK_LIN'] > 0:
                        maxP = gauss[2][i]*(1.+inpars['EPSPEAK_LIN']) #EPSPEAK_LIN in fraction
                        minP = gauss[2][i]/(1.+inpars['EPSPEAK_LIN'])
                    else:
                        maxP = np.inf
                        minP = 0.
                    params.add('g_'+name+complab[j]+'_Peak', value=gauss[2][i]/(3*j+1), vary=True, min=minP, max=maxP) # The i'th component is initialized to have an amplitude = main component / (3*j+1) 
                    if parobj_update:
                        if j == 0 and params['g_'+name+complab[j]+'_Peak'].value != parobj_update['g_'+name+complab[j]+'_Peak'].value: ipdb.set_trace()
                        if parobj_update['g_'+name+complab[j]+'_Peak'].value == 0 and params['g_'+name+complab[j]+'_Peak'].vary != parobj_update['g_'+name+complab[j]+'_Peak'].vary and init4fit != False:
                            params['g_'+name+complab[j]+'_Peak'].value = 1e-5
                        #if 'g_'+name+complab[j]+'_Peak' in parobj_update.keys():
                        #params['g_'+name+complab[j]+'_Peak'].value = parobj_update['g_'+name+complab[j]+'_Peak'].value

                    ### If in fixed doublet, force the intensity ratio
                    if name in list(doublets.keys()):
                        params['g_'+name+complab[j]+'_Peak'].set(expr='g_'+doublets[name][0]+complab[j]+'_Peak*'+str(doublets[name][1]))


        # Drude features
        for i in range(drude[0].size):
            if drude[1][i] > 0 or get_all == True or parobj_update != False:

                #name = str(round(drude[0][i]*1e1))
                #name += '_'+str(round(drude[0][i]*1e4))
                name = drude[3][i]

                if inpars['EPSWAVE0_PAH'] > 0:
                    maxW = drude[0][i]*(1.+inpars['EPSWAVE0_PAH']/2.998e5)
                    minW = drude[0][i]*(1.-inpars['EPSWAVE0_PAH']/2.998e5)
                else:
                    maxW =  np.inf
                    minW = -np.inf
                params.add('d'+name+'_Wave',  value=drude[0][i], vary=inpars['FITWAVE0_PAH'], min=minW, max=maxW)
                if parobj_update:
                    pass
                    # Wavelengths are different because the VGRAD has been applied to the feature wavelengths 
                    #if params['d'+name+'_Wave'].value != parobj_update['d'+name+'_Wave'].value: ipdb.set_trace()
                    #if 'd'+name+'_Wave' in parobj_update.keys():
                    #params['d'+name+'_Wave'].value = parobj_update['d'+name+'_Wave'].value

                if inpars['EPSGAMMA_PAH'] > 0:
                    maxG = drude[1][i]*(1.+inpars['EPSGAMMA_PAH']) # EPSGAMMA_PAH in fraction
                    minG = drude[1][i]/(1.+inpars['EPSGAMMA_PAH'])
                else:
                    maxG =  np.inf
                    minG = 0.
                params.add('d'+name+'_Gamma', value=drude[1][i], vary=inpars['FITGAMMA_PAH'], min=minG, max=maxG)
                if parobj_update:
                    if params['d'+name+'_Gamma'].value != parobj_update['d'+name+'_Gamma'].value: ipdb.set_trace()
                    #if 'd'+name+'_Gamma' in parobj_update.keys():
                    #params['d'+name+'_Gamma'].value = parobj_update['d'+name+'_Gamma'].value

                if inpars['EPSPEAK_PAH'] > 0:
                    maxP = drude[2][i]*(1.+inpars['EPSPEAK_PAH']) #EPSPEAK_PAH in fraction
                    minP = drude[2][i]/(1.*inpars['EPSPEAK_PAH'])
                else:
                    maxP =  np.inf
                    minP =  0.
                params.add('d'+name+'_Peak',  value=drude[2][i], vary=True,  min=minP, max=maxP)
                if parobj_update:
                    if params['d'+name+'_Peak'].value != parobj_update['d'+name+'_Peak'].value: ipdb.set_trace()
                    if parobj_update['d'+name+'_Peak'].value == 0 and params['d'+name+'_Peak'].vary != parobj_update['d'+name+'_Peak'].vary and init4fit != False:
                        params['d'+name+'_Peak'].value = 1e-5
                    #if 'd'+name+'_Peak' in parobj_update.keys():
                    #params['d'+name+'_Peak'].value = parobj_update['d'+name+'_Peak'].value


        # Opacity features
        for i in range(gauss_opc[0].size):
            if gauss_opc[1][i] > 0 or get_all == True or parobj_update != False:

                #name = gauss_opc[3][i].replace('(', '').replace('-', '').replace(')', '')
                #name += '_'+str(round(gauss_opc[0][i]*1e4))
                name = gauss_opc[3][i]

                ### Do wavelengths ###
                if inpars['EPSWAVE0_OPC'] > 0:
                    maxW = gauss_opc[0][i]*(1.+inpars['EPSWAVE0_OPC']/2.998e5) # EPSWAVE0_OPC in [km/s]
                    minW = gauss_opc[0][i]*(1.-inpars['EPSWAVE0_OPC']/2.998e5)
                else:
                    maxW =  np.inf
                    minW = -np.inf
                params.add('o_'+name+'_Wave', value=gauss_opc[0][i], vary=inpars['FITWAVE0_OPC'], min=minW, max=maxW)
                if parobj_update:
                    pass
                    # Wavelengths are different because the VGRAD has been applied to the feature wavelengths 
                    #if params['o_'+name+'_Wave'].value != parobj_update['o_'+name+'_Wave'].value: ipdb.set_trace()
                    #if 'o_'+name+'_Wave' in parobj_update.keys():
                    #params['o_'+name+'_Wave'].value = parobj_update['o_'+name+'_Wave'].value
                    
                ### Do widths (gammas) ###
                if inpars['EPSGAMMA_OPC'] > 0:
                    maxG = gauss_opc[1][i]*(1.+inpars['EPSGAMMA_OPC']) # EPSGAMMA_OPC in fraction
                    minG = gauss_opc[1][i]/(1.+inpars['EPSGAMMA_OPC']) # Minimum allowed is set to 1/1.05 the resolution (gamma)
                else:
                    maxG = np.inf
                    minG = 0.
                params.add('o_'+name+'_Gamma', value=gauss_opc[1][i], vary=inpars['FITGAMMA_OPC'], min=minG, max=maxG) # The i'th component is initialized to have a width 2*j+1 * instrumental FWHM
                if parobj_update:
                    if params['o_'+name+'_Gamma'].value != parobj_update['o_'+name+'_Gamma'].value: ipdb.set_trace()
                    #if 'o_'+name+'_Gamma' in parobj_update.keys():
                    params['o_'+name+'_Gamma'].value = parobj_update['o_'+name+'_Gamma'].value
                    
                ### Do amplitudes ###
                if inpars['EPSPEAK_OPC'] > 0:
                    maxP = inpars['EPSPEAK_OPC'] #EPSPEAK_OPC in absolute
                    minP = 0.
                else:
                    maxP =  np.inf
                    minP =  0.
                params.add('o_'+name+'_Peak', value=gauss_opc[2][i], vary=True, min=minP, max=maxP) # The i'th component is initialized to have an amplitude = main component / (3*j+1) 
                if parobj_update:
                    if params['o_'+name+'_Peak'].value != parobj_update['o_'+name+'_Peak'].value: idpb.set_trace()
                    if parobj_update['o_'+name+'_Peak'].vary == 0 and params['o_'+name+'_Peak'].vary != parobj_update['o_'+name+'_Peak'].vary and init4fit != False:
                        params['o_'+name+'_Peak'].value = 1e-5
                    #if 'o_'+name+'_Peak' in parobj_update.keys():
                    #params['o_'+name+'_Peak'].value = parobj_update['o_'+name+'_Peak'].value


        # Parameter that allows for the wavelength of all emission features to vary uniformly, simulating any potential velocity gradient
        params.add('VGRAD', value=1., vary=inpars['FITVGRAD'], min=-1.*inpars['EPSVGRAD'], max=1.*inpars['EPSVGRAD'])
        #if parobj_update:
        #    # WARNING: if parobj_update is True, VGRAD will be updated even if the value has already been applied to the line wavelengths!
        #    if params['VGRAD'].value != parobj_update['VGRAD'].value: ipdb.set_trace()
        #    params['VGRAD'].value = parobj_update['VGRAD'].value
        #    params['VGRAD'].vary = inpars['FITVGRAD']
        #    params['VGRAD'].min = params['VGRAD'].value - 1.*inpars['EPSVGRAD']
        #    params['VGRAD'].max = params['VGRAD'].value + 1.*inpars['EPSVGRAD']


        return params



class CAFE_prof_generator:
    """
    Class for loading all the needed profiles in CAFE.

    Parameters
    ----------
    wave: np.array
        input wavelength
    inpars: dict
        lmfit parameters object
    optfile:

    """
    def __init__(self, spec, inparfile, optfile, cafe_path='../CAFE/'):
        ## Read spec
        wave = spec.spectral_axis.value
        flux = spec.flux.value
        flux_unc = spec.uncertainty.quantity.value
        self.wave = wave
        self.flux = flux
        self.flux_unc = flux_unc

        #waveSED = wave

        # Read optfile
        inpars = cafeio.read_inifile(inparfile)
        inopts = cafeio.read_inifile(optfile)

        self.inpars = inpars
        self.inopts = inopts
        tablePath, _ = cafeio.init_paths(inopts, cafe_path=cafe_path)
        self.tablePath = tablePath

        # Define blackbody temperatures of ambient radiation field
        #self.T_bb = np.geomspace(3., 1750., num=30)
        # To make a temperature array that is finer at the high temperature end
        self.T_bb = np.geomspace(3.+30., 1750.+30., num=30)-30.
        
        # Defining waveSED vector. This is taken directly from IDL
        #waveSpecSED = np.geomspace(np.nanmin(wave), np.nanmax(wave), num=wave.size)
        #samplingSpec = np.log10(np.nanmax(waveSpecSED)/np.nanmin(waveSpecSED))/waveSpecSED.size # The Delta_dex
        samplingSpec = np.log10(np.nanmax(wave)/np.nanmin(wave))/wave.size

        #samplingUV = 20. # This is a downsampling factor wrt the spectroscopic sampling
        #minUV = 1e-3
        #maxUV = 0.3
        #nUV = int(np.ceil(np.log10(maxUV/minUV)/(samplingUV*samplingSpec)))
        #waveUV = np.geomspace(minUV, maxUV, num=nUV)
        
        samplingNIR = 4.
        minNIR = 0.7 #1. # originally 0.3
        maxNIR = wave[0] - (wave[1]-wave[0])
        nNIR = int(np.ceil(np.log10(maxNIR/minNIR)/(samplingNIR*samplingSpec)))
        waveNIR = np.geomspace(minNIR, maxNIR, num=nNIR)

        #samplingFIR = 4.
        #minFIR = wave[-1] + wave[-1] - wave[-2]
        #maxFIR = 1.3e3
        #nFIR = int(np.ceil(np.log10(maxFIR/minFIR)/(samplingFIR*samplingSpec)))
        #waveFIR = np.geomspace(minFIR, maxFIR, num=nFIR)
        
        if wave[-1] <= 11:
            samplingMIR = 4
            minMIR = wave[-1] + (wave[-1]-wave[-2])
            maxMIR = 11.
            nMIR = int(np.ceil(np.log10(maxMIR/minMIR)/(samplingMIR*samplingSpec)))
            waveMIR = np.geomspace(minMIR, maxMIR, num=nMIR)
            
            waveSED = np.concatenate((waveNIR, wave, waveMIR)) # waveUV, 

        else:
            #waveSED = np.concatenate((waveUV, waveNIR, wave, waveFIR)) # waveUV, 
            waveSED = np.concatenate((waveNIR, wave)) # waveUV, 
            #waveSED = np.sort(waveSED)
            
            
        # Remove lam that are too close
        close_lam_sampling = []
        for i in range(waveSED.size-1):
            if waveSED[i+1] - waveSED[i] < 1e-14: close_lam_sampling.append(i)
        waveSED = np.delete(waveSED, close_lam_sampling)

        self.waveSED = waveSED


    def load_grain_emissivity(self):
        waveSED = self.waveSED
        inpars = self.inpars
        inopts = self.inopts
        tablePath = self.tablePath
        T_bb = self.T_bb

        # Get the source of the continuum component
        srcs = inpars['COMPONENT SOURCE SEDs']
        sourceTypes = [srcs[key] for key in srcs.keys()]

        # Get silicate abosorption. This should be moved to opacity!!!!
        scaleOHMc = np.genfromtxt(tablePath+'/opacity/ohmc_scale.txt', comments=';')
        if inopts['MODEL OPTIONS']['DRAINE_OR_OHMC'] != 'OHMc':
            scaleOHMc[:,1] = np.ones(scaleOHMc.shape[1])
        
        if 'AGN' in sourceTypes:
            E_AGN = grain_emissivity(waveSED, T_bb, 'AGN', scaleOHMc, tablePath)
        if 'ISRF' in sourceTypes:
            E_ISRF = grain_emissivity(waveSED, T_bb, 'ISRF', scaleOHMc, tablePath)
        if 'SB2Myr' in sourceTypes:
            E_SB2Myr = grain_emissivity(waveSED, T_bb, 'SB2Myr', scaleOHMc, tablePath)
        if 'SB10Myr' in sourceTypes:
            E_SB10Myr = grain_emissivity(waveSED, T_bb, 'SB10Myr', scaleOHMc, tablePath)
        if 'SB100Myr' in sourceTypes:
            E_SB100Myr = grain_emissivity(waveSED, T_bb, 'SB100Myr', scaleOHMc, tablePath)
        
        if srcs['SOURCE_HOT'] == 'AGN': E_HOT = E_AGN
        elif srcs['SOURCE_HOT'] == 'ISRF': E_HOT = E_ISRF
        elif srcs['SOURCE_HOT'] == 'SB2Myr': E_HOT = E_SB2Myr
        elif srcs['SOURCE_HOT'] == 'SB10Myr': E_HOT = E_SB10Myr
        elif srcs['SOURCE_HOT'] == 'SB100Myr': E_HOT = E_SB100Myr
        else: raise ValueError('Invalid Source_HOT')
        if srcs['SOURCE_WRM'] == 'AGN': E_WRM = E_AGN
        elif srcs['SOURCE_WRM'] == 'ISRF': E_WRM = E_ISRF
        elif srcs['SOURCE_WRM'] == 'SB2Myr': E_WRM = E_SB2Myr
        elif srcs['SOURCE_WRM'] == 'SB10Myr': E_WRM = E_SB10Myr
        elif srcs['SOURCE_WRM'] == 'SB100Myr': E_WRM = E_SB100Myr
        else: raise ValueError('Invalid Source_WRM')
        if srcs['SOURCE_COO'] == 'AGN': E_COO = E_AGN
        elif srcs['SOURCE_COO'] == 'ISRF': E_COO = E_ISRF
        elif srcs['SOURCE_COO'] == 'SB2Myr': E_COO = E_SB2Myr
        elif srcs['SOURCE_COO'] == 'SB10Myr': E_COO = E_SB10Myr
        elif srcs['SOURCE_COO'] == 'SB100Myr': E_COO = E_SB100Myr
        else: raise ValueError('Invalid Source_COO')
        if srcs['SOURCE_CLD'] == 'AGN': E_CLD = E_AGN
        elif srcs['SOURCE_CLD'] == 'ISRF': E_CLD = E_ISRF
        elif srcs['SOURCE_CLD'] == 'SB2Myr': E_CLD = E_SB2Myr
        elif srcs['SOURCE_CLD'] == 'SB10Myr': E_CLD = E_SB10Myr
        elif srcs['SOURCE_CLD'] == 'SB100Myr': E_CLD = E_SB100Myr
        else: raise ValueError('Invalid Source_CLD')
        if srcs['SOURCE_CIR'] == 'AGN': E_CIR = E_AGN
        elif srcs['SOURCE_CIR'] == 'ISRF': E_CIR = E_ISRF
        elif srcs['SOURCE_CIR'] == 'SB2Myr': E_CIR = E_SB2Myr
        elif srcs['SOURCE_CIR'] == 'SB10Myr': E_CIR = E_SB10Myr
        elif srcs['SOURCE_CIR'] == 'SB100Myr': E_CIR = E_SB100Myr
        else: 
            raise ValueError('Invalid Source_CIR')

        result = {'E_CIR': E_CIR,
                  'E_COO': E_COO,
                  'E_CLD': E_CLD,
                  'E_WRM': E_WRM,
                  'E_HOT': E_HOT,
                  }

        #ipdb.set_trace()
        # Each emissivity returns:
        # {'wave':waveSED, 't_bb':T_bb, 'SilTot':eSilTotOut, 'SilCont':eSilContOut, 'AmoFeat':eAmoFeatOut,
        #    'FstFeat':eFstFeatOut, 'EnsFeat':eEnsFeatOut, 'Carb':eCarbOut}
        return result


    def load_grain_opacity(self):
        waveSED = self.waveSED
        inpars = self.inpars
        inopts = self.inopts
        tablePath = self.tablePath
        T_bb = self.T_bb

        # Load additional opacity sources and Draine/OHMc scaling factors
        scaleOHMc = np.genfromtxt(tablePath+'/opacity/ohmc_scale.txt', comments=';')
        if inopts['MODEL OPTIONS']['DRAINE_OR_OHMC'] != 'OHMc':
            scaleOHMc[:,1] = np.ones(scaleOHMc.shape[1])
        if not inopts['SWITCHES']['ORION_H2O']:
            kIce3 = load_opacity(waveSED, tablePath+'opacity/ice_opacity_idl_3um_upsampled.txt')
        else:
            kIce3 = load_opacity(waveSED, tablePath+'opacity/ice_opacity_idl_orion.txt')
        kIce6 = load_opacity(waveSED, tablePath+'opacity/ice_opacity_idl_6um_upsampled.txt')
        kHac = load_opacity(waveSED, tablePath+'opacity/hac_opacity_upsampled.txt')
        kCOrv = load_opacity(waveSED, tablePath+'opacity/corv_opacity_upsampled.txt')
        kCO2 = load_opacity(waveSED, tablePath+'opacity/CO2_opacity_4um.ecsv')
        kCrySi_233 = load_opacity(waveSED, tablePath+'opacity/crystallineSi_opacity_233.ecsv')

        # Temperature is set to 0 to get grain opacities
        kAbs, kExt = grain_opacity(waveSED, 0., scaleOHMc, tablePath, noPAH=False) #, cutoff='big'

        # Consider absorption or absorption+scattering 
        Ext_or_Abs = inopts['MODEL OPTIONS']['EXTORABS']

        result = {'wave': waveSED,
                  'kIce3': kIce3,
                  'kIce6': kIce6,
                  'kHac': kHac,
                  'kCOrv': kCOrv,
                  'kCO2': kCO2,
                  'kCrySi_233': kCrySi_233,
                  'kAbs': kAbs,
                  'kExt': kExt,
                  }

        # kExt returns
        # {'wave':wave, 't_bb':T_bb, 'SilCont':kExtSilCont,
        #           'SilAmoTot':kExtSilTot, 
        #           'SilAmoFeat':kExtAmoFeat,
        #           'SilFstFeat':kExtFstFeat,
        #           'SilEnsFeat':kExtEnsFeat,
        #           'Carb':kExtCarb}
        return result


    def get_sed(self):
        waveSED = self.waveSED
        tablePath = self.tablePath        

        source2Myr = sourceSED(waveSED, 'SB2Myr', tablePath, norm=True, Jy=True)[1]
        source10Myr = sourceSED(waveSED, 'SB10Myr', tablePath, norm=True, Jy=True)[1]
        source100Myr = sourceSED(waveSED, 'SB100Myr', tablePath, norm=True, Jy=True)[1]
        sourceStr = sourceSED(waveSED, 'ISRF', tablePath, norm=True, Jy=True)[1]
        sourceDsk = sourceSED(waveSED, 'AGN', tablePath, norm=True, Jy=True)[1]

        ## Neet to convert to um for integral for normalization
        #const = 3e14/waveSED
        #source2Myr/=np.trapz(const*source2Myr, np.log(waveSED))
        #source10Myr/=np.trapz(const*source10Myr, np.log(waveSED))
        #source100Myr/=np.trapz(const*source100Myr, np.log(waveSED))
        #sourceStr/=np.trapz(const*sourceStr, np.log(waveSED))
        #sourceDsk/=np.trapz(const*sourceDsk, np.log(waveSED))

        #ipdb.set_trace()
        result = {'source2Myr': source2Myr,
                  'source10Myr': source10Myr,
                  'source100Myr': source100Myr,
                  'sourceStr': sourceStr,
                  'sourceDsk': sourceDsk,
                 }

        return result


    def make_cont_profs(self):
        
        inopts = self.inopts
        wave = self.wave
        flux = self.flux
        
        # waveSED
        # -------
        waveSED = self.waveSED
        
        # wave0
        # -----
        # Read reference wavelength (wave0) of the continuum components together with 
        # PAH listed in the optfile and output reference wavelength dictionary, 
        # i.e., 'WAVE_CIR' -> 'CIR'
        wave0 = {}
        for key in inopts['REFERENCE WAVELENGTHS'].keys():
            keystr = str(key)
            wave0[keystr.split('_')[1]] = inopts['REFERENCE WAVELENGTHS'][key]
            
        # flux0
        # -----
        # Read in reference wavelengths, then get the measured flux at each of those wavelengths
        # Note np.interp does one-dimensional linear interpolation for monotonically 
        # increasing sample points.    
        flux0 = {'CIR':np.interp(np.log(wave0['CIR']), np.log(wave), flux),
                 'CLD':np.interp(np.log(wave0['CLD']), np.log(wave), flux),
                 'COO':np.interp(np.log(wave0['COO']), np.log(wave), flux),
                 'WRM':np.interp(np.log(wave0['WRM']), np.log(wave), flux),
                 'HOT':np.interp(np.log(wave0['HOT']), np.log(wave), flux),
                 'STR':np.interp(np.log(wave0['STR']), np.log(wave), flux),
                 'STB':np.interp(np.log(wave0['STB']), np.log(wave), flux),
                 'DSK':np.interp(np.log(wave0['DSK']), np.log(wave), flux),
                 'PAH':np.interp(np.log(wave0['PAH']), np.log(wave), flux)}
        
        # =================
        # Grain emissivity
        # =================
        grain_emissivity_dict = self.load_grain_emissivity()
        
        E_CIR = grain_emissivity_dict['E_CIR']
        E_COO = grain_emissivity_dict['E_COO']
        E_CLD = grain_emissivity_dict['E_CLD']
        E_WRM = grain_emissivity_dict['E_WRM']
        E_HOT = grain_emissivity_dict['E_HOT']
        
        # ========================================================================
        # Opacity profiles (kIce3, kIce6, kHac, kCOrv) & grain opacity (kAbs, kExt)
        # ========================================================================
        grain_opacity_dict = self.load_grain_opacity()
        
        kIce3 = grain_opacity_dict['kIce3']
        kIce6 = grain_opacity_dict['kIce6']
        kHac = grain_opacity_dict['kHac']
        kCOrv = grain_opacity_dict['kCOrv']
        kCO2 = grain_opacity_dict['kCO2']
        kCrySi_233 = grain_opacity_dict['kCrySi_233']
        kAbs = grain_opacity_dict['kAbs']
        kExt = grain_opacity_dict['kExt']
        
        # ========
        # Load SED
        # ========
        sed = self.get_sed()
        
        source2Myr = sed['source2Myr']
        source10Myr = sed['source10Myr']
        source100Myr = sed['source100Myr']
        sourceStr = sed['sourceStr']
        sourceDsk = sed['sourceDsk']
        
        #ipdb.set_trace()
        # ---
        cont_profs = {'waveSED':waveSED, # wavelength sampling
                      'wave0':wave0, 'flux0':flux0, # reference wavelengths and fluxes at those waves of continuum features
                      
                      'E_CIR':E_CIR, 'E_COO':E_COO, 'E_CLD':E_CLD, 'E_WRM':E_WRM, 'E_HOT':E_HOT, # dust emissivities
                      
                      'kIce3':kIce3, 'kIce6':kIce6, 'kHac':kHac, 'kCOrv':kCOrv, 'kCO2':kCO2, # molecules and non-amorphous dust opacity
                      'kCrySi_233': kCrySi_233, # crystalline silicate opacity
                      'kAbs':kAbs, 'kExt':kExt, # (Carb, Sil) grain opacity
                      
                      'source100Myr':source100Myr, 'source10Myr':source10Myr, # SED
                      'source2Myr':source2Myr, 'sourceDSK':sourceDsk, 'sourceSTR':sourceStr, # SED
                      
                      #'pwaves':pwave, 'filters':pfilt, # photometry info
                      'pwaves':None, 'filters':None, # tmp - Don't consider photometry
                      
                      'DoFilter':inopts['MODEL OPTIONS']['DOFILTER'], # setting
                      'ExtOrAbs':inopts['MODEL OPTIONS']['EXTORABS'], # setting
                      'FASTTEMP':inopts['FIT OPTIONS']['FASTTEMP'], # setting
                      'z': 0.,
        } 
        
        return cont_profs



class CAFE_parcube_generator:

    def __init__(self, cube, params, inparfile, optfile):

        self.parcube_header = cube.header #cube['FLUX'].header
        self.nx = cube.nx #cube[extract].header['NAXIS1']
        self.ny = cube.ny #cube[extract].header['NAXIS2']
        self.params = params
        self.inparfile = inparfile
        self.optfile = optfile


    def make_parcube(self):    

        keys = list(self.params.keys())
        
        ini_parcube = np.full((len(keys), self.ny, self.nx), np.nan)
        primary_hdu = fits.PrimaryHDU() # What should we put in primary?
        
        parcube = fits.HDUList(primary_hdu)

        name_list = ['VALUE', 'STDERR', 'VARY', 'MIN', 'MAX']
        for name in name_list:
            globals()[name] = fits.ImageHDU(ini_parcube.copy(), name=name, header=self.parcube_header)
            parcube.append(globals()[name])
        
        # create Bin table to store EXPR and PARNAME
        #expr_arr = np.full((len(keys), self.ny, self.nx), '')
        #expr_rec = np.rec.array((expr_arr), formats='U24', names='expr')
        
        #expr = fits.BinTableHDU(data=expr_rec.copy(), name='EXPR')
        #expr = fits.FITS_rec(btbl_expr.copy(), name='EXPR')
        
        expr_arr = np.full((len(keys)), '')
        expr_tbl = [(i, k) for i, k in zip(np.arange(len(keys)), expr_arr)]
        expr_rec = np.rec.array(expr_tbl, formats='int16, U24', names='idx, expr')
        
        expr = fits.BinTableHDU(expr_rec.copy(), name='EXPR')
        parcube.append(expr)
        
        parname_tbl = [(i, k) for i, k in zip(np.arange(len(keys)), keys)]
        parname_rec = np.rec.array(parname_tbl, formats='int16, U32', names='idx, parname')

        parname = fits.BinTableHDU(parname_rec.copy(), name='PARNAME')
        parcube.append(parname)
        
        return parcube

    
def parobj2parcube(parobj, parcube, x=0, y=0):
    """
    Insert values in the Parameter object into an existing parcube
    
        Parameters
        ----------
        parobj: (Parameter object)
        parcube: (hdul)
        x, y: 
            spaxel coordinate    
    """

    # Check parobj is a Parameter object
    if not isinstance(parobj, lm.Parameters):
        raise TypeError("Input parobj should be a Parameter object")
    if not isinstance(parcube, fits.HDUList):
        raise TypeError("Input parcube should be a HDUList")

    for parname in parobj.keys():
        value = parobj[parname].value
        stderr = parobj[parname].stderr  # will keep nan if stderr is None
        vary = parobj[parname].vary  # will be 0 if vary is False or 1 if it's True
        vmin = parobj[parname].min
        vmax = parobj[parname].max
        expr = parobj[parname].expr  # 

        try:
            ind = parcube['PARNAME'].data['parname'].tolist().index(parname) # find the index (parameter) that matches the parname in the z dimention
        except:
            ipdb.set_trace()
            raise ValueError('Parameter in parameter object is not in parameter cube, which should have all the parameters')
        else:
            for i, item in zip(['VALUE', 'STDERR', 'VARY', 'MIN', 'MAX'], [value, stderr, vary, vmin, vmax]):
                parcube[i].data[ind, y, x] = item
            parcube['EXPR'].data['expr'][ind] = expr
        
    return parcube



def parcube2parobj(parcube, x=0, y=0, parobj=None):
    """
    Turn parameters in parcube at a specific spaxel into a Parameter object
    
        Parameters
        ----------
        parcube: Parameter cube (hdul)
        x, y: Spaxel coordinate
        parobj: Parameter object (lmfit parameters)
    
        If parobj is provided the parcube parameters that are in parobj will be injected in parobj
        and len(parobj) will have len(parobj) and not len(parcube)
    """

    if parobj is None:
        parobj = lm.Parameters()

        for z, parname in enumerate(parcube['PARNAME'].data['parname']):
            if np.isnan(parcube['VALUE'].data[z, y, x]) == False:
                value = parcube['VALUE'].data[z, y, x]
                stderr = None if np.isnan(parcube['STDERR'].data[z, y, x]) else parcube['STDERR'].data[z, y, x]
                vary = False if parcube['VARY'].data[z, y, x] == 0. else True
                vmin = parcube['MIN'].data[z, y, x]
                vmax = parcube['MAX'].data[z, y, x]
                expr = None if parcube['EXPR'].data['expr'][z] == 'None' else parcube['EXPR'].data['expr'][z]
                
                parobj.add(parname, value=value, vary=vary, min=vmin, max=vmax, expr=expr)
                parobj[parname].stderr = stderr

    else:
        count=0
        for z, parname in enumerate(parcube['PARNAME'].data['parname']):
            try:
                ind = list(parobj.keys()).index(parname)
            except:
                continue
            else:
                if ~np.isnan(parcube['VALUE'].data[z, y, x]):
                    parobj[parname].value = parcube['VALUE'].data[z, y, x]
                    parobj[parname].stderr =  None if np.isnan(parcube['STDERR'].data[z, y, x]) else parcube['STDERR'].data[z, y, x]
                    parobj[parname].vary = False if parcube['VARY'].data[z, y, x] == 0. else True
                    parobj[parname].min = parcube['MIN'].data[z, y, x]
                    parobj[parname].max = parcube['MAX'].data[z, y, x]
                    parobj[parname].expr = None if parcube['EXPR'].data['expr'][z] == 'None' else parcube['EXPR'].data['expr'][z]
                else:
                    raise ValueError(parname+' in parcube is NaN')
                    
                count += 1

        if count != len(parobj.keys()):
            ipdb.set_trace()
            raise ValueError('Some parameters of parjobj have been eliminated when they should not have been')

    return parobj



def parobj2df(parobj):
    """
        Convert a parameter object to a DataFrame
        """
    key_list = []
    value_list = []
    stderr_list = []
    
    for k in parobj.keys():
        value_list.append(parobj[k].value)
        stderr_list.append(parobj[k].stderr)
        
    df = pd.DataFrame({'parname': list(parobj.keys()), 
                       'value': value_list, 
                       'stderr': stderr_list})
    df.set_index('parname', inplace=True)
    
    return df


def parcube2df(parcube, x, y):
    """
        Convert a parameter object to a DataFrame
        """
    key_list = []
    value_list = []
    stderr_list = []
    
    for z, key in enumerate(parcube['PARNAME'].data['parname']):
        value_list.append(parcube['VALUE'].data[z,y,x])
        stderr_list.append(parcube['STDERR'].data[z,y,x])
        
    df = pd.DataFrame({'parname': list(parcube['PARNAME'].data['parname']), 
                       'value': value_list, 
                       'stderr': stderr_list})
    df.set_index('parname', inplace=True)
    
    return df
