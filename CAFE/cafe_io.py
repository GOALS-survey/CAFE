import sys
import os
import warnings
import numpy as np
import configparser
import ast
from os.path import exists
import pandas as pd
from specutils import Spectrum1D, SpectrumList
import astropy
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from astropy.io import fits
import asdf
from asdf import AsdfFile

import CAFE
from CAFE.component_model import pah_drude, gauss_prof, drude_prof, drude_int_fluxes

#import ipdb


class cafe_io:

    def __init__(self):
        
        pass


    @staticmethod
    def init_paths(inopts, cafe_path=None, file_name=None, output_path=None):
        # Path to load external tables
        if not inopts['PATHS']['TABPATH']:
            tablePath = cafe_path+'tables/'
        else: 
            tablePath = inopts['PATHS']['TABPATH']
        
        # Create an output directory if necessary
        if output_path is None:
            if inopts['PATHS']['OUTPATH']:
                if not os.path.exists(inopts['PATHS']['OUTPATH']):
                    os.mkdir(inopts['PATHS']['OUTPATH'])
                outPath = inopts['PATHS']['OUTPATH']
            else:
                outPath = cafe_path+'output/'
        else:
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            if file_name is not None:
                if not os.path.exists(output_path+file_name):
                    os.mkdir(output_path+file_name)
                outPath = output_path+file_name+'/'
            else:
                outPath = output_path

        return tablePath, outPath

    
    ##### Function that reads a multi-extension .fits file from CRETA           ###
    ###############################################################################    
    # @file_name: The fits filename . (string)
    ########### --> Return res_spec1d  ############################
        
    def read_cretacube(self, file_name, extract):
        
        cube = fits.open(file_name)
        cube.info()
        
        self.cube = cube
        self.header = cube['FLUX'].header
        self.waves = cube['Wave'].data
        self.fluxes = cube[extract].data
        self.flux_uncs = cube[extract.replace('Flux','Err')].data
        self.masks = cube['DQ'].data
        self.bandnames = cube['Band_name'].data['Band_name']
        if self.fluxes.ndim != 1:
            self.nz, self.ny, self.nx = self.fluxes.shape
        else:
            self.nz, self.ny, self.nx = self.fluxes.shape, 1, 1

        cube.close()

        return self

        
    ##### Function that reads a TABLE .fits file from CRETA           ###
    ###############################################################################    
    # @filename: The fits filename . (string)
    ########### --> Return res_spec1d  ############################
    # @res_spec1d: A list of extracted spectra. (list of Spectrum1D)        
    ###############################################################################   
    
    @staticmethod
    def customFITSReader(file_name, extract): 
        
        # Options for spec are:
        # 'ap': Aperture flux
        # 'ap_PSC': Aperture flux point-source corrected
        # 'ap_st': Aperture flux stitched (if PSC has been applied, that is the stitched spectrum)
        
        hdu_list = fits.open(file_name)
        res_spec1d = []

        for i in range(len(hdu_list[1].data)):
            table = hdu_list[1].data
            wave = table["Wave"] * u.um
            flux = table[extract][0] * u.Jy
            error = table[extract.replace('Flux','Err')][0] * u.Jy
            DQ = table["DQ"][0]

            metad =  hdu_list[1].header[str(i)]
            dict_list = metad.split(",")

            meta_dict={}
            meta_dict['band_name'] = table["Band_name"]
            for j in range(len(dict_list)):
                line = dict_list[j]
                key = line.split(":")[0]
                value = line.split(":")[1]
                meta_dict[key] = value
            

            q = astropy.units.Quantity(np.array(flux), unit=u.Jy) 
            unc = StdDevUncertainty(np.array(error))
            spec1d = Spectrum1D(spectral_axis=wave[i].T, flux=q, uncertainty=unc, mask=DQ, meta=meta_dict)
            res_spec1d.append(spec1d)
            
        if len(hdu_list[1].data) == 1: res_spec1d = res_spec1d[0]

        hdu_list.close()

        #self.spec = res_spec1d
        #self.waves = res_spec1d.spectral_axis.value
        #self.fluxes = res_spec1d.flux.value
        #self.flux_uncs = res_spec1d.uncertainty.quantity.value
        #self.masks = DQ
        #self.bandnames = meta_dict['band_name'][0]
        #self.meta = meta_dict

        return res_spec1d

    

    @staticmethod
    def read_inst(instnames, waves, tablePath):
        """
        instnames: instrument/module names, taken from the .ini list (list)
        """

        wMins = np.asarray([]) ; wMaxs = np.asarray([]) ; rSlopes = np.asarray([]) ; rBiases = np.asarray([])
        
        if instnames[0] != 'PHOTOMETRY':
            # Load all files in the resolving power folder
            files = os.listdir(tablePath+'resolving_power/')
            for i in files: #exclude hidden files from mac
                if i.startswith('.'):
                    files.remove(i)
        
            # Get the names of the files for each instrument/module
            inst_files = []
            for inst in list(map(str.upper,instnames)):
                if any(inst in file for file in files):
                    for file in files:
                        if inst in file: inst_files.append(file) 
                else:
                    raise IOError('One or more resolving-power files not in directory. Or check the names.')
                
            # For each instrument/module
            for inst_fn in inst_files:
                try:
                    data = np.genfromtxt(tablePath+'resolving_power/'+inst_fn, comments=';')
                except:
                    try:
                        rfitstable = fits.open(tablePath+'resolving_power/'+inst_fn)
                        data = np.full(5, np.nan)
                        data[1] = rfitstable[1].data[0][0]  # Min wave
                        data[2] = rfitstable[1].data[-1][0] # Max wave
                        data[3] = (rfitstable[1].data[-1][2] - rfitstable[1].data[0][2]) / (rfitstable[1].data[-1][0] - rfitstable[1].data[0][0]) # R_s
                        data[4] = rfitstable[1].data[-1][2] - data[3] * rfitstable[1].data[-1][0] # Bias
                        rfitstable.close()
                    except:
                        raise Exception('Resolving power table format not recognized')
                    
                wMins = np.concatenate((wMins, [data[1]]))
                wMaxs = np.concatenate((wMaxs, [data[2]]))
                rSlopes = np.concatenate((rSlopes, [data[3]]))
                rBiases = np.concatenate((rBiases, [data[4]]))
    
        else:
            wMins = np.concatenate((wMins, [np.nanmin(waves)]))
            wMaxs = np.concatenate((wMaxs, [np.nanmax(waves)]))
            rSlopes = np.concatenate((rSlopes, [0.]))
            if len(instnames) == 2:
                rBiases = np.concatenate((rBiases, [float(instnames[1])]))
            else:
                print('No resolving power provided. Assuming R=50')
                rBiases = np.concatenate((rBiases, [50.]))
        
        instnms = instnames if instnames[0] != 'PHOTOMETRY' else [instnames[0]]
        inst_df = pd.DataFrame({'inst': instnms,
                                'wMin': wMins,
                                'wMax': wMaxs,
                                'rSlope': rSlopes,
                                'rBias': rBiases})
        
        return inst_df



    @staticmethod
    def read_inifile(fname):

        config = configparser.ConfigParser()
        config.read(fname)
        ### ConfigParser uses strings, which is irritating,
        ### so evalueate all as literal
        cdict = {}
        try:
            for section in config.keys():
                sdict = {}
                for key in config[section]:
                    try:
                        line = config[section][key]
                        line = ast.literal_eval(line)
                        if type(line) is tuple or type(line) is list: 
                            line = list(line)
                            for i in range(len(line)):
                                if line[i] == 'np.inf': line[i] = np.inf
                                elif line[i] ==  '-np.inf': line[i] = -np.inf
                        sdict[key.upper()] = line
                    ### Some strings will break literal_eval, so just treat as strings
                    except Exception as E:
                        sdict[key.upper()] = config[section][key]
                cdict[section] = sdict
        except ValueError as E:
            print('Error in '+fname)
            print('Section:', section, 'Parameter:', key)
            sys.exit()
        
        return cdict

    
    @staticmethod
    def write_inifile(params, opts, fname):
        
        config = configparser.ConfigParser()
        ### Unpack lm parameters object into dict
        opts['FITTED PARAM VALUES AND OPTIONS'] = {}
        for par in params:
            val = params[par].value
            vary = params[par].vary
            low = params[par].min
            if not np.isfinite(low): low = '-np.inf'
            high = params[par].max
            if not np.isfinite(high): high = 'np.inf'
            try:
                expr = params[par].expr
                if expr is not None:
                    opts['FITTED PARAM VALUES AND OPTIONS'][par] = [val, vary, low, high, expr]
                else:
                    opts['FITTED PARAM VALUES AND OPTIONS'][par] = [val, vary, low, high]
            except AttributeError:
                opts['FITTED PARAM VALUES AND OPTIONS'][par] = [val, vary, low, high]
        ### Put opts dictionary into config format
        for sect in opts.keys():
            sdict = {}
            for item in opts[sect]:
                sdict[item.upper()] = str(opts[sect][item])
            config[sect] = sdict
        ### Write out
        with open(fname, 'w+') as outpars:
            config.write(outpars)



    @staticmethod
    def pah_table(parcube, compdict=None, x=0, y=0, parobj=False, 
                  pah_complex=True, pah_obs=False, savetbl=None):
        """
        Output the table of PAH integrated powers
    
        Parameters
        ----------
        parcube:
            The parameter cube that stores the fitted parameter values

        compdict: dict
            Component dictionary, which is the dictionary that stores the fitted line/continuum profiles.
            This is needed for EQW measurements. If set None, EQW will not be presented in the PAH table.

        pah_complex: bool (default: True)
            Whether to present the PAH complex fluxes. If set to False, "ALL" the PAH fluxes 
            will be provided.
    
        pah_obs: bool (default: False)
            In default, the presented PAH flux is the intrinsic flux, which has been corrected
            based on the fitted extinction profile. If set to True, the output will be the 
            "observed" flux which is measured directly from the input spectrum, without
            considering extinction correction. This can only be set to True when there is 
            continuum dictionry as input. 

        savetbl:
            Save the table to a .ecsv file
    
        Output
        ------
        PAH table in DataFrame
        """
        if parobj == True:
            from CAFE.cafe_helper import parobj2df
            df = parobj2df(parcube) # parcube is a parobj
        else:
            from CAFE.cafe_helper import parcube2df
            df = parcube2df(parcube, x, y)
    
        if (compdict is None) & (pah_obs is True):
            warnings.warn("pah_obs is set True; however, no compdict is provided. Thus, observed PAH flux cannot be calculated.")

        pah_parname = [i[0]=='d' for i in df.index]
    
        pah_name = list(set([n.split('_')[1] for n in df[pah_parname].index]))
    
        pah_lam_list = []
        pah_gamma_list = []
        pah_peak_list = []
        pah_strength_list = []
        pah_strength_unc_list = []
        pah_strength_obs_list = []
        pah_strength_obs_unc_list = []
        eqw_list = []
        eqw_list_indiv = []
        for n in pah_name:
            p = df.filter(like=n, axis=0)
    
            # -------------
            # Flux estimate
            # -------------
            lam = p.filter(like='Wave', axis=0).value.iloc[0] * u.micron
            pah_lam_list.append(lam.value)
            
            gamma = p.filter(like='Gamma', axis=0).value.iloc[0]
            pah_gamma_list.append(gamma)
    
            peak = p.filter(like='Peak', axis=0).value.iloc[0] * u.Jy
            pah_peak_list.append(peak.value)
    
            # integrated intensity (strength) -- in unit of W/m^2
            pah_strength = (np.pi * const.c.to('micron/s') / 2) * (peak * gamma / lam)# * 1e-26 # * u.watt/u.m**2
            
            # Make the unit to appear as W/m^2
            pah_strength_list.append(pah_strength.to(u.Watt/u.m**2).value)
    
            if compdict is not None:
                
                wave = compdict['CompFluxes']['wave']
        
                if pah_obs is True:
                    # Calculate the "observed" PAH flux, which is the fluxes with no extinction correction.
                    # In this case, the derived PAH flux should always be smaller or equal to the extinction-corrected PAH flux.
                    ext_scale = np.interp(lam.value, wave, compdict['extComps']['extPAH'])
        
                    pah_strength_obs_list.append(pah_strength.to(u.Watt/u.m**2).value * ext_scale)
            
            # -------------------------
            # Flux uncertainty estimate
            # -------------------------
            _lam_unc = p.filter(like='Wave', axis=0).stderr.iloc[0]
            _gamma_unc = p.filter(like='Gamma', axis=0).stderr.iloc[0]
            _peak_unc = p.filter(like='Peak', axis=0).stderr.iloc[0]
    
            # Only proceed if uncertainties exist
            if (_lam_unc is not None) & (_gamma_unc is not None) & (_peak_unc is not None):
                lam_unc = _lam_unc * u.micron
                gamma_unc = _gamma_unc
                peak_unc = _peak_unc * u.Jy
    
                g_over_w_unc = gamma / lam * np.sqrt((gamma_unc/gamma)**2 + (lam_unc/lam)**2) # uncertainty of gamma/lam
                
                pah_strength_unc = (np.pi * const.c.to('micron/s') / 2) * \
                                    (peak * gamma / lam) * np.sqrt((peak_unc/peak)**2 + (g_over_w_unc/(gamma / lam))**2)# * 1e-26# * u.watt/u.m**2
    
                # Make the unit appear as W/m^2
                pah_strength_unc_list.append(pah_strength_unc.to(u.Watt/u.m**2).value)
                
                if (compdict is not None) & (pah_obs is True):
                    pah_strength_obs_unc_list.append(pah_strength_unc.to(u.Watt/u.m**2).value * ext_scale)
            else:
                pah_strength_unc_list.append(np.nan)
                
                if (compdict is not None) & (pah_obs is True):
                    pah_strength_obs_unc_list.append(np.nan)

            if compdict is not None:
                # -----------------------------------
                # EQW measurements for individual PAH
                # -----------------------------------
                fwhm = gamma * lam.value
        
                # Continuum profile
                I_nu_C = compdict['CompFluxes']['fCON'] # The straightout continuum flux level, without extinction correction
        
                # PAH profile
                I_nu_P = drude_prof(wave, [[lam.value], [gamma], [peak.value]]) + I_nu_C
                
                prof_df = pd.DataFrame({'wave': wave,
                                       'I_nu_C': I_nu_C,
                                       'I_nu_P': I_nu_P,
                                       })
        
                # Set the integration range
                int_range = 3*fwhm # set the range as +/- 3*FWHM
                pah_range = prof_df[(prof_df.wave >= lam.value-int_range) & (prof_df.wave < lam.value+int_range)]
        
                EQW = np.trapz((pah_range.I_nu_P - pah_range.I_nu_C) / pah_range.I_nu_C, pah_range.wave)
        
                eqw_list.append(EQW)

                
        if (compdict is None) & (pah_obs is False): # No EQW can be provided
            all_pah_df = pd.DataFrame({'pah_name': pah_name, 
                                       'pah_lam': pah_lam_list, 
                                       'pah_strength': pah_strength_list,
                                       'pah_strength_unc': pah_strength_unc_list,
                                       }
                                     )
        
        elif (compdict is not None) & (pah_obs is False): # Output the extinction corrected PAH fluxes with the EQW
            all_pah_df = pd.DataFrame({'pah_name': pah_name, 
                                       'pah_lam': pah_lam_list, 
                                       'pah_strength': pah_strength_list,
                                       'pah_strength_unc': pah_strength_unc_list,
                                       'pah_eqw': eqw_list,
                                       }
                                     )
        
        if (compdict is not None) & (pah_obs is True): # Output the extinction corrected and observed PAH fluxes with the EQW
            all_pah_df = pd.DataFrame({'pah_name': pah_name, 
                                       'pah_lam': pah_lam_list, 
                                       'pah_strength': pah_strength_list,
                                       'pah_strength_unc': pah_strength_unc_list,
                                       'pah_strength_obs': pah_strength_obs_list,
                                       'pah_strength_obs_unc': pah_strength_obs_unc_list,
                                       'pah_eqw': eqw_list,
                                       }
                                     )

        #all_pah_df['pah_eqw'] = eqw_list
        
        all_pah_df['pah_gamma'] = pah_gamma_list
        all_pah_df['pah_peak'] = pah_peak_list
        all_pah_df = all_pah_df.sort_values('pah_lam')
    
        all_pah_df.set_index('pah_name', inplace=True)
    
        # Change the names that are supposed to be aliphatic features
        for i in range(len(all_pah_df)):
            if list(all_pah_df.index)[i] == 'PAH34':
                all_pah_df.rename(index={'PAH34': 'ali34'}, inplace=True)
            elif list(all_pah_df.index)[i] == 'PAH35':
                all_pah_df.rename(index={'PAH35': 'ali35'}, inplace=True)
    
        # create the DataFrame which includes the info of gamma and will only be used in EQW calculation 
        all_pah_df_tmp = all_pah_df.copy()
    
        # ===========================
        # Calculation for PAH complex
        # ===========================
        # Define main PAH band dictionary
        mainpah_dict = {'PAH33': {'range': [3.25, 3.32]},
                        'ali34': {'range': [3.35, 3.44]},
                        'ali345': {'range': [3.45, 3.5]},
                        'PAH62': {'range': [6.2, 6.3]},
                        'PAH77_C': {'range': [7.3, 7.9]},
                        'PAH83': {'range': [8.3, 8.4]},
                        'PAH86': {'range': [8.6, 8.7]},
                        'PAH113_C': {'range': [11.2, 11.4]},
                        'PAH120': {'range': [11.9, 12.1]},
                        'PAH126_C': {'range': [12.6, 12.7]},
                        'PAH136': {'range': [13.4, 13.6]},
                        'PAH142': {'range': [14.1, 14.2]},
                        'PAH164': {'range': [16.4, 16.5]},
                        'PAH170_C': {'range': [16.4, 17.9]},
                        'PAH174': {'range': [17.35, 17.45]}
                        }
    
        # Generate PAH complex table
        pah_complex_list = []                                                                                                                  
        for pah_lam in all_pah_df.pah_lam:
            match = False
            for mainpah_key in mainpah_dict.keys():
                if (pah_lam >= mainpah_dict[mainpah_key]['range'][0]) & (pah_lam < mainpah_dict[mainpah_key]['range'][1]):
                    pah_complex_list.append(mainpah_key)
                    match = True
                    break
            if match is False:
                pah_complex_list.append(None)
                    
        all_pah_df['pah_complex'] = pah_complex_list
    
        # Flux and flux uncertainty
        pah_complex_strength = all_pah_df.groupby('pah_complex')['pah_strength'].sum()
    
        pah_complex_strength_unc = all_pah_df.groupby('pah_complex')['pah_strength_unc'].apply(lambda x: np.sqrt(np.sum(x**2)))
    
        if pah_obs is False:
            pah_complex_df = pd.concat([pah_complex_strength, pah_complex_strength_unc], 
                                      axis=1)
        else:
            pah_complex_strength_obs = all_pah_df.groupby('pah_complex')['pah_strength_obs'].sum()
    
            pah_complex_strength_obs_unc = all_pah_df.groupby('pah_complex')['pah_strength_obs_unc'].apply(lambda x: np.sqrt(np.sum(x**2)))            
    
            pah_complex_df = pd.concat([pah_complex_strength, pah_complex_strength_unc, 
                                      pah_complex_strength_obs, pah_complex_strength_obs_unc],
                                      axis=1)
        
        if compdict is not None:
            # --------------------------------
            # EQW measurements for PAH complex
            # --------------------------------
            # define the function that will be used later to find PAH bands that are closest to the limits
            def find_closest_limits(elements, limits):
                lower_limit, upper_limit = min(limits), max(limits)
        
                # Function to perform binary search
                def binary_search(arr, target):
                    lo, hi = 0, len(arr) - 1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if arr.iloc[mid] == target:
                            return mid
                        elif arr.iloc[mid] < target:
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    # Ensure the index is within the valid range of the array
                    return min(max(lo, 0), len(arr) - 1)
        
                # Find the closest indices
                lower_index = binary_search(elements, lower_limit)
                upper_index = binary_search(elements, upper_limit)
        
                # Adjust indices to find the closest elements
                if lower_index > 0 and (lower_limit - elements.iloc[lower_index - 1] < elements.iloc[lower_index] - lower_limit):
                    lower_index -= 1
        
                # Check if elements are within the limits
                closest_to_lower = elements.iloc[lower_index] if lower_limit <= elements.iloc[lower_index] <= upper_limit else None
                closest_to_upper = elements.iloc[upper_index - 1] if upper_index > 0 and lower_limit <= elements.iloc[upper_index - 1] <= upper_limit else None
        
                # Return None if no elements are within the limits
                if closest_to_lower is None and closest_to_upper is None:
                    return None
        
                # Return a single number if both elements are identical, or if one of them is None
                if closest_to_lower == closest_to_upper or closest_to_upper is None:
                    return [closest_to_lower]
                elif closest_to_lower is None:
                    return [closest_to_upper]
                return [closest_to_lower, closest_to_upper]
            # -
        
            # Continuum profile
            I_nu_C = compdict['CompFluxes']['fCON'] # The straightout continuum flux level, without extinction correction
        
            prof_df = pd.DataFrame({'wave': wave,
                                    'I_nu_C': I_nu_C,
                                    #'I_nu_P': I_nu_P,
                                    })
            
            pah_complex_eqw_list = []
            for mainpah_key in mainpah_dict.keys():
                pah_element = find_closest_limits(all_pah_df.pah_lam, mainpah_dict[mainpah_key]['range'])
        
                if pah_element is None:
                    EQW = None
        
                elif len(pah_element) == 1:
                    # These params are for calculating the range of the PAH complex
                    gamma = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[0]].pah_gamma.values[0]
                    lam = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[0]].pah_lam.values[0]
                    fwhm = gamma * lam
        
                    # Set the integration range
                    int_range = 3*fwhm # set the range as +/- 3*FWHM
        
                    pah_comp_wmin = lam-int_range
                    pah_comp_wmax = lam+int_range
        
                    for i in range(len(all_pah_df_tmp.pah_lam)):
                        lam = all_pah_df_tmp.pah_lam.iloc[i]
                        if (lam >= mainpah_dict[mainpah_key]['range'][0]) & (lam < mainpah_dict[mainpah_key]['range'][1]):
                            wave = prof_df.wave
                            gamma = all_pah_df_tmp.pah_gamma.iloc[i]
                            peak = all_pah_df_tmp.pah_peak.iloc[i]
                            
                            # PAH profile
                            I_nu_Pi = drude_prof(wave, [[lam], [gamma], [peak]]) + I_nu_C
        
                    prof_df['I_nu_P'] = I_nu_Pi
                    pah_range = prof_df[(prof_df.wave >= pah_comp_wmin) & (prof_df.wave < pah_comp_wmax)]
        
                    EQW = np.trapz((pah_range.I_nu_P - pah_range.I_nu_C) / pah_range.I_nu_C, pah_range.wave)
        
                elif len(pah_element) == 2:
                    # lam of the PAH feature at the shortest(longest) wavelength within the PAH complex range
                    lam1 = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[0]].pah_lam.values[0]
                    lam2 = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[1]].pah_lam.values[0]
                    
                    # gamma of the PAH feature at the shortest(longest) wavelength within the complext
                    gamma1 = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[0]].pah_gamma.values[0] 
                    gamma2 = all_pah_df_tmp[all_pah_df_tmp.pah_lam == pah_element[1]].pah_gamma.values[0]
        
                    # fwhm of the PAH feature at the shortest(longest) wavelength within the PAH complex rnage
                    fwhm1 = gamma1 * lam1
                    fwhm2 = gamma2 * lam2
        
                    # Set the integration range
                    pah_comp_wmin = lam1-3*fwhm1
                    pah_comp_wmax = lam2+3*fwhm2
        
                    # create a zero I_nu_Pi array
                    I_nu_Pi = np.zeros(len(I_nu_C))
                    for i in range(len(all_pah_df_tmp.pah_lam)):
                        lam = all_pah_df_tmp.pah_lam.iloc[i]
                        if (lam >= mainpah_dict[mainpah_key]['range'][0]) & (lam < mainpah_dict[mainpah_key]['range'][1]):
                            wave = prof_df.wave
                            gamma = all_pah_df_tmp.pah_gamma.iloc[i]
                            peak = all_pah_df_tmp.pah_peak.iloc[i]
                            
                            # PAH profile
                            I_nu_Pi += drude_prof(wave, [[lam], [gamma], [peak]])
                    
                    # Add all the PAH profiles with the continuum
                    I_nu_Pi = I_nu_Pi + I_nu_C
        
                    prof_df['I_nu_P'] = I_nu_Pi
                    pah_range = prof_df[(prof_df.wave >= pah_comp_wmin) & (prof_df.wave < pah_comp_wmax)]
        
                    EQW = np.trapz((pah_range.I_nu_P - pah_range.I_nu_C) / pah_range.I_nu_C, pah_range.wave)
                
                pah_complex_eqw_list.append(EQW)
    
            pah_complex_eqw_df = pd.DataFrame({'pah_complex': list(mainpah_dict.keys()),
                                               'pah_complex_eqw': pah_complex_eqw_list,
                                               }).set_index('pah_complex')

            pah_complex_df = pd.merge(pah_complex_df, pah_complex_eqw_df, 
                                      how='inner', left_index=True, right_index=True)
    
        pahs = pah_complex_df if pah_complex is True else all_pah_df
    
        if savetbl is not None:
            pahs_reset = pahs.reset_index()
            t = Table.from_pandas(pahs_reset)
            
            # Add units 
            for col in t.colnames:
                if col in ['pah_strength', 'pah_strength_unc', 'pah_strength_obs', 'pah_strength_obs_unc']:
                    t[col].unit = u.W/u.m**2
                if col in ['pah_complex_eqw']:
                    t[col].unit = u.micron
            
            t.write(savetbl, overwrite=True)
            
            print("PAH table is saved in: "+savetbl)
        
        return pahs

    
    @staticmethod
    def line_table(parcube, x=0, y=0, lineext=None, parobj=False):
        """
        Output the table of line integrated powers
        """
        #fitPars = self.parcube.params
        #df = self.parobj2df(fitPars)

        if parobj == True:
            from CAFE.cafe_helper import parobj2df
            df = parobj2df(parcube) # parcube is a parobj
        else:
            from CAFE.cafe_helper import parcube2df
            df = parcube2df(parcube, x, y)

        line_parname = [i[0]=='g' for i in df.index]
        #line_name = list(set([n.split('_')[1] for n in df[line_parname].index]))
        line_name = list(set(['_'.join(n.split('_')[1:3]) for n in df[line_parname].index]))

        line_wave_list = []
        line_strength_list = []
        line_strength_unc_list = []
        for n in line_name:
            p = df.filter(like=n, axis=0)

            wave = p.filter(like='Wave', axis=0).value.values[0] * u.micron
            wave_unc = p.filter(like='Wave', axis=0).stderr.values[0] * u.micron
            
            gamma = p.filter(like='Gamma', axis=0).value.values[0]
            gamma_unc = p.filter(like='Gamma', axis=0).stderr.values[0]
            
            peak = p.filter(like='Peak', axis=0).value.values[0] * u.Jy
            peak_unc = p.filter(like='Peak', axis=0).stderr.values[0] * u.Jy
            if lineext is not None:
                ext = np.interp(wave.value, lineext['wave'], lineext['ext'])
                peak *= ext ; peak_unc *= ext

            #x = np.linspace(2.5, 38, 200) * u.micron
            #y = peak * gamma**2 / ((x/wave - wave/x)**2 + gamma**2)

            # integrated intensity (strength) -- in unit of W/m^2
            # Gauss = 1 / np.sqrt(np.pi * np.log(2)) * Drude
            # 1 / np.sqrt(np.pi * np.log(2)) ~ 0.678
            line_strength = 1 / np.sqrt(np.pi * np.log(2)) * (np.pi * const.c.to('micron/s') / 2) * (peak * gamma / wave)# * 1e-26 # * u.watt/u.m**2
            
            g_over_w_unc = gamma / wave * np.sqrt((gamma_unc/gamma)**2 + (wave_unc/wave)**2) # uncertainty of gamma/wave
            
            line_strength_unc = 1 / np.sqrt(np.pi * np.log(2)) * (np.pi * const.c.to('micron/s') / 2) * \
                                (peak * gamma / wave) * np.sqrt((peak_unc/peak)**2 + (g_over_w_unc/(gamma / wave))**2)# * 1e-26# * u.watt/u.m**2

            line_wave_list.append(wave.value)
            # Make unit to appear as W/m^2
            line_strength_list.append(line_strength.to(u.Watt/u.m**2).value)
            line_strength_unc_list.append(line_strength_unc.to(u.Watt/u.m**2).value)

        all_line_df = pd.DataFrame({'line_name': line_name, 
                                    'line_wave': line_wave_list, 
                                    'line_strength': line_strength_list,
                                    'line_strength_unc': line_strength_unc_list,}
                                  ).sort_values('line_wave')
        all_line_df.set_index('line_name', inplace=True)

        return all_line_df



    @staticmethod
    def save_line_table(lines, file_name=None, overwrite=False):

        from astropy.table import QTable
        t = QTable([lines.index.values, lines.line_wave, lines.line_strength, lines.line_strength_unc],
                   names = ('line_name', 'line_wave', 'line_strength', 'line_strength_unc'),
                   meta={'wavelength': 'micron',
                         'flux': 'W/m^2',
                   }
        )
        if file_name is None:
            t.write(self.cafe_dir+'output/last_unnamed_linetable.asdf')
        else:
            t.write(file_name+'.ecsv', overwrite=overwrite)



    @staticmethod
    def save_asdf(cafe, pah_tbl=True, line_tbl=True, file_name=None, **kwargs):
        """
        Write the asdf file containing all the info about the model components
        """

        if hasattr(cafe, 'parcube') is False:
            raise ValueError('The spectrum is not fitted yet.')
        else:
            from CAFE.cafe_helper import parcube2parobj
            fitPars = parcube2parobj(cafe.parcube, **kwargs)
            
        from CAFE.cafe_lib import mask_spec, get_model_fluxes, get_feat_pars

        wave, flux, flux_unc, bandname, mask = mask_spec(cafe)
        #spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=cafe.z)
        #
        #prof_gen = CAFE_prof_generator(spec, inparfile, optfile, cafe_path=cafe.cafe_dir)
        #cont_profs = prof_gen.make_cont_profs()
        
        # Get fitted results
        try:
            CompFluxes, CompFluxes_0, extComps, e0, tau0, _ = get_model_fluxes(fitPars, wave, cafe.cont_profs, comps=True)
        except:
            #ipdb.set_trace()
            raise ValueError('Could not get the model fluxes')

        # Narrow gauss and drude components have now an extra "redshift" from the VGRAD parameter
        gauss, drude, gauss_opc = get_feat_pars(fitPars, apply_vgrad2waves=True)
        
        # Get PAH powers (intrinsic/extinguished)
        from CAFE.component_model import drude_int_fluxes
        pah_power_int = drude_int_fluxes(CompFluxes['wave'], drude)
        pah_power_ext = drude_int_fluxes(CompFluxes['wave'], drude, ext=extComps['extPAH'])
        
        # Quick hack for output PAH and line results
        output_gauss = {'wave':gauss[0], 'gamma':gauss[1], 'peak':gauss[2], 'name':gauss[3], 'strength':np.sqrt(2.*np.pi)*2.998e5*gauss[1]*gauss[2]} #  Should add integrated gauss
        output_drude = {'wave':drude[0], 'gamma':drude[1], 'peak':drude[2], 'name':drude[3], 'strength':pah_power_int.value}
        
        # Make dict to save in .asdf
        obsspec = {'wave': wave, 'flux': flux, 'flux_unc': flux_unc}
        cafefit = {'cafefit': {'obsspec': obsspec,
                               'fitPars': fitPars.valuesdict(),
                               'cont_profs': cafe.cont_profs,
                               'CompFluxes': CompFluxes,
                               'CompFluxes_0': CompFluxes_0,
                               'extComps': extComps,
                               'e0': e0,
                               'tau0': tau0,
                               'gauss': output_gauss,
                               'drude': output_drude
        }}
        
        # Save output result to .asdf file
        target = AsdfFile(cafefit)
        if file_name is None:
            target.write_to(self.cafe_dir+'output/last_unnamed_cafefit.asdf')
        else:
            target.write_to(file_name+'.asdf')
