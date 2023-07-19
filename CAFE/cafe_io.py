import sys
import os
import numpy as np
import configparser
import ast
#import pdb, ipdb
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


class cafe_io:

    def __init__(self):
        
        pass


    @staticmethod
    def init_paths(inopts, cafe_path=None, file_name=None):
        # Path to load external tables
        if not inopts['PATHS']['TABPATH']:
            tablePath = cafe_path+'tables/'
        else: 
            tablePath = inopts['PATHS']['TABPATH']
        
        if file_name is not None:
            if not inopts['PATHS']['OUTPATH']:
                if not os.path.exists(cafe_path+'output/'+file_name):
                    os.mkdir(cafe_path+'output/'+file_name)
                outPath = cafe_path+'output/'+file_name+'/'
            else:
                if not os.path.exists(inopts['PATHS']['OUTPATH']):
                    os.mkdir(inopts['PATHS']['OUTPATH'])
                outPath = inopts['PATHS']['OUTPATH']
        else:
            outPath = './'

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
    def read_inst(instnames, tablePath):
        wMins = np.asarray([]) ; wMaxs = np.asarray([]) ; rSlopes = np.asarray([]) ; rBiases = np.asarray([])
        
        files = os.listdir(tablePath+'resolving_power/')
        for i in files: #exclude hidden files from mac
            if i.startswith('.'):
                files.remove(i)
                
        inst_files = []
        for inst in list(map(str.upper,instnames)):
            if any(inst in file for file in files):
                for file in files:
                    if inst in file: inst_files.append(file)
            else:
                raise IOError('One or more resolving-power files not in directory. Or check the names.')
                
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
                    Exception('Resolving power table format not recognized')
                    
            wMins = np.concatenate((wMins, [data[1]]))
            wMaxs = np.concatenate((wMaxs, [data[2]]))
            rSlopes = np.concatenate((rSlopes, [data[3]]))
            rBiases = np.concatenate((rBiases, [data[4]]))
    
        inst = pd.DataFrame({'inst': instnames,
                             'wMin': wMins,
                             'wMax': wMaxs,
                             'rSlope': rSlopes,
                             'rBias': rBiases
        })
        
        
        return inst



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
    def pah_table(parcube, x=0, y=0, parobj=False, all_pah=False, write2disk=False):
        """
        Output the table of PAH integrated powers

        Parameters
        ----------
            allpah : bool (default: False)
                If allpah is True, return measurements of ALL the 
                individual PAH features. Otherwise, return the complex 
                PAH measurements. 
        """
        #fitPars = self.parcube.params
        #df = self.parobj2df(fitPars)

        if parobj == True:
            from cafe_helper import parobj2df
            df = parobj2df(parcube) # parcube is a parobj
        else:
            from cafe_helper import parcube2df
            df = parcube2df(parcube, x, y)

        pah_parname = [i[0]=='d' for i in df.index]

        pah_name = list(set([n.split('_')[1] for n in df[pah_parname].index]))

        pah_wave_list = []
        pah_strength_list = []
        pah_strength_unc_list = []
        for n in pah_name:
            p = df.filter(like=n, axis=0)

            # --------------
            # Flux estimates
            # --------------
            wav = p.filter(like='Wave', axis=0).value[0] * u.micron
            
            gamma = p.filter(like='Gamma', axis=0).value[0]
            
            peak = p.filter(like='Peak', axis=0).value[0] * u.Jy

            x = np.linspace(2.5, 38, 200) * u.micron
            y = peak * gamma**2 / ((x/wav - wav/x)**2 + gamma**2)

            # integrated intensity (strength) -- in unit of W/m^2
            pah_strength = (np.pi * const.c.to('micron/s') / 2) * (peak * gamma / wav)# * 1e-26 # * u.watt/u.m**2
            
            pah_wave_list.append(wav.value)
            # Make unit to appear as W/m^2
            pah_strength_list.append(pah_strength.to(u.Watt/u.m**2).value)

            # --------------------------
            # Flux uncertainty estimates
            # --------------------------
            _wave_unc = p.filter(like='Wave', axis=0).stderr[0]
            _gamma_unc = p.filter(like='Gamma', axis=0).stderr[0]
            _peak_unc = p.filter(like='Peak', axis=0).stderr[0]

            # Only proceed if uncertainties exist
            if (_wave_unc is not None) & (_gamma_unc is not None) & (_peak_unc is not None):
                wav_unc = _wave_unc * u.micron
                gamma_unc = _gamma_unc
                peak_unc = _peak_unc * u.Jy

                g_over_w_unc = gamma / wav * np.sqrt((gamma_unc/gamma)**2 + (wav_unc/wav)**2) # uncertainty of gamma/wav
                
                pah_strength_unc = (np.pi * const.c.to('micron/s') / 2) * \
                                    (peak * gamma / wav) * np.sqrt((peak_unc/peak)**2 + (g_over_w_unc/(gamma / wav))**2)# * 1e-26# * u.watt/u.m**2

                # Make unit to appear as W/m^2
                pah_strength_unc_list.append(pah_strength_unc.to(u.Watt/u.m**2).value)
            else:
                pah_strength_unc_list.append(np.nan)

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

        #if output_unc is True:
        all_pah_df = pd.DataFrame({'pah_name': pah_name, 
                                   'pah_wave': pah_wave_list, 
                                   'pah_strength': pah_strength_list,
                                   'pah_strength_unc': pah_strength_unc_list,}
                                 ).sort_values('pah_wave')
        all_pah_df.set_index('pah_name', inplace=True)

        # Generate PAH complex table
        pah_complex_list = []                                                                                                                  
        for pah_wave in all_pah_df.pah_wave:
            match = False
            for mainpah_key in mainpah_dict.keys():
                if (pah_wave >= mainpah_dict[mainpah_key]['range'][0]) & (pah_wave < mainpah_dict[mainpah_key]['range'][1]):
                    pah_complex_list.append(mainpah_key)
                    match = True
                    break
            if match is False:
                pah_complex_list.append(None)
                    
        all_pah_df['pah_complex'] = pah_complex_list

        pah_complex_strength = all_pah_df.groupby('pah_complex')['pah_strength'].sum()

        # if output_unc is True:
        pah_complex_strength_unc = all_pah_df.groupby('pah_complex')['pah_strength_unc'].apply(lambda x: np.sqrt(np.sum(x**2)))

        pah_complex_df = pd.merge(pah_complex_strength, pah_complex_strength_unc, left_index=True, right_index=True)

        pahs = pah_complex_df if all_pah is False else all_pah_df

        return pahs



    @staticmethod
    def save_pah_table(pahs, all_pah=False, file_name=None, overwrite=False):

        from astropy.table import QTable
        if all_pah == False:
            t = QTable([pahs.index.values, pahs.pah_strength, pahs.pah_strength_unc],
                       names = ('pah_complex', 'pah_strength', 'pah_strength_unc'),
                       meta={'wavelength': 'micron',
                             'flux': 'W/m^2',
                       }
            )
        else:
            t = QTable([pahs.index.values, pahs.pah_wave, pahs.pah_strength, pahs.pah_strength_unc, pahs.pah_complex],
                       names = ('pah_name', 'pah_wave', 'pah_strength', 'pah_strength_unc', 'pah_complex'),
                       meta={'wavelength': 'micron',
                             'flux': 'W/m^2',
                       }
            )

        if file_name is None:
            t.write(self.cafe_dir+'output/last_unnamed_pahtable.asdf')
        else:
            t.write(file_name+'.ecsv', overwrite=overwrite)



    @staticmethod
    def line_table(parcube, x=0, y=0, parobj=False):
        """
        Output the table of line integrated powers
        """
        #fitPars = self.parcube.params
        #df = self.parobj2df(fitPars)

        if parobj == True:
            from cafe_helper import parobj2df
            df = parobj2df(parcube) # parcube is a parobj
        else:
            from cafe_helper import parcube2df
            df = parcube2df(parcube, x, y)

        line_parname = [i[0]=='g' for i in df.index]

        line_name = list(set([n.split('_')[1] for n in df[line_parname].index]))

        line_wave_list = []
        line_strength_list = []
        line_strength_unc_list = []
        for n in line_name:
            p = df.filter(like=n, axis=0)

            wav = p.filter(like='Wave', axis=0).value[0] * u.micron
            wav_unc = p.filter(like='Wave', axis=0).stderr[0] * u.micron
            
            gamma = p.filter(like='Gamma', axis=0).value[0]
            gamma_unc = p.filter(like='Gamma', axis=0).stderr[0]
            
            peak = p.filter(like='Peak', axis=0).value[0] * u.Jy
            peak_unc = p.filter(like='Peak', axis=0).stderr[0] * u.Jy

            x = np.linspace(2.5, 38, 200) * u.micron
            y = peak * gamma**2 / ((x/wav - wav/x)**2 + gamma**2)

            # integrated intensity (strength) -- in unit of W/m^2
            # Gauss = 1 / np.sqrt(np.pi * np.log(2)) * Drude
            # 1 / np.sqrt(np.pi * np.log(2)) ~ 0.678
            line_strength = 1 / np.sqrt(np.pi * np.log(2)) * (np.pi * const.c.to('micron/s') / 2) * (peak * gamma / wav)# * 1e-26 # * u.watt/u.m**2
            
            g_over_w_unc = gamma / wav * np.sqrt((gamma_unc/gamma)**2 + (wav_unc/wav)**2) # uncertainty of gamma/wav
            
            line_strength_unc = 1 / np.sqrt(np.pi * np.log(2)) * (np.pi * const.c.to('micron/s') / 2) * \
                                (peak * gamma / wav) * np.sqrt((peak_unc/peak)**2 + (g_over_w_unc/(gamma / wav))**2)# * 1e-26# * u.watt/u.m**2

            line_wave_list.append(wav.value)
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
    def save_asdf(cafe, pah_tbl=True, line_tbl=True, file_name=None):
        """
        Write the asdf file containing all the info about the model components
        """

        if hasattr(cafe, 'parcube') is False:
            raise ValueError('The spectrum is not fitted yet.')
        else:
            from cafe_helper import parcube2parobj
            fitPars = parcube2parobj(cafe.parcube)
            
        from cafe_lib import mask_spec, get_model_fluxes, get_feat_pars

        wave, flux, flux_unc, bandname, mask = mask_spec(cafe)
        #spec = Spectrum1D(spectral_axis=wave*u.micron, flux=flux*u.Jy, uncertainty=StdDevUncertainty(flux_unc), redshift=cafe.z)
        #
        #prof_gen = CAFE_prof_generator(spec, inparfile, optfile, cafe_path=cafe.cafe_dir)
        #cont_profs = prof_gen.make_cont_profs()
        
        # Get fitted results
        CompFluxes, CompFluxes_0, extComps, e0, tau0 = get_model_fluxes(fitPars, wave, cafe.cont_profs, comps=True)
        
        # Narrow gauss and drude components have now an extra "redshift" from the VGRAD parameter
        gauss, drude, gauss_opc = get_feat_pars(fitPars)
        
        # Get PAH powers (intrinsic/extinguished)
        from component_model import drude_int_fluxes
        pah_power_int = drude_int_fluxes(CompFluxes['wave'], drude)
        pah_power_ext = drude_int_fluxes(CompFluxes['wave'], drude, ext=extComps['extPAH'])
        
        # Quick hack for output PAH and line results
        output_gauss = {'wave':gauss[0], 'width':gauss[1], 'peak':gauss[2], 'name':gauss[3], 'strength':np.sqrt(2.*np.pi)*2.998e5*gauss[1]*gauss[2]} #  Should add integrated gauss
        output_drude = {'wave':drude[0], 'width':drude[1], 'peak':drude[2], 'name':drude[3], 'strength':pah_power_int.value}
        
        # Make dict to save in .asdf
        obsspec = {'wave': wave, 'flux': flux, 'flux_unc': flux_unc}
        cafefit = {'cafefit': {'obsspec': obsspec,
                               'fitPars': fitPars.valuesdict(),
                               'CompFluxes': CompFluxes,
                               'CompFluxes_0': CompFluxes_0,
                               'extComps': extComps,
                               #'e0': e0,
                               #'tau0': tau0,
                               'gauss': output_gauss,
                               'drude': output_drude
        }}
        
        # Save output result to .asdf file
        target = AsdfFile(cafefit)
        if file_name is None:
            target.write_to(self.cafe_dir+'output/last_unnamed_cafefit.asdf')
        else:
            target.write_to(file_name+'.asdf')
