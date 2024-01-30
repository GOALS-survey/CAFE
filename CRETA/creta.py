
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:02:01 2021

@author: roub
"""
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time
import datetime
import os
from specutils import Spectrum1D
import astropy
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.table import Table
import shutil

import CRETA
from CRETA.cube_preproc import cube_preproc
from CRETA.userAPI import userAPI
from CRETA.write_single_fitscube import write_single_fitscube
from CRETA.write_grid_fitscube import write_grid_fitscube

import ipdb

preprocess = cube_preproc()
user = userAPI()
#current_path = os.path.abspath(os.getcwd())+'/'

class creta:

    def __init__(self, creta_dir='../CRETA/'):

        self.creta_dir = creta_dir
        print('CAFE Region Extraction Tool Automaton (CRETA) initialized')
    
#%%
    ##### Function for single point extraction                                #####
    ###############################################################################  
        
    # @aperture_type: Aperture type: 0 for Circular, 1 for Rectangular. (int)
    # @convolve: Fix resolution option. (Boolean)
    # @parameter_file: Use the parameters file or the command execution option. (boolean)
    # @user_ra: Center RA in degrees. (float)
    # @user_dec: Center Dec in degrees. (float)
    # @user_r_ap: user defined radius in arcsec. (float)
    # @point_source: Point or extended source extraction option. (boolean)
    # @lambda_ap: Wavelength that aperture is defined, only for point source (float)
    # @apperture_currection: Apperture correction option (boolean)
    # @centering: Center user input with a 11x11 box (boolean)
    # @lambda_cent: Wavelength of centering (float)
    # @background: Background subtraction option (boolean)
    # @r_ann_in: Inner annulus radius (float)
    # @width: Width of annulus (float)    
    ########### --> Return [df_res,data,,sp1d]   ############################
    # @df_res: A dataframe with extraction information. (pandas.DataFrame)
    # @data: A list of data elements. (list of SubeCube)
    # @meta: Dictionary with metadata information. (dict)
    # @sp1d: The spectrum 1D element. (Spectrum1D)        
    ###############################################################################   
        
    def singleExtraction(self, data_path, parfile_path, output_path=None, parfile_name='single_params.txt',
                         PSFs_path=None, output_filebase_name='last_result',
                         aperture_type=0, convolve=False, user_ra=0., user_dec=0.,
                         user_r_ap=[0.25], point_source=False, lambda_ap=None, aperture_correction=False, centering=False,
                         lambda_cent=None, perband_cent=False, background=False, r_ann_in=None, ann_width=None, parameter_file=True):
        

        import time
        start_time = time.time()

        preprocess = cube_preproc()

        isnotPSF = False
        isPSF = True
        from pathlib import Path
        user = userAPI()
        # print(parfile_name)
        
        if data_path[-1] != '/': data_path+'/'
        if parfile_path[-1] != '/': parfile_path+'/'

        if output_path is None:
            output_path = './extractions/'
            os.makedirs(output_path)
            print('Ouput path /extractions/ created.')
        if output_path[-1] != '/': output_path+'/'

        if PSFs_path is None:
            PSFs_path = self.creta_dir+'PSFs/'
        if PSFs_path[-1] != '/': PSFs_path+'/'

        # Read the parameter file
        if parameter_file:
            #params = user.loadUserParams(parfile_name) #Load User Parameters
            params = user.read_inipars(parfile_path+parfile_name) #Load User Parameters
            params = params['FAKE SECTION']

            sel_cubes = params['cubes'].split("#")[0].split(",")
            sel_cubes = [sel_cube.strip(' ') for sel_cube in sel_cubes]

            # Store remaining parameters
            params['point_source'] = params['point_source'].split("#")[0].replace(" ","")
            params['aperture_correction'] = params['aperture_correction'].split("#")[0].replace(" ","")
            params['centering'] = params['centering'].split("#")[0].replace(" ","")
            params['background_sub'] = params['background_sub'].split("#")[0].replace(" ","")

            point_source = params['point_source'] == 'True'
            lambda_ap = float(params['lambda_ap'].split("#")[0])
            aperture_correction = params['aperture_correction'] == 'True'
            centering = params['centering'] == 'True'
            l_c =  float(params['lambda_cent'].split("#")[0])
            background =  params['background_sub'] == 'True'
            r_in = float(params['r_ann_in'].split("#")[0])
            width = float(params['ann_width'].split("#")[0])
            
            # Read the list of cubes in data
            files = os.listdir(data_path)
            for i in files: #exclude hidden files from mac
                if i.startswith('.'):
                    files.remove(i)
            # Read the list of cubes in PSF
            if aperture_correction or convolve:
                PSF_files = os.listdir(PSFs_path)
                for i in PSF_files: #exclude hidden files from mac
                    if i.startswith('.'):
                        PSF_files.remove(i)
 
            # Check all requested cubes for extraction are in place
            files_sort = []
            PSF_files_sort = []
            for sel_cube in sel_cubes:
                if any(sel_cube in file for file in files):
                    for file in files:
                        if sel_cube in file: files_sort.append(file)
                else:
                    raise Exception('One or more cubes not in data directory. Or make sure you point to the right path with the "data_path" command-line keyword. Currently you are pointing at: '+data_path)
                if aperture_correction or convolve:
                    if any(sel_cube in PSF_file for PSF_file in PSF_files):
                        for PSF_file in PSF_files:
                            if sel_cube in PSF_file: PSF_files_sort.append(PSF_file)
                    else:
                        raise Exception('One or more cubes not in data directory')

            # Store aperture radii
            aper_rs = params['user_r_ap'].split("#")[0].split(",")
            # print(repr(aper_rs))
            user_rs_arcsec = []
            for i in range(len(aper_rs)):
                user_rs_arcsec.append(float(aper_rs[i]))

            params['user_ra'] = params['user_ra'].split("#")[0].replace(" ", "")
            params['user_dec'] = params['user_dec'].split("#")[0].replace(" ", "")

            # Store aperture coordinates
            if 'm' in params['user_ra'] and 'm' in params['user_dec']:
                from astropy.coordinates import SkyCoord
                Stringc = SkyCoord(params['user_ra'], params['user_dec'], frame='icrs')
                user_ra = Stringc.ra.value # float(repr(Stringc.ra).split(" ")[1])
                user_dec = Stringc.dec.value # float(repr(Stringc.dec).split(" ")[1])
                user_ra_sex, user_dec_sex = params['user_ra'], params['user_dec']
            else:    
                user_ra = float(params['user_ra'])
                user_dec = float(params['user_dec'])
                user_radec_sex = SkyCoord(user_ra, user_dec, frame='icrs', unit='deg')
                user_ra_sex = user_radec_sex.ra.to_string(unit=u.hour)
                user_dec_sex = user_radec_sex.dec.to_string()


        # Parameters are given by command line
        else:
            user_rs_arcsec = user_r_ap
            l_c = lambda_cent
            r_in = r_ann_in
            width = ann_width
            params = []
            params.append(str(sel_cubes))
            params.append(str(user_r_ap))
            params.append(str(user_ra))
            params.append(str(user_dec))
            params.append(str(point_source))
            params.append(str(lambda_ap))
            params.append(str(aperture_correction))
            params.append(str(centering))
            params.append(str(lambda_cent))
            params.append(str(background))
            params.append(str(r_ann_in))
            params.append(str(ann_width))
            user_ra_sex, user_dec_sex = str(user_ra), str(user_dec)
            
                         
        #%% 
        ###
        # Step 2: Print user parameters
        ###       
        
        print('PSFs:', PSFs_path)
        print('Data:', data_path)
        #Print user parameters
        print('########################################')
        print('     Load User Parameters ')
        print('########################################')
        print('Cubes: '+str(sel_cubes))
        print('Aperture radii: '+str(user_rs_arcsec)+' (arcsec)')
        print('RA,δ: ['+str(user_ra_sex)+','+str(user_dec_sex)+'] (degrees)')
        print('Point Source: '+str(point_source))
        print('Aperture Correction: '+str(aperture_correction)+' (PSF Correction)')
        print('Centering: '+str(centering))
        print('Centering lambda: '+str(l_c)+'μm')
        print('Background Subtraction: '+str(background))
        if background:
            print('Background Inner Radious, Annulus Width: '+str(r_in)+','+str(width)+' (arcsec,arcsec)')
        print('PSF sub-cubes Path: '+PSFs_path)
        print('Data sub-cubes Path: '+data_path)
        print('########################################')
        #print("!!!!! Loading User's Parameters': %s seconds !!!!!" % (time.time() - time_parameters_loading))    
        
        
        #%% 
        ###
        # Step 3: Create the metadata Dictionary that we will use it for the Spectrum1D output file
        ###
        if point_source:
            aper_type = "point source"
        else:
            aper_type = "extended source"
        from astropy.coordinates import SkyCoord    

        #%% Load Data
        print('Loading Data')    
        ## getSubCubes is in userAPI.py file
        realData_all = user.getSubCubes(data_path, files_sort, user_rs_arcsec, lambda_ap, point_source, isnotPSF, centering, background, r_in, width, aperture_type, False)
        timePSF_loading = time.time()

        #%% Load PSFs: PSF_all is a list with all PSF sub-cubes sorted by wavelength
        if aperture_correction or convolve:
            print('Loading PSFs')
            PSF_all = user.getSubCubes(PSFs_path, PSF_files_sort, user_rs_arcsec, lambda_ap, point_source, isPSF, centering, background, r_in, width, aperture_type, convolve)
            print("PSF Cubes loaded in': %s seconds" % (time.time() - timePSF_loading))
        
        if convolve:
            for i in range(len(realData_all)):
                realData_all[i].fixConvolved(PSF_all[-1].psf_sigma_eff[-1],PSF_all[i].psf_sigma_eff) 
                    

        #%% Centering
        time_centering = time.time()
        if centering:
            new_sky, l_c = preprocess.lambdaBasedCentering(realData_all, user_ra, user_dec, l_c) ######## TDS\
            params['lambda_cent'].split("#")[0] = str(l_c)
            print('Old coordinates were:', user_ra, user_dec)
            print('New coordinates are:', new_sky[0])
            ra_cent = new_sky[0].ra
            dec_cent = new_sky[0].dec
        else:
            from astropy.coordinates import SkyCoord
            c = SkyCoord(ra=user_ra*u.degree, dec=user_dec*u.degree, frame='icrs')    
            ra_cent = c.ra
            dec_cent = c.dec
            

        #%% Generate detector/pixel coordinates for Apertures Photometry PSFs/Data
        if aperture_correction:
            for i in range(len(PSF_all)):

                filename = self.creta_path+"centroids/xys_"+PSF_all[i].name_band+".csv"                
                # PSF Centroids
                if os.path.isfile(filename):
                    if i == 0: print('Loading PSF XY Centers')
                    PSF_all[i].xys = user.readCubeCentroids(filename) #read PSF centroids from file
                else:
                    PSF_all[i].doCenters(PSF_all[i].CRVAL1, PSF_all[i].CRVAL2, isPSF, False) #centering PSF cube  ######### TDS
                    user.writeCubeCentroids(PSF_all[i])  #PSF centroids in file
                    
                # INF Fluxes
                PSF_inf_filename = self.creta_path+"PSF_infaps/inf_"+PSF_all[i].name_band+".csv"
                if os.path.isfile(PSF_inf_filename):
                    if i == 0: print('Loading PSF Total Fluxes')
                    PSF_all[i].PSF_inf_flux = user.readPSFInfFlux(PSF_inf_filename) #read PSF centroids from file
                else:
                    user.writePSFInfFlux(PSF_all)
        
        # Transform sky center into pixel centerS[wave]
        # If perband_cent = False, all pixel centers will be the same for each wave for each sub-band
        # based on the coordinates (obtained after performing -or not- lambdaBasedCentering),
        # otherwise a per-band re-centering process will be applied.
        # This populates the .xys of the sub-cubes
        for i in range(len(realData_all)):
            realData_all[i].doCenters(ra_cent, dec_cent, isnotPSF, perband_cent)
            
        #%% PSF Photometry
        time_PSF_photometry_all = time.time()    
        if aperture_correction:
            for i in range(len(PSF_all)):
                if background:
                    PSF_all[i].doBackgroundSubtraction(point_source, r_in, width)  ## Background Subtraction if needed

                PSF_all[i].doSinglePhotometry(isPSF, background) 
                #PSF_all[i].doFluxUnitConversion()
            print("PSF Photometry executed in: %s seconds" % (time.time() - time_PSF_photometry_all))      
        
        #%% DATA Photometry
        time_data_photometry_all = time.time() 
        for i in range(len(realData_all)):
            if background:
                realData_all[i].doBackgroundSubtraction(point_source, r_in, width)   ## Background Subtraction if needed
            # Photometry
            realData_all[i].doSinglePhotometry(isnotPSF, background)
            # realData_all[i].doAreaCalculations() #calculate the area 
            realData_all[i].doFluxUnitConversion() #change the photometry unit to MJ/sr
        print("Photometry exectued in': %s seconds" % (time.time() - time_data_photometry_all))    
        
        
        #%% PSF CORRECTION
        if aperture_correction:
            for i in range(len(realData_all)):
                realData_all[i].PSFCorrection(PSF_all[i].PSF_correction, PSF_all[i].ls)
        
        
        #%% Create all all_lists 
        time_create_list_all  = time.time() 
        [all_rs_arcsec, all_ls, all_apers, all_xys, all_area_pix, all_bright, all_error_spectrum, all_corrected_spectrum, all_delta,\
         all_names, all_unit_ratio, all_background, all_r_in, all_rs, all_ps, all_psc_flux, all_psc_err] =\
            preprocess.getSubcubesAll(realData_all, background, aperture_correction)

        
        #%%aperture_correction 
        time_create_list_all  = time.time() 
        if aperture_correction:
            PSF_ratio = []
            spectrum_PSF_corrected = []
            error_PSF_corrected = []
            for i in range(len(PSF_all)):
                PSF_ratio.append([])
                spectrum_PSF_corrected.append([])
                error_PSF_corrected.append([]) 
                for j in range(len(PSF_all[i].rs[0])):
                    PSF_ratio[i].append(np.array(PSF_all[i].PSF_correction)[j,:])
                    spectrum_PSF_corrected[i].append(np.array(realData_all[i].spectrum_PSF_corrected)[j,:])
                    error_PSF_corrected[i].append(np.array(realData_all[i].error_PSF_corrected)[j,:])
        
                        
        #%%
        print("Initiating stitching process")            
        data_dict = {}
        for i in range(len(realData_all)):
            data_dict[realData_all[i].name_band] = realData_all[i]
            
        all_s_ratios = []
        # For every aperture, calculate the stitching ratio
        for aperi in range(len(realData_all[0].rs[0])):
            s_ratios = realData_all[0].preprocess.calculateStitchRatios(realData_all, aperture_correction, aperi, False)
            all_s_ratios.append(s_ratios)
        print('Stitch ratios:', all_s_ratios)

        dfs_alls = []
        meta_alls = []
        filenames_alls = []
        # Apply the stitching ratio
        for j in range(len(realData_all[0].rs[0])):          #for every aperture radius

             file_naming = output_filebase_name+'_SingleExt_r'+str(user_rs_arcsec[j])+'as'

             meta_dict = {'extraction_RA':ra_cent, 'extraction_DEC':dec_cent, "r_ap":aper_rs[j], "exrtaction_type":aper_type,
                          "ap_corr":aperture_correction, "Centering":centering, 'Centering_lambda':l_c,
                          "bkg_sub":background, "bkg_r_in":r_ann_in, "bkg_an_w":ann_width
             }
             
             print("For radius", str(aper_rs[j]), "arcsec:")
             for i in range(len(realData_all)):         # for every band name that would exist, except the last
                 ##print('i == ', i , "j === ",j)
                 #if cubesNames[i] in data_dict:        # if the datacube is avaliable
                 data = data_dict[realData_all[i].name_band]
                 
                 #if cubesNames[i+1] in data_dict:  # if we can calculate the stitching ratio                         
                 if aperture_correction: #if PSC, use stitch corrected spectrum
                     beforeStitch = np.array(data.spectrum_PSF_corrected)[j,:]                     
                     beforeStitch_error = np.array(data.error_PSF_corrected)[j,:]
                     
                 else:
                     beforeStitch = np.array(data.corrected_spectrum)[:,j]                     
                     beforeStitch_error = np.array(data.error)[:,j]
                     
                 stitched_flux = preprocess.stitchSpectrum(list(np.array(all_s_ratios)[j,:]), i, beforeStitch) #stitch aperture
                 data.stitched_spectrum.append(stitched_flux)#stitched spectrum
                 stitched_error= preprocess.stitchSpectrum(list(np.array(all_s_ratios)[j,:]), i, beforeStitch_error) #stitch aperture
                 data.stitched_error.append(stitched_error) #stitched spectrum                                    

                 #else: #if next cube does not exists
                 #    
                 #    data.stitched_spectrum.append([np.NaN] * len(data.apers))
                 #    data.stitched_error.append([np.NaN] * len(data.apers))

             ## The last sub-band is append as is, without stitching
             #data = data_dict[cubesNames[len(cubesNames)-1]]     
             #if aperture_correction:
             #    data.stitched_spectrum.append(np.array(data.spectrum_PSF_corrected)[j,:])
             #    data.stitched_error.append(np.array(data.error_PSF_corrected)[j,:])
             #else:
             #    data.stitched_spectrum.append(np.array(data.corrected_spectrum)[:,j])
             #    data.stitched_error.append(np.array(data.error)[:,j])

             all_stitched_spectrum = []
             all_stitched_error = []
             final_apers = []
             final_ls = []
             for i in range(len(realData_all)):

                 final_apers.extend(np.array(realData_all[i].apers)[j,:]) 
                 final_ls.extend(np.array(realData_all[i].ls)) 
                 # print(realData_all[i].name_band , "  exei stitched ", np.array(realData_all[i].stitched_spectrum)[j,0])
                 all_stitched_spectrum.extend(np.array(realData_all[i].stitched_spectrum)[j,:])
                 all_stitched_error.extend(np.array(realData_all[i].stitched_error)[j,:]) #if aperture correction error user corrected error

             
             #Check if r_ap photometry contains NaNs
             for i in range(len(realData_all)):
                 if np.isnan(np.sum(final_apers[i])):
                     print('WARNING: r_ap in', realData_all[i].name_band, 'contains NaNs or/and extends beyond the cube FOV at some wavelength')


             spectrum_PSF_corrected_all = [] 
             error_PSF_corrected_all = [] 
             PSF_ratio_all = []      
            
             #PSF CORRECTION
             if aperture_correction:
                for i in range(len(spectrum_PSF_corrected)):
                      spectrum_PSF_corrected_all.extend(np.array(spectrum_PSF_corrected[i])[j,:])  
                      error_PSF_corrected_all.extend(np.array(error_PSF_corrected[i])[j,:])  
                      PSF_ratio_all.extend(np.array(PSF_ratio[i])[j,:])
                      

            
             #%%          
             time_stitch = time.time()
             res_all = []
             res_all.append(all_ls)
             res_all.append(all_names)
             res_all.append(np.array(all_corrected_spectrum)[:,j])
             res_all.append(np.array(all_error_spectrum)[:,j])
             res_all.append(np.array(all_rs_arcsec)[:,j])
            
             if background:
                 res_all.append(all_background)
               
             if aperture_correction:
                 res_all.append(spectrum_PSF_corrected_all)
                 res_all.append((error_PSF_corrected_all))
                 res_all.append((PSF_ratio_all))
                
             if len(np.array(final_apers).shape)!=1:
                 res_all.append(np.array(all_stitched_spectrum)[j,:])
                 res_all.append(np.array(all_stitched_error)[j,:])
             else:
                 res_all.append(all_stitched_spectrum)
                 res_all.append(np.array(all_stitched_error))
                    
                
             # print("ERROR SHAPE: ",res_all)
             all_DQ_list = []
             for i in range(len(realData_all)):
                 cube = realData_all[i]
                 temp = cube.preprocess.getApertureDQList(cube)
                 
                 all_DQ_list.extend(temp)
             res_all.append(all_DQ_list) 

             print("Stitching performed in: %s seconds" % (time.time() - time_stitch))
        #%%Create DF

             time_writing_output = time.time()
            
             column_names = ['Wave', 'Band_name', 'Flux_ap', 'Err_ap', 'R_ap']
            
             if background:
                column_names.append('Background')
             if aperture_correction:
                column_names.append('Flux_ap_PSC')
                column_names.append('Err_ap_PSC')
                column_names.append('PSC')
                
             column_names.append('Flux_ap_st')    
             column_names.append('Err_ap_st')
             column_names.append('DQ')

             # print(background,aperture_correction,len(res_all))
                
             df = pd.DataFrame(res_all)
            
             df = df.T
             df.columns = column_names
             df = df.sort_values(by=['Wave'])  

             #CHANGE DF dType
             df['Wave']= df['Wave'].astype(float)
             df['Band_name']= df['Band_name'].astype(str)
             df['Flux_ap']= df['Flux_ap'].astype(float)
             df['Err_ap']= df['Err_ap'].astype(float)
             df['R_ap']= df['R_ap'].astype(float)
             if aperture_correction:
                 df['Flux_ap_PSC']= df['Flux_ap_PSC'].astype(float)
                 df['Err_ap_PSC']= df['Err_ap_PSC'].astype(float)
                 df['PSC']= df['PSC'].astype(float)                 
             df['Flux_ap_st']= df['Flux_ap_st'].astype(float)
             df['Err_ap_st']= df['Err_ap_st'].astype(float)
             df['DQ']= df['DQ'].astype(float)

             #%% PLOT SPECTRA
             fig = plt.figure(figsize=(11,8.5))

             plt.loglog(df['Wave'],df['Flux_ap'],label = 'Flux', alpha=1., linewidth=0.25)
             if aperture_correction:
                plt.loglog(df['Wave'],df['Flux_ap_PSC'],label = 'Flux After PSC', alpha=1., linewidth=0.25)
             plt.loglog(df['Wave'],df['Flux_ap_st'],label = 'Flux Stitched', alpha=1., linewidth=0.25)
             plt.xlabel("Wavelength [μm]", fontsize=12)
             plt.ylabel("Flux [Jy]", fontsize=12)

             plt.loglog(df['Wave'],df['Err_ap'], linestyle='dashed', linewidth=0.1, label='Error')
             if aperture_correction:
                   plt.loglog(df['Wave'],df['Err_ap_PSC'], linestyle='dashed', linewidth=0.1, label='Error PSC')
             plt.loglog(df['Wave'],df['Err_ap_st'], linestyle='dashed', linewidth=0.1, label='Error Stitched')             
             #plt.xlabel("Wavelength [μm]", fontsize=12)
             #plt.ylabel("Flux [Jy]", fontsize=12)

             plt.legend(fontsize=12)
             plt.savefig(output_path+file_naming+'_spectra.png', dpi=385)
             #plt.show()
             plt.close()
             
             aperture_lamda_issue = -1
             if background:
                
                if  len(np.where(np.array(all_rs)[:,j] > np.array(all_r_in))[0]) != 0 : 
                    index_with_issue = np.where(np.array(all_rs)[:,j] > np.array(all_r_in))[0][0]
                    aperture_lamda_issue = all_ls[index_with_issue]
                    
                    
             #create output file name based on timestamp       
             now = datetime.datetime.now()
             now = now.strftime("%Y-%m-%d %H:%M:%S")
             now_str = str(now)
             now_str = now_str.replace(':', '-')
             now_str = now_str.replace(' ', '_')    
             
             #file_naming = "JWST_"+str(now_str)+'_'+str(user_rs_arcsec[j])+'as'
             filenames_alls.append(file_naming)
             user.writeResultsFile(file_naming+'.csv', params, df, all_s_ratios, output_path, ra_cent, dec_cent, aperture_lamda_issue, 0, 0, 0, 0, PSFs_path, data_path)
            
             dfs_alls.append(df)
             meta_alls.append(meta_dict)
             print("Output written in: %s seconds" % (time.time() - time_writing_output))

        #return [dfs_alls, realData_all, meta_alls, filenames_alls]
        ##print("Execution Time: %s seconds" % (time.time() - start_time))


        # For each aperture
        for aperi in range(len(dfs_alls)):
            sp1d = self.create1DSpectrum(dfs_alls[aperi], meta_alls[aperi])
            
            ts = time.time()
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S")
            now_str = str(now)
            now_str = now_str.replace(':', '-')
            now_str = now_str.replace(' ', '_')
            
            outparfile_name = output_path+filenames_alls[aperi]+'_params_file'+'.txt'
            shutil.copyfile(parfile_path+parfile_name, outparfile_name)

            bandNameList = (dfs_alls[aperi]['Band_name'].values.tolist())
            for i in range(len(bandNameList)):
                bandNameList[i] = str(bandNameList[i])

            print(len(filenames_alls),'file output(s) named:', filenames_alls[aperi])    
            output_file_name = output_path+filenames_alls[aperi]+".fits"
            t = self.customFITSWriter([dfs_alls[aperi]], output_file_name, [sp1d], aperture_correction, bandNameList, overwrite=True)  
            
            user.write_single_fitscube(output_file_name)

            user_rs_arcsec = [name.split('_')[-1][:-2] for name in filenames_alls]
            for i in range(len(realData_all)):
                realData_all[i].plotApertures(background, user_rs_arcsec, output_path, file_naming)
            
            #self.plotStoreApertures(realData_all, background, user_r_ap)
            print('Total execution time of single region extraction: %s seconds' % str(time.time() - start_time))


        
        
#%%
    ##### Function for Spectrum1D output file creation                          ###
    ###############################################################################    
    # @df_res: A dataframe with extraction information. (pandas.DataFrame)
    # @meta: Dictionary with metadata information. (dict)
    ########### --> Return [df_res,data, meta, sp1d]   ############################
    # @sp1d: The spectrum 1D element. (Spectrum1D)        
    ###############################################################################    
    
    def create1DSpectrum(self, df_res, meta):
            fluxes = []
            errors = []
            df = df_res
            fluxes.append(df['Flux_ap'].values * u.Jy)
            errors.append(df['Err_ap'].values * u.Jy)
            DQ = df['DQ']
            if 'Err_ap_PSC' in df.columns:
                fluxes.append(df['Flux_ap_PSC'].values * u.Jy)
                errors.append(df['Err_ap_PSC'].values * u.Jy)
                
            
            fluxes.append(df['Flux_ap_st'].values * u.Jy)
            errors.append(df['Err_ap_st'].values * u.Jy)
            fluxes.append(DQ)
            errors.append(DQ)
                
            wave = df_res['Wave'].values * u.um    
            wave_all = []
            for i in range(len(fluxes)):
                wave_all.append(wave)
            q = astropy.units.Quantity(np.array(fluxes), unit=u.Jy) 
            # q = astropy.units.Quantity(np.array(fluxes[1]), unit=u.Jy) 
            wave_all = np.array(wave_all).T * u.um
            unc = StdDevUncertainty(np.array(errors))
            sp1d = Spectrum1D(spectral_axis=wave, flux=q ,uncertainty = unc, meta=meta)   
            return sp1d
        


#%%
    ##### Function that writes an extraction fits file.           ###
    ###############################################################################    
    # input: Data Frames and spectrum1d's for the metadata dictionary. (string)
    # output: --> Writes an astropy Table in a .fits file  ############################

    def customFITSWriter(self, df_res, filename, spec1ds, aperture_correction, band_name, overwrite=False):
        
        waves = []
        band_name = []
        Flux = []
        Err = []
        r_aps = []
        Flux_st = []
        Err_st = []
        DQ = []
        names = ["Wave", "Band_name", "Flux", "Err", "R_ap", "Flux_st", "Err_st", "DQ"]
        
        if aperture_correction:
            Flux_PSC = []
            Err_PSC = []
            names.append('Flux_PSC')
            names.append('Err_PSC')
        for i in range(len(df_res)):
            waves.append(df_res[i]["Wave"])
            band_name.append(list(df_res[i]["Band_name"]))
            Flux.append(df_res[i]["Flux_ap"])
            Err.append(df_res[i]["Err_ap"])
            r_aps.append(df_res[i]["R_ap"])
            
            Flux_st.append(df_res[i]["Flux_ap_st"])
            Err_st.append(df_res[i]["Err_ap_st"])
            DQ.append(df_res[i]["DQ"])

            if aperture_correction:
                Flux_PSC.append(df_res[i]["Flux_ap_PSC"])
                Err_PSC.append(df_res[i]["Err_ap_PSC"])
                
        all_data = [waves, band_name, Flux, Err, r_aps, Flux_st, Err_st, DQ]
        if aperture_correction:
            all_data.append(Flux_PSC)
            all_data.append(Err_PSC)

        meta_dict = {}
        for i in range(len(spec1ds)):
            print(str(spec1ds[i].meta))
            meta_str = str(spec1ds[i].meta)
            meta_str = meta_str.replace("{", "")
            meta_str = meta_str.replace("}", "")
            meta_dict[str(i)] = meta_str
            
        tab = Table(all_data, names=names, meta=meta_dict)
        tab.write(filename, format="fits", overwrite=overwrite)  
        return tab
           
           
       
#%%
    ##### Function that reads an extraction fits file.           ###
    ###############################################################################    
    # @filename: CRETA fits filename . (string)
    ########### --> Return res_spec1d  ############################
    # @res_spec1d: A list of extracted spectra. (list of Spectrum1D)        
    ###############################################################################   

    def customFITSReader(self, filename):

        from astropy.io import fits 

        hdu_list = fits.open(filename)
        res_spec1d = []
        for i in range(len(hdu_list[1].data)):
            table = hdu_list[1].data
            wave = table["Wave"] * u.um 
            Flux = table["Flux"] * u.Jy
            Err = table["Err"] * u.Jy
            Flux_st = table["Flux_st"] * u.Jy
            Err_st = table["Err_st"] * u.Jy
            DQ = table["DQ"]

            try:
                Flux_PSC = table["Flux_PSC"]
            except:
                fluxes = [Flux[i], Flux_st[i], DQ[i]]
                errors = [Err[i], Err_st[i]]
            else:
                Err_PSC = table["Err_PSC"]
                fluxes = [Flux[i], Flux_PSC[i], Flux_st[i], DQ[i]]
                errors = [Err[i], Err_PSC[i], Err_st[i]]
            errors.append(len(DQ[i]) * [0])

            metad =  hdu_list[1].header[str(i)]
            dict_list = metad.split(",")

            meta_dict={}
            meta_dict['band_name'] = table["band_name"]
            for j in range( len(dict_list)):
                line = dict_list[j]
                key = line.split(":")[0]
                value = line.split(":")[1]
                meta_dict[key] = value                
            
            q = astropy.units.Quantity(np.array(fluxes), unit=u.Jy) 
            unc = StdDevUncertainty(np.array(errors))
            sp1d = Spectrum1D(spectral_axis=wave[i].T, flux=q ,uncertainty = unc, meta = meta_dict) 
            res_spec1d.append(sp1d)

        hdu_list.close()
        return res_spec1d
    


    #%%
    ##### Function that create grid extraction with default set of parameters.           ###
    ###############################################################################    
    # @path: Path to data files. (string)
    ########### --> Return cube_data  ############################
    # @res_spec1d: A list of data sub-channels. (list of SubCube)        
    ###############################################################################     
    def gridExtraction(self, data_path, parfile_path, output_path=None, parfile_name='grid_params.txt',
                       PSFs_path=None, output_filebase_name='last_result',
                       point_source=False, lambda_ap=None,  centering=False, lambda_cent=None, perband_cent=False,
                       parameter_file=True, plots=False, nx_steps=-1, ny_steps=-1, spax_size=-1, step_size=-1,
                       user_ra=0., user_dec=0., user_center=True, aperture_correction=False, convolve=False):
        
        
        import time
        start_time = time.time()

        if data_path[-1] != '/': data_path+'/'
        if parfile_path[-1] != '/': parfile_path+'/'

        if output_path is None:
            output_path = './extractions/'
            os.makedirs(output_path)
            print('Ouput path /extractions/ created.')
        if output_path[-1] != '/': output_path+'/'

        if PSFs_path is None:
            PSFs_path = self.creta_dir+'PSFs/'
        if PSFs_path[-1] != '/': PSFs_path+'/'


        if parameter_file:
            grid_params = userAPI.read_inipars(parfile_path+parfile_name)
            grid_params = grid_params['FAKE SECTION']
            
            sel_cubes = grid_params['cubes'].split("#")[0].split(",")
            sel_cubes = [sel_cube.strip(' ') for sel_cube in sel_cubes]

            # Read the list of cubes in data
            files = os.listdir(data_path)
            for i in files: #exclude hidden files from mac
                if i.startswith('.'):
                    files.remove(i)
            # Read the list of cubes in PSFs
            if aperture_correction or convolve:
                PSF_files = os.listdir(PSFs_path)
                for i in PSF_files: #exclude hidden files from mac
                    if i.startswith('.'):
                        PSF_files.remove(i)
 
            # Check all requested cubes for extraction are in place
            files_sort = []
            PSF_files_sort = []
            for sel_cube in sel_cubes:
                if any(sel_cube in file for file in files):
                    for file in files:
                        if sel_cube in file: files_sort.append(file)
                else:
                    raise Exception('One or more cubes not in data directory. Or make sure you point to the right path with the "data_path" command-line keyword. Currently you are pointing at: '+data_path)
                if aperture_correction or convolve:
                    if any(sel_cube in PSF_file for PSF_file in PSF_files):
                        for PSF_file in PSF_files:
                            if sel_cube in PSF_file: files_sort.append(PSF_file)
                    else:
                        raise Exception('One or more PSF cubes not in data directory')


            #user_ra = float(grid_params['user_ra'].split("#")[0])
            #user_dec = float(grid_params['user_dec'].split("#")[0])
            #grid_params['point_source'] = grid_params['point_source'].split("#")[0].replace(" ","")
            #point_source = grid_params['point_source'].split("#")[0] == 'True'
            #grid_params['aperture_correction'] = grid_params['aperture_correction'].split("#")[0].replace(" ","")
            #aperture_correction = grid_params['aperture_correction'].split("#")[0] == 'True'
            centering = grid_params['centering'].split("#")[0].replace(" ","") == 'True'
            lambda_cent =  float(grid_params['lambda_cent'].split("#")[0])
            nx_steps = int(grid_params['nx_steps'].split("#")[0])
            ny_steps = int(grid_params['ny_steps'].split("#")[0])
            r_ap = float(grid_params['spax_size'].split("#")[0]) / 2
            step_size = float(grid_params['step_size'].split("#")[0])
            convolve = grid_params['convolve'].split("#")[0].replace(" ","") == 'True'
            
            grid_params['user_ra'] = grid_params['user_ra'].split("#")[0]
            grid_params['user_dec'] = grid_params['user_dec'].split("#")[0]

            if 'm' in grid_params['user_ra'] and 'm' in grid_params['user_dec']:
                from astropy.coordinates import SkyCoord
                Stringc = SkyCoord(grid_params['user_ra'], grid_params['user_dec'], frame='icrs')
                user_ra = Stringc.ra.value # float(repr(Stringc.ra).split(" ")[1])
                user_dec = Stringc.dec.value # float(repr(Stringc.dec).split(" ")[1])
                user_ra_sex, user_dec_sex = grid_params['user_ra'], grid_params['user_dec']
            else:    
                user_ra = float(grid_params['user_ra'])
                user_dec = float(grid_params['user_dec'])
                user_radec_sex = SkyCoord(user_ra, user_dec, frame='icrs', unit='deg')
                user_ra_sex = user_radec_sex.ra.to_string(unit=u.hour)
                user_dec_sex = user_radec_sex.dec.to_string()

                
            user_center = grid_params['user_center'].split("#")[0].replace(" ",'') == 'True'
            
            
        import time 
        ts = time.time()
                
        cubes = []
        for i in range(len(files_sort)):
            cube =preprocess.getFITSData(data_path+files_sort[i])
            cubes.append(cube)

        if lambda_ap == None:
            last_cube = cubes[-1]
            print('Last cube is: ', last_cube['cube_name'])
            last_cube_data = last_cube['cube_data'].copy()
            
            CDELT1 = last_cube['CDELT1']
            CDELT2 = last_cube['CDELT2']
            pixel_scale = np.sqrt(CDELT1*CDELT2)
            DQ = last_cube['DQ']
            CRPIX3 = last_cube['CRPIX3']
            CRVAL3 = last_cube['CRVAL3']
            CDELT3 = last_cube['CDELT3']
            
            nan_mask = DQ != 0
            last_cube_data[nan_mask] = np.NaN 
            wcs = WCS(last_cube['headers'])
            ls = []
            for i in range(last_cube_data.shape[0]):
                l_i = preprocess.getSimulatedL(CRPIX3, CRVAL3, CDELT3 ,i)
                ls.append(l_i)
            l_ap = ls[-1]   #use as l_ap the last lambda of longer wavelength 
        
        else:
            l_ap = lambda_ap

        
        if r_ap <= 0:  
            r_ap = pixel_scale/2

        if step_size <= 0:
            step_size = 2*r_ap
            
        [NZ, NY, NX] = last_cube_data.shape
        x_pix = float((NX-1)/2)
        y_pix = float((NY-1)/2)
        
        #IF the user does not define values for grid points in X or Y coordinate
        if nx_steps < 0 or ny_steps < 0:
            
            if step_size < 0:
                nx_steps = NX
                ny_steps = NY
            else:
                nx_steps = (NX*pixel_scale)/step_size
                ny_steps = (NY*pixel_scale)/step_size
                r_ap = step_size/2
        
        
        print('Cubes:', str(sel_cubes))
        print('RA,δ: ['+str(user_ra_sex)+','+str(user_dec_sex)+'] (degrees)')
        print('Grid Extraction Parameters:')
        print('NX Steps:', nx_steps)
        print('NY Steps:', ny_steps)
        print('Spaxel Size: ', 2*r_ap)
        print('Step Size:', step_size)
        #print('Centered at (x,y): ', x_pix, y_pix)
        # print(NX, "oupla upla ",NY)
        
        file_naming = output_filebase_name+"_GridExt_"+str(nx_steps)+"x"+str(ny_steps)+"_s"+str(2*r_ap)+"as"

        #%% Load Data
        print('Loading Data')  
        realData_all = user.getSubCubes(data_path, files_sort, r_ap, l_ap, point_source, False, False, False, 0, 0 , 1, convolve)
        for i in range(len(realData_all)):
            realData_all[i].rs = [realData_all[i].rs] 
            if convolve:
                realData_all[i].fixConvolved(PSF_all[-1].psf_sigma_eff[-1], PSF_all[i].psf_sigma_eff)
                # print(PSF_all[i].name_band)    
            #print("The real data are ", len(realData_all), " cubes, : ", nx_steps*ny_steps)
            
        #%% Load PSF
        if aperture_correction or convolve:
            print('Loading PSFs')         
            PSF_files = os.listdir(PSFs_path)
            for i in PSF_files: #exclude hidden files from mac
                if i.startswith('.'):
                    PSF_files.remove(i)  
            PSF_all = user.getSubCubes(PSFs_path, PSF_files, r, l_ap, point_source, True, centering, False, 0, 0, 1, convolve)         
            
        
        #%% Centering Process
        if user_center:
            if centering:
                new_sky, lambda_cent =  preprocess.lambdaBasedCentering(realData_all, user_ra, user_dec, lambda_cent) #center by labda, using the 11x11 box
                print('Old coordinates were:', user_ra, user_dec)
                print('New coordinates are:', new_sky[0])
                ra_cent = new_sky[0].ra
                dec_cent = new_sky[0].dec
            else:   
                ra_cent = user_ra
                dec_cent = user_dec
        else:
            print('Using center of longest-wavelength cube to center the grid')
            sky = wcs.pixel_to_world(x_pix, y_pix, ls[-1]*u.um)
            ra_cent = sky[0].ra
            dec_cent = sky[0].dec
        
        
        # Perform grid Photometry on each available sub-band
        all_photometries = []
        all_aps = []
        for i in range(len(realData_all)):
            
            time_photometry = time.time()
            # Perform per-band re-centering if requested, otherwise same center for all cubes
            if perband_cent or i == 0: 
                
                if perband_cent:
                    realData_all[i].doCenters(ra_cent, dec_cent, False, perband_cent)
                    x, y = realData_all[i].xys[0]
                    new_sky = realData_all[i].wcs.pixel_to_world(x, y, realData_all[i].ls[0])
                    ra = new_sky[0].ra
                    dec = new_sky[0].dec
                else:
                    ra = ra_cent
                    dec = dec_cent
                    
                #%%Create Grid points    
                sky_list, pixel_indices, names, sky_ra, sky_dec = preprocess.createGridInArcSec(ra, dec, step_size, nx_steps, ny_steps, realData_all[i], r_ap, False, l_ap)
                
            realData_all[i].grid_cent_ra = ra
            realData_all[i].grid_cent_dec = dec
            # Now plotGrid is executed at the end
            #preprocess.plotGrid(ra, dec, step_size, nx_steps, ny_steps, realData_all[i], r_ap, output_path, file_naming)   

            subband_photometry, aps, DQ_list = realData_all[i].doGridPhotometry(sky_ra, sky_dec, r_ap, realData_all[i].cube_before, plots)
            all_photometries.append(subband_photometry)
            all_aps.append(aps)
            
            print(realData_all[i].name_band+" photometry exectued in: %s seconds" % (time.time() - time_photometry))        
        
        
        #%% PQD
        if aperture_correction:   
            for i in range(len(PSF_all)):
                PSF_all[i].rs = [PSF_all[i].rs]
                
                # Centroid positions in detector coordinates
                filename = self.creta_path+"PSF_centroids/xys_"+PSF_all[i].name_band+".csv"                   
                print(filename)
                if os.path.isfile(filename):
                    if i == 0: print('Loading PSF XY Centers')
                    PSF_all[i].xys = user.readCubeCentroids(filename) #read PSF centroids from file
                else:
                    PSF_all[i].doCenters(PSF_all[i].CRVAL1, PSF_all[i].CRVAL2, True) #centering PSF cube  ######### TDS
                    user.writeCubeCentroids(PSF_all[i])  #PSF centroids in file
                    
                    
                #Load PSF "infitite-aperture" fluxes
                PSF_inf_filename = self.creta_path+"PSF_infaps/inf_"+PSF_all[i].name_band+".csv"
                if os.path.isfile(PSF_inf_filename):
                    if i == 0: print('Loading PSF Total Fluxes')
                    PSF_all[i].PSF_inf_flux = user.readPSFInfFlux(PSF_inf_filename) #read PSF centroids from file
                else:
                    user.writePSFInfFlux(PSF_all)
                    
            time_PSF_photometry_all = time.time()
            
            # Center grid to PSF
            for i in range(len(PSF_all)):
                PSF_sky_cnt_filename =  self.creta_path+"PSF_centroids_sky/sky_"+PSF_all[i].name_band+".csv"
                if os.path.isfile(PSF_sky_cnt_filename):
                    if i == 0: print('Loading PSF Sky Centers')
                    PSF_all[i].xys_sky = user.readCentroidSky(PSF_sky_cnt_filename) #read PSF centroids from file
                else:
                    user.writeCentroidSky(PSF_all)  #PSF centroids in file  
                    
            PSF_correction_ratio = []
            subband_correction_ratio = []
            # Calculate 1 correction value and apply for each spaxel
            
            for i in range(len(PSF_all)):
                # PSF_cnt_sky_list =  preprocess.getPSFSkyCenters(PSF_all[i])
                
                # for j in range(len(PSF_cnt_sky_list)): #all subband slices
                # PSF_cnt_sky = PSF_cnt_sky_list[j]
                
                    # PSF_sky_PSF_list,pixel_indices,PSF_names, PSF_sky_ra, PSF_sky_dec = preprocess.crateGridInArcSec(PSF_cnt_sky[0].ra, PSF_cnt_sky[0].dec, step_size, nx_steps, ny_steps, PSF_all[i], r_ap, PSF_all, point_source, l_ap)
                    # PSF_photometry, PSF_aps, PSF_DQ_list = PSF_all[i].sliceGridExtractionPhotometry(PSF_sky_ra, PSF_sky_dec, r_ap, PSF_all[i].cube_before, plots, j)       
                PCR = []
                band_photos = PSF_all[i].doPSFGridPhotometry()
                # print('Band Photozzz ',np.array(band_photos).shape)
                for j in range(len(band_photos)):
                    PCR.append(PSF_all[i].PSF_inf_flux[j] / band_photos[j])
                    # print(PCR)
                    
                PSF_correction_ratio.append(PCR)
                # preprocess.plotGrid(PSF_cnt_sky[0].ra,PSF_cnt_sky[0].dec, step_size, nx_steps, ny_steps, PSF_all[i], r_ap)
                # PSF_correction_ratio.append(np.array(subband_correction_ratio))
                # subband_correction_ratio = []
                PSC = np.array(PSF_correction_ratio) 
                
            for i in range(len(PSF_all)):
                print(PSF_all[i].name_band, realData_all[i].name_band)
                
                #%% 
        ###
        # Step 3: Create the metadata Dictionary that we will use it for the Spectrum1D output file
        ###
        
        aper_type = "extended_source"

        cmap = plt.cm.get_cmap('hsv', len(sky_list)+1)

        all_dfs = []
        all_spec1ds = []
        all_meta_dicts = []
        # For each grid point, generate the DFs
        for grid_point_idx in range(len(sky_list)):

            # Note that the sky_ra is that of the last sub-band cube, which may differ from other cubes if per-band centering is turned on
            # Also note that the grid center is the original (lambda-based centered) coordinates, before the per-band centering (if turned on)
            meta_dict = {'extraction_RA':sky_ra[grid_point_idx], 'extraction_DEC':sky_dec[grid_point_idx], 'grid_center_RA':ra_cent,
                         'grid_center_DEC':dec_cent, 'exrtaction_type':aper_type, 'ap_corr':aperture_correction,
                         'Centering':centering, 'Centering_lambda':lambda_cent, 'NX':nx_steps, 'NY':ny_steps, 'spax_size':2*r_ap,
                         'step_size':step_size, 'step_indx': pixel_indices[grid_point_idx][0], 'step_indy':pixel_indices[grid_point_idx][1],
                         'CDELT1':step_size, 'CDELT2':step_size, 'CRVAL3':cubes[0]['CRVAL3'], 'CRPIX3':cubes[0]['CRPIX3'], 'CDELT3':cubes[0]['CDELT3'],
                         'bkg_sub':False, 'bkg_r_in':0., 'bkg_an_w':0.
            }
            all_meta_dicts.append(meta_dict)    
            
            
            all_apers = []
            all_error = []
            PSC_flux = []
            PSC_err = []
            PSC_ratio = []
            # all_photometries[cubes][waves][grid_points]
            for i in range(len(all_photometries)): # for each sub-channel
                cube_apers = []
                cube_error = []
                cube_xys = []
                realData_all[i].spectrum_PSF_corrected = []
                realData_all[i].error_PSF_corrected = []
                # sub-channel i
                for j in range(len(all_photometries[i])): # for each wavelength
                    #[sub-channel][wavelength][grid_point]
                    photo = all_photometries[i][j][grid_point_idx]["aperture_sum"]
                    err = all_photometries[i][j][grid_point_idx]["aperture_sum_err"]
                    xx = all_photometries[i][j][grid_point_idx]["xcenter"]
                    yy = all_photometries[i][j][grid_point_idx]["ycenter"]
                    all_apers.append(photo)
                    all_error.append(err)
                    
                    cube_apers.append(photo)
                    cube_error.append(err)
                    cube_xys.append([xx,yy])
                    # cube_area.append()
                    
                realData_all[i].apers = [cube_apers]
                realData_all[i].error = [cube_error]
                realData_all[i].xys = [cube_xys]
                realData_all[i].area = all_aps[i]
                realData_all[i].doFluxUnitConversion()
                
                
                #print( np.array(realData_all[i].corrected_spectrum).shape, (len(all_photometries[i])))
                #print(realData_all[i].name_band)
                if aperture_correction:
                    for j in range(len(PSC[i])):
                        # print(PSF_all[j].name_band, realData_all[j].name_band)
                        realData_all[i].spectrum_PSF_corrected.append(PSC[i][j] * np.array(realData_all[i].corrected_spectrum[0,j]))
                        realData_all[i].error_PSF_corrected.append(PSC[i][j] * np.array(realData_all[i].error[0,j]))
                        
                        
            background = False
            # aperture_correction = False
            time_create_list_all  = time.time() 
            [all_rs_arcsec, all_ls, all_apers, all_xys, all_area_pix, all_bright, all_error_spectrum, all_corrected_spectrum, all_delta,\
             all_names, all_unit_ratio, all_background, all_r_in, all_rs, all_ps, all_psc_flux, all_psc_err]\
             = preprocess.getSubcubesAll(realData_all, background, aperture_correction)
            
            
            # Create a dictionary that contains the data cubes
            data_dict = {}
            for i in range(len(realData_all)):
                data_dict[realData_all[i].name_band] = realData_all[i]
            
            
            # Calculate the stitching ratio between all avaliable sub-bands                
            all_s_ratios = []
            s_ratios = preprocess.calculateStitchRatios(realData_all, aperture_correction, 0, True)
            all_s_ratios = s_ratios
            
            for i in range(len(realData_all)):         # for every band name that  exists
                data = data_dict[realData_all[i].name_band]
                
                if aperture_correction:
                    beforeStitch = np.array(data.spectrum_PSF_corrected)
                    beforeStitch_error = np.array(data.error_PSF_corrected) 
                else:
                    beforeStitch = np.array(data.corrected_spectrum)[0]
                    beforeStitch_error = np.array(data.error)[0]    
                    
                #perform the stitching and assign it back to the cubes                                        
                stitched_flux = preprocess.stitchSpectrum(list(np.array(all_s_ratios)), i, beforeStitch) #stitch aperture
                data.stitched_spectrum = stitched_flux #stitched spectrum
                stitched_error= preprocess.stitchSpectrum(list(np.array(all_s_ratios)), i, beforeStitch_error) #stitch aperture
                data.stitched_error = stitched_error                          
                
            
            all_stitched_spectrum = []
            all_stitched_error = []
            final_apers = []
            final_ls = []
            for i in range(len(realData_all)-1):
                
                final_apers.extend(np.array(realData_all[i].apers)[0,:]) 
                final_ls.extend(np.array(realData_all[i].ls)[:]) 
                all_stitched_spectrum.extend(realData_all[i].stitched_spectrum)
                all_stitched_error.extend((realData_all[i].stitched_error))#if aperture correction error user corrected error
                
                
            #attach the speuctum of last sub-channel @stitched spectrum 47
            final_apers.extend(np.array(realData_all[-1].apers)[0,:])
            final_ls.extend(np.array(realData_all[-1].ls)[:])                         
            if aperture_correction:
                #print('Ta teleutaia nabwww')
                all_stitched_spectrum.extend(np.array(realData_all[-1].spectrum_PSF_corrected))  
                all_stitched_error.extend(np.array(realData_all[-1].error_PSF_corrected))
            else:
                all_stitched_spectrum.extend(np.array(realData_all[-1].corrected_spectrum)[0])  
                all_stitched_error.extend(np.array(realData_all[-1].error)[0])             
                
            final_error= []   
            PSF_ratio_all = []
            
            #plt.ion()
            #plt.plot(all_stitched_spectrum)
            #plt.plot(all_stitched_error)
            #plt.show()
            
            
            #%%  create lists that contains  the extracted results, for all data cubes
            all_psc_flux = []
            all_psc_err = []
            all_corrected_spectrum = []
            all_error_spectrum = []
            for datai in range(len(realData_all)):
                all_psc_flux.extend(realData_all[datai].spectrum_PSF_corrected)
                all_psc_err.extend(realData_all[datai].error_PSF_corrected)
                all_corrected_spectrum.extend(realData_all[datai].corrected_spectrum[0])
                all_error_spectrum.extend(realData_all[datai].error[0])                
                
                
            time_stitch = time.time()
            res_all = []
            res_all.append(all_ls)
            res_all.append(all_names)
            res_all.append((all_corrected_spectrum))
            res_all.append((all_error_spectrum))
            res_all.append((all_rs_arcsec))
            
            res_all.append(all_stitched_spectrum)
            res_all.append(all_stitched_error)
            
            if aperture_correction:
                res_all.append(np.array(all_psc_flux))
                res_all.append(np.array(all_psc_err))
                res_all.append(np.array(PSC_ratio))
                
            all_DQ_list = []
            for i in range(len(realData_all)):
                all_DQ_list.extend(list(np.array(realData_all[i].DQ_lista) // 513)) 
                
            res_all.append(np.array(all_DQ_list)[:,grid_point_idx])
            
            print("Grid point", grid_point_idx, "stitched in: %s seconds" % (time.time() - time_stitch))            
            #%%Create a data frame that contains the extracted information based on the above lists
            
            time_writing_output = time.time()
            
            column_names = ['Wave', 'Band_name', 'Flux_ap', 'Err_ap', 'R_ap']
            column_names.append('Flux_ap_st')    
            column_names.append('Err_ap_st')
            
            if aperture_correction:
                column_names.append('Flux_ap_PSC')
                column_names.append('Err_ap_PSC')
                column_names.append('PSC')
                
            # print(background,aperture_correction,len(res_all))
            column_names.append('DQ')
            
            df = pd.DataFrame(res_all)
            
            df = df.T
            df.columns = column_names
            df = df.sort_values(by=['Wave']) 
            df = df.fillna(value=np.nan)
            
            #CHANGE DF data Type
            df['Wave']= df['Wave'].astype(float)
            df['Band_name']= df['Band_name'].astype(str)
            df['Flux_ap']= df['Flux_ap'].astype(float)
            df['Err_ap']= df['Err_ap'].astype(float)
            df['R_ap']= df['R_ap'].astype(float)
            if aperture_correction:
                df['Flux_ap_PSC']= df['Flux_ap_PSC'].astype(float)
                df['Err_ap_PSC']= df['Err_ap_PSC'].astype(float)
                df['PSC']= df['PSC'].astype(float)                 
            df['Flux_ap_st']= df['Flux_ap_st'].astype(float)
            df['Err_ap_st']= df['Err_ap_st'].astype(float)
            df['DQ']= df['DQ'].astype(float)
            
            
            if grid_point_idx == 0:
                fig = plt.figure(figsize=(8.5,11))
                plt.xlabel("Wavelength [μm]", fontsize=12)
                plt.ylabel("Flux [Jy]", fontsize=12)

            #%% Plot the resulting spectra
            #plt.ion() 
            #plt.loglog(df['Wave'], df['Flux_ap'], label = 'Flux')  
            #if aperture_correction:
            #    plt.loglog(df['Wave'], np.array(all_psc_flux),label = 'Flux After PSC')
            plt.loglog(df['Wave'], df['Flux_ap_st'], label = 'Flux Stitched '+str(pixel_indices[grid_point_idx]), color=cmap(grid_point_idx), alpha=1., linewidth=0.25)
            
            #plt.xlabel("Wavelength (μm)")
            #plt.ylabel("Flux (Jy)")
            #plt.legend()
            #plt.show()
            
            #plt.ion() 
            #plt.loglog(df['Wave'], df['Err_ap'], '--' ,markersize=1,label = 'Flux Error')
            #if aperture_correction:
            #    plt.loglog(df['Wave'],df['Err_ap_PSC'], '--' ,markersize=1,label = 'Flux Error PSC')
            #plt.loglog(df['Wave'],df['Err_ap_st'], '--' ,markersize=1, label = 'Flux Error STC', color=cmap(grid_point_idx))
            
            #plt.show()
            #plt.close()
            
            ##%%
            #aperture_lamda_issue = -1
            #if background:
            #    
            #    if len(np.where(np.array(all_rs)[:,j] > np.array(all_r_in))[0]) != 0 : 
            #        index_with_issue = np.where(np.array(all_rs)[:,j] > np.array(all_r_in))[0][0]
            #        aperture_lamda_issue = all_ls[index_with_issue]
                        
            #create output file name based on timestamp       
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M:%S")
            now_str = str(now)
            now_str = now_str.replace(':', '-')
            now_str = now_str.replace(' ', '_') 

            all_dfs.append(df)
            spec1d = self.create1DSpectrum(df, all_meta_dicts[grid_point_idx])
            all_spec1ds.append(spec1d)            
            
            print("Output written in: %s seconds" % (time.time() - time_writing_output))
            
        plt.legend(fontsize=12)

        plt.savefig(output_path+file_naming+"_spectra.png", dpi=500)
        #plt.show()
        plt.close()


        ts = time.time()
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        now_str = str(now)
        now_str = now_str.replace(':', '-')
        now_str = now_str.replace(' ', '_')
        
        output_file_name = output_path+file_naming+".fits"

        # It writes a .fits file that contains a list of DFs (one per extraction; r_aps or grid points), names of columns
        # and meta data associated to each extraction
        t = self.customFITSWriter(all_dfs, output_file_name, all_spec1ds, aperture_correction, all_names, overwrite=True)  

        # self.customFITSReader(output_file_name) 
        
        if parameter_file:
            #outparfile_name_grid = output_path+"JWST_"+str(now_str)+"_"+str(i)+"_Grid_params_file.txt"
            #shutil.copyfile("grid_params.txt",outparfile_name_grid)
            outparfile_name_grid = output_path+file_naming+"_params_file.txt"
            shutil.copyfile(parfile_path+parfile_name, outparfile_name_grid)
            
            #ff = open(outparfile_name_grid, "a")
            #ff.write("\npoint_source = "+str(point_source)+"\n")
            #ff.write("lambda_ap = "+str(l_ap)+"\n")
            #ff.write("centering = "+str(centering)+"\n")
            #ff.write("lambda_cent = "+str(lambda_cent)+"\n")
            #ff.write("aperture_correction = "+str(aperture_correction)+"\n")
            #ff.write("convolve = "+str(convolve)+"\n")
            #ff.close()
            
        #return all_dfs, res_all, realData_all, all_meta_dicts
             
        for i in range(len(realData_all)):
            preprocess.plotGrid(realData_all[i], realData_all[i].grid_cent_ra, realData_all[i].grid_cent_dec, step_size, nx_steps, ny_steps, r_ap, output_path, file_naming)
        
        user.write_grid_fitscube(output_file_name)
      
        print('Total execution time of grid extraction: %s seconds' % str(time.time() - start_time))


