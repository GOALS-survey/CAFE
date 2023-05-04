"""
Created on Mon Aug 30 12:07:56 2021

@author: roub
"""
import ipdb
import configparser
from astropy.coordinates import SkyCoord
import glob
import os
from astropy.io import fits
from cube_preproc import cube_preproc
from cube_handler import cube_handler
import pandas as pd
import numpy as np
from astropy import units as u
from specutils import Spectrum1D
from astropy.nddata import StdDevUncertainty
from specutils import SpectrumList
import pandas as pd 
       

preprocess = cube_preproc()
current_path = os.path.abspath(os.getcwd())


class userAPI:

    def __init__(self):

        print('User API Created')


    # get PSFs/REAL DATA cubes from API   
    def getSubCubes(self, path, files, user_r_arcsec, lambda_ap, point_source, isPSF, centering, background, r_in, width, aperture_type, convolve):

        # Create the list with subCubes elements 
        res = []
        for i in range(len(files)):
            res.append(cube_handler(path, files[i], user_r_arcsec, lambda_ap, point_source, isPSF, centering, background, r_in, width, aperture_type, convolve))
    
        return res

    #%%    
    def sortCubesByLambda(self, cubes, lambdas, files):
        
        lambdas_cp = lambdas.copy()
        cubes_cp = cubes.copy()
        files_cp = files.copy()
        res = []
        res_files = []
        for i in range(len(lambdas)):
            minLambdaIndex = lambdas_cp.index(min(lambdas_cp))
            res.append(cubes_cp[minLambdaIndex])
            res_files.append(files_cp[minLambdaIndex])
            lambdas_cp.remove(min(lambdas_cp))
            del cubes_cp[minLambdaIndex]
            del files_cp[minLambdaIndex]
            
        return [res, res_files]    
                 
      #%%
    def loadUserParams(self, filename):
        
        f = open(filename, "r")
        res = []
        for x in f:
            x = x.replace("\n"," ")
            [key,value] = x.split('=')
            res.append(value)

        return res
    
    @staticmethod
    def read_inipars(fname):
        config = configparser.RawConfigParser()
        #config.read(fname)
        with open(fname) as inifile:
            config.read_string("[FAKE SECTION]\n"+inifile.read())
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
                        sdict[key] = line
                    ### Some strings will break literal_eval, so just treat as strings
                    except Exception as E:
                        sdict[key] = config[section][key]
                cdict[section] = sdict
        except ValueError as E:
            print('Error in '+fname)
            print('Section:', section, 'Parameter:', key)
            sys.exit()

        return cdict


    #%%Write centroids to file
    def writeCubeCentroids(self,cube):
            print('Writing XY Centroids to file')
        
            f = open("centroids/xys_"+cube.name_band+".csv", "w")
            for i in range(len(cube.ls)):
                line = str(cube.ls[i]) +"," +str(cube.xys[i][0])+","+str(cube.xys[i][1])+"\n"
                f.write(line)
            f.close()
            
    # Read Centroids from file        
    def readCubeCentroids(self,file):
             res = []       
             f = open(file, "r")
             for line in f:
                 # print(line)
                 [l,x,y] = line.split(",")
                 res.append([float(x),float(y)])

             f.close()    
             return res
         #%%PSF INF FLUX
    def writePSFInfFlux(self,PSFs):
            print('Writing INF flux to file')
        
            for j in range(len(PSFs)):
                f = open("PSF_infaps/inf_"+PSFs[j].name_band+".csv", "w")
                inf_flux = preprocess.PSFInfFlux(PSFs[j].cube_before, PSFs[j].CDELT1_pix * PSFs[j].CDELT2_pix)
                # print('INF FLUX IS '+str(inf_flux))
                PSFs[j].PSF_inf_flux = inf_flux
                line = str(inf_flux) +'\n'
                # print(line)
                f.write(line)
            f.close()
            
    # Read Centroids from file        
    def readPSFInfFlux(self,file):
             # print(file)
             res = []       
             f = open(file, "r")
             for line in f:

                 line = line.split('[')[1]
                 line = line.split(']')[0]
                 lines = line.split(',')

                 for i in lines:

                     # print(i)
                     res.append(float(i))

             f.close()    
             return res    
         
            
         #%%PSF INF FLUX
    def writeCentroidSky(self,PSFs):
            print('Writing Sky PSF Centroids to file')
        
        
            for j in range(len(PSFs)):
                f = open("PSF_centroids_sky/sky_"+PSFs[j].name_band+".csv", "w")
                res = []
                for i in range(len(PSFs[j].ls)):
                    [jj,kk] = PSFs[j].xys[i]
                    sky = PSFs[j].wcs.pixel_to_world(jj,kk,PSFs[j].ls[i])
                    res.append(sky)
                    line = sky[0].to_string() +'\n'
                    f.write(line)
            f.close()
            
    # Read Centroids from file        
    def readCentroidSky(self,file):
             # print(file)
             res = []       
             f = open(file, "r")
             for line in f:

                 lines = line.split(' ')
                 ra = float(lines[0])
                 dec = float(lines[1])
                 c = SkyCoord(ra, dec, unit="deg") 
                 res.append(c)
             f.close()    
             return res                
#%%
    def writeResultsFile(self, filename, user_params, df, final_ratio, output_path, new_ra, new_dec, ap_l_iss, grid_extraction, grid_NX, grid_NY, step_size, PSFs_path, Data_path):

        df.to_csv(output_path+filename,index=False)
        
        if ap_l_iss != -1:
            warning_message = "######################################## WARNING/ERRORS \n \
            r_ap > annulus r_in from wavelength: "+str(ap_l_iss)
        else:
            warning_message=""
            
        if grid_extraction == 1:
            grid_txt = "######################################## Grid  EXtraction: {NX: "+str(grid_NX)+"(steps), NY: "+str(grid_NY) + "(steps), step size: "+str(step_size)+ "(pix)}\n"
        else:
             grid_txt = ""

        user_radec = SkyCoord(user_params['user_ra'], user_params['user_dec'], unit='deg')
        line = '########################################'\
               +'# Output file of spectrum extraction'\
               +' \n# Data files path: '+Data_path\
               + '\n# PSFs flies path: '+PSFs_path\
               + '\n######################################## User Paramaeters'\
               + '\n# r_ap: '+user_params['user_r_ap']+' [arcsec]'\
               + '\n# Input [RA,dec]: ['+user_radec.ra.to_string(unit=u.hour, sep=('h', 'm', 's'))+', '+str(user_radec.dec)+'] [degrees]'\
               + '\n# Point source: '+user_params['point_source']+'\n# Lambda aperture:'+user_params['lambda_ap']\
               + '\n# Aperture correction:'+user_params['aperture_correction']\
               + '\n# Centering:'+user_params['centering']\
               + '\n# Centering lambda: '+user_params['lambda_cent']+' [um]'\
               + '\n# New [RA,dec] = ['+new_ra.to_string(unit=u.hour, sep=('h', 'm', 's'))+', '+str(new_dec)+']'\
               + '\n# Background Subtraction:'+user_params['background_sub'] \
               + '\n# Background Inner Radious, Annulus Width: '+user_params['r_ann_in']+','+user_params['ann_width']+' [arcsec] \n########################################'\
               + ' Output File description \n# COLUMN_NAME: DESCRIPTION [UNIT]'\
               + '\n# Wave: wavelength [um]'\
               + '\n# Cube_name: Name of original cube'\
               + '\n# Flux_ap: Aperture flux density [Jy]'\
               + '\n# Flux_err_ap: Aperture flux density error [Jy]'\
               + '\n# R_ap: Aperture radius (arcsec)'\
               + '\n# Background: Background flux surface brightness [MJy/sr]'\
               + '\n# Flux_ap_PSC: Flux density after point source correction [Jy]'\
               + '\n# Flux_err_ap_PSC: Flux density error after point source correction [Jy]'\
               + '\n# PSC: Point-source aperture correction factor'\
               + '\n# Flux_ap_stitched: Flux density after band scaling [Jy]'\
               + '\n# Flux_err_ap_stitched: Flux density error after band scaling [Jy]'\
               + '\n# DQ: Data Quality Flag. 0 = OK'\
               + '\n' +  warning_message+grid_txt\
               + '######################################## Results'
        
        with open(output_path+filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\n') + '\n' )
            f.write('Stitching Ratio: '+str(final_ratio).rstrip('\n') + '\n' + content) 
            f.close()
        # print(str(final_ratio))
        

             


    # -*- coding: utf-8 -*-
    """
    Created on Mon Jul  4 11:09:13 2022
    
    @author: roub
    """
    # This function reads a CRETA extraction .fits file and writes the results into a readable .fits cube
    # including the header
    
    def write_grid_fitscube(self, file_name, output_name=None):
        
        if output_name is None: output_name = file_name.split('.fits')[0]+'_cube.fits'
        
        hdu_list = fits.open(file_name)
        all_spec1d = []
        # For every extraction
        for ext in range(len(hdu_list[1].data)):
            metad =  hdu_list[1].header[str(ext)]
            dict_list = metad.split(",")
            dct={}
            for j in range(len(dict_list)):
                line = dict_list[j]
                key = line.split(":")[0]
                value = line.split(":")[1]
                dct[key] = value
            aperture_correction = dct[" 'ap_corr'"] == ' True'
            
            table = hdu_list[1].data
            wave = table["Wave"] * u.um 
            Band_Name = table["Band_Name"]
            dct['Band_name'] = Band_Name
            Flux = table["Flux"] * u.Jy
            Err = table["Err"] * u.Jy
            Flux_st = table["Flux_st"] * u.Jy
            Err_st = table["Err_st"] * u.Jy
            DQ = table["DQ"]
            if aperture_correction:
                Flux_PSC = table['Flux_PSC'] * u.Jy
                Err_PSC = table['Err_PSC'] * u.Jy
            
            fluxes = [Flux[ext], Flux_st[ext], DQ[ext]]
            errors = [Err[ext], Err_st[ext]]
            errors.append(len(DQ[ext]) * [0])
            if aperture_correction:
                fluxes.append(Flux_PSC[ext])
                errors.append(Err_PSC[ext])
                
            q = u.Quantity(np.array(fluxes), unit=u.Jy) 
            unc = StdDevUncertainty(np.array(errors))
            spec1d = Spectrum1D(spectral_axis=wave[ext].T, flux=q, uncertainty=unc, meta=dct)
            all_spec1d.append(spec1d)
            
        spec1dlist = SpectrumList(all_spec1d)
        

        dct_grid = {}
        for i in range(len(spec1dlist)):
            xx = int(spec1dlist[i].meta[" 'step_indx'"])
            #yy = int(spec1dlist[i].meta[" 'step_indy'"])
            dct_grid[xx] = {}
        for i in range(len(spec1dlist)):
            xx = int(spec1dlist[i].meta[" 'step_indx'"])
            yy = int(spec1dlist[i].meta[" 'step_indy'"])
            dct_grid[xx][yy] = spec1dlist[i]    
            
            
        # [wave, y, x]
        fluxes = np.empty([len(spec1dlist[0].flux[0]), len(dct_grid[0]), len(dct_grid)])  
        fluxes_stitched = np.empty([len(spec1dlist[0].flux[1]), len(dct_grid[0]) ,len(dct_grid)]) 
        errors = np.empty([len(spec1dlist[0].flux[0]), len(dct_grid[0]), len(dct_grid)])
        errors_stitched = np.empty([len(spec1dlist[0].flux[1]), len(dct_grid[0]), len(dct_grid)]) 
        DQ = np.empty([len(spec1dlist[0].flux[2]), len(dct_grid[0]), len(dct_grid)])
        
        if aperture_correction:
            fluxes_PSC = np.empty([len(spec1dlist[0].flux[3]), len(dct_grid[0]), len(dct_grid)]) 
            errors_PSC= np.empty([len(spec1dlist[0].flux[3]), len(dct_grid[0]), len(dct_grid)]) 
        
        # i = x ; j = y
        for i in range(len(dct_grid)):
            for j in range(len(dct_grid[i])):
                fluxes[:,j,i] = dct_grid[i][j].flux[0,:]
                errors[:,j,i] = dct_grid[i][j].uncertainty.array[0,:]
                fluxes_stitched[:,j,i] = dct_grid[i][j].flux[1,:]
                errors_stitched[:,j,i] = dct_grid[i][j].uncertainty.array[1,:]
                DQ[:,j,i] = dct_grid[i][j].flux[2,:] 
                if aperture_correction:
                    fluxes_PSC[:,j,i] = dct_grid[i][j].flux[3,:]
                    errors_PSC[:,j,i] = dct_grid[i][j].uncertainty.array[3,:]
                            
        
        #%%write multi-FITS file
        #keys = spec1dlist[0].meta.keys()
        #values = list(spec1dlist[0].meta.values())
        NAXIS1, NAXIS2, NAXIS3 = fluxes.shape
        
        hdu = fits.PrimaryHDU()
        
        fits_flux = fits.ImageHDU(fluxes, name='Flux')
        header = fits_flux.header
        dictionary =spec1dlist[0].meta

        header['PCOUNT'] = 0
        header['GCOUNT'] = 1
        header['EXTNAME'] = 'FLUX'
        header['EXTRTYPE'] = 'EXTENDED'
        header['BUNIT'] = 'Jy/pix'
        header['WCSAXES'] = 3
        header['CRPIX1'] = (int(dictionary[" 'step_indx'"]) + 1) #CRPIX1 starts from 1 
        header['CRPIX2'] = (int(dictionary[" 'step_indy'"]) + 1) #CRPIX2 starts from 1 
        header['CRPIX3'] = 1 #CRPIX2 starts from 1 
        header['CRVAL1'] = float(dictionary["'extraction_RA'"].split(" ")[2]) #Extraction RA
        header['CRVAL2'] = float(dictionary[" 'extraction_DEC'"].split(" ")[2])  #Extraction DEC
        header['CRVAL3'] = 0.
        header['CDELT1'] = float(dictionary[" 'CDELT1'"]) / 3600 #in degrees
        header['CDELT2'] = float(dictionary[" 'CDELT2'"]) / 3600 #in degrees
        header['CDELT3'] = 1.
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC---TAN'
        header['CTYPE3'] = 'WAVE'
        header['CUNIT1'] = 'deg'
        header['CUNIT2'] = 'deg'
        header['CUNIT3'] = 'um '
        header['PC1_1']  = -1
        header['PC1_2']  = 0.
        header['PC1_3']  = 0.
        header['PC2_1']  = 0.
        header['PC2_2']  = 1.
        header['PC2_3']  = 0.
        header['PC3_1']  = 0.
        header['PC3_2']  = 0.
        header['PC3_3']  = 1.
        #values[1] = values[1].replace("'", "")
        header['EXTRTYPE'] =dictionary[" 'exrtaction_type'"]
        header['SPAXSIZE'] = float(dictionary[" 'spax_size'"])
        header['STEPSIZE'] = float(dictionary[" 'step_size'"])

        from astropy.table import Table
        df_names = pd.DataFrame(spec1dlist[0].meta['Band_name'][0])
        df_names.columns = ['Band_name']
        t_names = Table.from_pandas(df_names)
        
        fits_err= fits.ImageHDU(errors, name='Err')
        fits_flux_stitched = fits.ImageHDU(fluxes_stitched, name='Flux_st')
        fits_err_stitched = fits.ImageHDU(errors_stitched, name='Err_st')
        fits_wave = fits.ImageHDU(spec1dlist[0].spectral_axis.value, name='Wave')
        fits_dq = fits.ImageHDU(DQ, name='DQ')
        
        if aperture_correction:
            fits_flux_PSC= fits.ImageHDU(fluxes_PSC, name='Flux_PSC')
            fits_err_PSC = fits.ImageHDU(errors_PSC, name='Err_PSC')
            
        names_array = np.array(list(df_names['Band_name']))
        col1 = fits.Column(name='Band_name', format='20A', array=names_array)
        coldefs = fits.ColDefs([col1])
        fits_bandnames = fits.BinTableHDU.from_columns(coldefs, name="Band_name")
        
        if aperture_correction:
            hdulist = fits.HDUList([hdu, fits_flux, fits_err, fits_flux_PSC, fits_err_PSC,\
                                    fits_flux_stitched, fits_err_stitched, fits_dq, fits_wave, fits_bandnames])
        else:
            hdulist = fits.HDUList([hdu, fits_flux, fits_err,\
                                    fits_flux_stitched, fits_err_stitched, fits_dq, fits_wave, fits_bandnames]) 
            
        hdulist.writeto(output_name, overwrite=True)
        
        hdulist.close()

        
        
    def write_single_fitscube(self, file_name, output_name=None): 
        
        if output_name is None: output_name = file_name.split('.fits')[0]+'_cube.fits'
        
        hdu_list = fits.open(file_name)
        all_spec1d = []
        for i in range(len(hdu_list[1].data)):
            metad =  hdu_list[1].header[str(i)]
            dict_list = metad.split(",")
            dct={}
            for j in range(len(dict_list)):
                line = dict_list[j]
                key = line.split(":")[0]
                value = line.split(":")[1]
                dct[key] = value
                # print(line, "   ")
            aperture_correction = dct[" 'ap_corr'"] == ' True'
            
            table = hdu_list[1].data
            wave = table["Wave"] * u.um 
            Band_Name = table["Band_Name"]
            dct['Band_name'] = Band_Name
            Flux = table["Flux"] * u.Jy
            Err = table["Err"] * u.Jy
            Flux_st = table["Flux_st"] * u.Jy
            Err_st = table["Err_st"] * u.Jy
            DQ = table["DQ"]
            if aperture_correction:
                Flux_PSC = table['Flux_PSC'] * u.Jy
                Err_PSC = table['Err_PSC'] * u.Jy
            
            fluxes = [Flux[i], Flux_st[i], DQ[i]]
            errors = [Err[i], Err_st[i]]
            errors.append(len(DQ[i]) * [0])
            if aperture_correction:
                fluxes.append(Flux_PSC[i])
                errors.append(Err_PSC[i])
            q = u.Quantity(np.array(fluxes), unit=u.Jy) 
            unc = StdDevUncertainty(np.array(errors))
            pec1d = Spectrum1D(spectral_axis=wave[i].T, flux=q, uncertainty=unc, meta=dct)
            all_spec1d.append(pec1d)
            spec1dlist = SpectrumList(all_spec1d)    

                        
            fluxes = spec1dlist[0].flux[0]
            fluxes_stitched =spec1dlist[0].flux[1]
            errors = spec1dlist[0].uncertainty.array[0]
            errors_stitched = spec1dlist[0].uncertainty.array[1]
            DQ = spec1dlist[0].flux[2]
            
            if aperture_correction:
                fluxes_PSC = spec1dlist[0].flux[3]
                errors_PSC = spec1dlist[0].uncertainty.array[3]
            

            #dct_grid = {}   
            #dct_grid = spec1dlist[0].meta
            #
            # for i in range(len(dct_grid)):
            #     for j in range(len(dct_grid[i])):
            #         fluxes[:,j,i] = dct_grid[i][j].flux[0,:]
            #         fluxes_stitched[:,j,i] = dct_grid[i][j].flux[1,:]
            #         DQ[:,j,i] = dct_grid[i][j].flux[2,:]  
            #         errors[:,j,i] = dct_grid[i][j].uncertainty.array[0,:]
            #         errors_stitched[:,j,i] = dct_grid[i][j].uncertainty.array[1,:]
            #         if aperture_correction:
            #             fluxes_PSC[:,j,i] = dct_grid[i][j].flux[2,:]
            #             errors_PSC= dct_grid[i][j].uncertainty.array[2,:]
            #             DQ[:,j,i] = dct_grid[i][j].flux[2,:] 
                        
                    
            
            
            #%%write FITS multicard
            #keys = spec1dlist[0].meta.keys()
            #values = list(spec1dlist[0].meta.values())
            NAXIS1, NAXIS2, NAXIS3 = 1, 1, len(fluxes)
            
            hdu = fits.PrimaryHDU()
            
            fits_flux = fits.ImageHDU(fluxes.value, name='Flux')
            header = fits_flux.header  
            dictionary =spec1dlist[0].meta

            header['PCOUNT'] = 0
            header['GCOUNT'] = 1
            header['EXTNAME'] = 'FLUX'
            header['SRCTYPE'] = 'EXTENDED'
            header['BUNIT'] = 'Jy/pix'
            header['WCSAXES'] = 3 
            header['CRPIX1'] = 1 #CRPIX1 starts from 1 
            header['CRPIX2'] = 1 #CRPIX2 starts from 1 
            header['CRPIX3'] = 1 #CRPIX2 starts from 1 
            header['CRVAL1'] = float(dictionary["'extraction_RA'"].split(" ")[2]) #Extraction RA
            header['CRVAL2'] = float(dictionary[" 'extraction_DEC'"].split(" ")[2])  #Extraction DEC
            header['CRVAL3'] = 0.
            header['CDELT1'] = 0.
            header['CDELT2'] = 0.
            header['CDELT3'] = 1.
            header['CTYPE1'] = 'RA---TAN'
            header['CTYPE2'] = 'DEC---TAN'
            header['CTYPE3'] = 'WAVE'
            header['CUNIT1'] = 'deg'
            header['CUNIT2'] = 'deg'
            header['CUNIT3'] = 'um '
            header['PC1_1']  = -1.
            header['PC1_2']  = 0.
            header['PC1_3']  = 0.
            header['PC2_1']  = 0.
            header['PC2_2']  = 1.
            header['PC2_3']  = 0.
            header['PC3_1']  = 0.
            header['PC3_2']  = 0.
            header['PC3_3']  = 1.
            #values[1] = values[1].replace("'", "")
            header['EXTRTYPE'] =dictionary[" 'exrtaction_type'"]
            header['APRAD'] = float(dictionary[" 'r_ap'"].split("'")[1])
            #add GRCNTRA , dec
            from astropy.table import Table
            df_names = pd.DataFrame(spec1dlist[0].meta['Band_name'][0])
            df_names.columns = ['Band_name']
            t_names = Table.from_pandas(df_names)
            
            fits_err= fits.ImageHDU(errors, name='Err')
            fits_flux_stitched = fits.ImageHDU(fluxes_stitched.value, name='Flux_st')
            fits_err_stitched = fits.ImageHDU(errors_stitched, name='Err_st')
            fits_wave = fits.ImageHDU(spec1dlist[0].spectral_axis.value, name='Wave')
            fits_dq = fits.ImageHDU(DQ.value, name='DQ')
            
            if aperture_correction:
                fits_flux_PSC= fits.ImageHDU(fluxes_PSC.value, name='Flux_PSC')
                fits_err_PSC = fits.ImageHDU(errors_PSC, name='Err_PSC')
                
            names_array = np.array(list(df_names['Band_name']))
            col1 = fits.Column(name='Band_name', format='20A', array=names_array)
            coldefs = fits.ColDefs([col1])
            fits_bandnames = fits.BinTableHDU.from_columns(coldefs, name = "Band_name")
            
            if aperture_correction:
                hdulist = fits.HDUList([hdu, fits_flux,fits_err, fits_flux_PSC, fits_err_PSC,\
                                        fits_flux_stitched, fits_err_stitched, fits_dq, fits_wave, fits_bandnames])
            else:
                hdulist = fits.HDUList([hdu, fits_flux,fits_err,\
                                        fits_flux_stitched, fits_err_stitched, fits_dq, fits_wave, fits_bandnames]) 
                
            hdulist.writeto(output_name, overwrite=True)

            hdulist.close()
