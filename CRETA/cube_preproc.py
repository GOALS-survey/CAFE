# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:06:50 2021

@author: roub
"""

import sys
import ipdb
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from astropy.wcs import WCS
from matplotlib.pyplot import loglog
from photutils.centroids import centroid_com
from photutils.centroids import centroid_1dg, centroid_2dg
from photutils.aperture import CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import RectangularAperture
import os
import glob

current_path = os.path.abspath(os.getcwd())
from photutils.aperture import CircularAperture

# Ingests one or all sub-cubes. Alwasy returns something.
class cube_preproc:
    def __init__(self):
        pass
        
        
#    def getTargetR(self, base_l, base_r, target_l, pixel_scale, base_pixel_scale):
#
#        return base_r * (target_l/base_l) * (base_pixel_scale/pixel_scale)

    #Load a subcube
    def getFITSData(self, cube_file):
        print('Load file: '+cube_file)
        hdu_list = fits.open(cube_file)
        # hdu_list.info()
        SCI = hdu_list['SCI']
        ERR = hdu_list['ERR'].data
        DQ = hdu_list['DQ'].data
        CRPIX3 = hdu_list['SCI'].header['CRPIX3'] - 1
        CRVAL1 = hdu_list['SCI'].header['CRVAL1']
        CRVAL2 = hdu_list['SCI'].header['CRVAL2']
        CRVAL3 = hdu_list['SCI'].header['CRVAL3']
        CDELT1 = hdu_list['SCI'].header['CDELT1'] * 3600 #arcsec / pixel
        CDELT2 = hdu_list['SCI'].header['CDELT2'] * 3600 #arcsec/ pixel
        CDELT3 = hdu_list['SCI'].header['CDELT3'] #um / 'pixel'
        pixelScale = np.sqrt(hdu_list['SCI'].header['CDELT1']*hdu_list['SCI'].header['CDELT2']) * 3600 # arcsec / pixel
        if hdu_list['PRIMARY'].header['INSTRUME'] == 'NIRSPEC':
            cube_name = str(hdu_list['PRIMARY'].header['GRATING'])
        else:
            cube_name = 'ch' + str(hdu_list['PRIMARY'].header['CHANNEL']) + '_' + str(hdu_list['PRIMARY'].header['BAND'])
        output_file_name = str(hdu_list['PRIMARY'].header['OBS_ID'])
        hdu_list.close()
        
        cube_data = fits.getdata(cube_file)
        headers = SCI.header

        if 'MJD-BEG' in headers: del headers[10:16]
        if 'REFFRAME' in headers: del headers[12:30]

        headers_txt = repr(headers)
        headers_list= headers_txt.split("\n")
        
        primaryDict = {}
        for i in range(len(headers_list)):
            line = headers_list[i]
            if line.find('=') != -1:
                split1 = line.split("=")
                kkey = split1[0]
                kkey = "".join(kkey.split())
                
                split2 = split1[1].split('/')        
                vvalue = split2[0]
                vvalue = "".join(vvalue.split())
                primaryDict[kkey]=vvalue 
                
        res = {}
        res['instrument'] = hdu_list['PRIMARY'].header['INSTRUME']
        res['cube_data'] = cube_data       # 0
        res['primaryDict'] = primaryDict   # 1
        res['headers'] = headers           # 2
        res['CRPIX3'] = CRPIX3             # 3
        res['CRVAL3'] = CRVAL3             # 4
        res['CDELT3'] = CDELT3             # 5
        res['pixelScale'] = pixelScale     # 6
        res['err_data'] = ERR              # 7
        res['CDELT1'] = CDELT1             # 8
        res['CDELT2'] = CDELT2             # 9
        res['cube_name'] = cube_name       # 10
        res['output_file_name'] = output_file_name  # 11
        res['DQ'] = DQ                     # 12
        res['CRVAL1'] = CRVAL1             # 13
        res['CRVAL2'] = CRVAL2             # 14

        return res

         
    def getPSFData(self, cube_file):
        print('Load file: '+cube_file)
        hdu_list = fits.open(cube_file)
        # hdu_list.info()
        try:
            SCI = hdu_list['SCI']
        except:
            hdu_list[0].header['EXTNAME'] = 'SCI'
            SCI = hdu_list['SCI']
            ERR = hdu_list['SCI'].data
            DQ = hdu_list['SCI'].data * 0.
        else:
            ERR = hdu_list['ERR'].data
            DQ = hdu_list['DQ'].data

        CRPIX3 = hdu_list['SCI'].header['CRPIX3'] - 1
        CRVAL1 = hdu_list['SCI'].header['CRVAL1']
        CRVAL2 = hdu_list['SCI'].header['CRVAL2']
        CRVAL3 = hdu_list['SCI'].header['CRVAL3']
        CDELT1 = hdu_list['SCI'].header['CDELT1'] * 3600 #arcsec / pixel
        CDELT2 = hdu_list['SCI'].header['CDELT2'] * 3600 #arcsec/ pixel
        CDELT3 = hdu_list['SCI'].header['CDELT3'] #um / 'pixel'
        pixelScale = np.sqrt(hdu_list['SCI'].header['CDELT1']*hdu_list['SCI'].header['CDELT2']) * 3600 # arcsec / pixel

        try:
            inst = hdu_list['PRIMARY'].header['INSTRUME']
        except:
            instkey = 'SCI'
        else:
            instkey = 'PRIMARY'
            
        if hdu_list[instkey].header['INSTRUME'] == 'NIRSPEC':
            cube_name = str(hdu_list[instkey].header['GRATING'])
        else:
            cube_name = 'ch' + str(hdu_list[instkey].header['CHANNEL'])+ '_' + str(hdu_list[instkey].header['BAND'])
        output_file_name = str(hdu_list[instkey].header['OBS_ID'])
            
        hdu_list.close()
        
        cube_data = fits.getdata(cube_file)
        headers = SCI.header

        if 'MJD-BEG' in headers: del headers[10:16]
        if 'REFFRAME' in headers: del headers[12:30]

        headers_txt = repr(headers)
        headers_list= headers_txt.split("\n")
        
        primaryDict = {}
        for i in range(len(headers_list)):
            line = headers_list[i]
            if line.find('=') != -1:
                split1 = line.split("=")
                kkey = split1[0]
                kkey = "".join(kkey.split())
                
                split2 = split1[1].split('/')        
                vvalue = split2[0]
                vvalue = "".join(vvalue.split())
                primaryDict[kkey]=vvalue 
                
        res = {}
        res['instrument'] = hdu_list[instkey].header['INSTRUME']
        res['cube_data'] = cube_data       # 0
        res['primaryDict'] = primaryDict   # 1
        res['headers'] = headers           # 2
        res['CRPIX3'] = CRPIX3             # 3
        res['CRVAL3'] = CRVAL3             # 4
        res['CDELT3'] = CDELT3             # 5
        res['pixelScale'] = pixelScale     # 6
        res['err_data'] = ERR              # 7
        res['CDELT1'] = CDELT1             # 8
        res['CDELT2'] = CDELT2             # 9
        res['cube_name'] = cube_name       # 10
        res['output_file_name'] = output_file_name  # 11
        res['DQ'] = DQ                     # 12
        res['CRVAL1'] = CRVAL1             # 13
        res['CRVAL2'] = CRVAL2             # 14

        return res

         
    def getChannelLs(self,subcube):
        all_li = []

        for i in range(subcube.cube_before.shape[0]):
            l_i =  self.getSimulatedL(subcube.CRPIX3, subcube.CRVAL3, subcube.CDELT3, i)
            all_li.append(l_i)

        return all_li
    
    # calculates the wavelength 
    #CDELT3 pixel scale of z axe
    def getSimulatedL(self,CRPIX3, CRVAL3, CDELT3 ,i):

        return CRVAL3 + (i-CRPIX3) * CDELT3


         #%%   
    def getPointSourceRs(self, subcube, base_r):
        all_ri = []
        for i in range(subcube.cube_before.shape[0]):
            
            r_i = self.getRpixPointSource(base_r, subcube.pixel_scale, subcube.ls[i], subcube.base_l)
            all_ri.append(r_i)

        return all_ri
    
   #%% Get the radius in pixels 
   # @r_ap: the user radius in arc sec
   # @ps: the channel's pixel scale
   # @l_i: the target lambda
   ##
  
    def getRpixPointSource(self, r_ap, ps, l_i, l_ap):
        return (np.array(r_ap)/ps) * (l_i/l_ap)
    
             #%%   
    def getExtendedSourceRs(self, subcube, base_r):
        all_ri = []

        for i in range(subcube.cube_before.shape[0]):
            r_i = self.getRpixExtendedSource(base_r, subcube.pixel_scale)
            all_ri.append(r_i)

        return all_ri

   #%% Get the radius in pixels 
   # @r_ap: the user radius in arc sec
   # @ps: the channel's pixel scale
   # @l_i: the target lambda
   ##
    
    def getRpixExtendedSource(self, r_ap, ps):
        return (np.array(r_ap)/ps)  
    

#%%
    ##### Function for PSF sub-channel centering. Use a sub-image in order to avoid bad pixels#####
    ###############################################################################  
    # @subcube: The corresponding PSF sub-channel. (SubeCube)
    # @image: The data that we will use, with or without background subtraction based on user option. (np.array)
    ########### --> Return res   ############################
    # @res: A list with resulting centroids in pixels. (list)   
    ###############################################################################
    def getPSFPixelCenters(self, subcube, image):

        c1 = SkyCoord(subcube.user_ra, subcube.user_dec, unit="deg")  #### TDS
        xx, yy, zz = subcube.wcs.world_to_pixel(c1, subcube.ls[0]*u.um)   #### TDS
        res = []
        for i in range(image.shape[0]):
            sliceIm = image[i,:,:]
            #x, y = self.userCentroid(sliceIm, xx, yy)
            x, y = self.imageCentroid(sliceIm)
            res.append([x, y]) # in image coordinates
        return res


    def getPSFSkyCenters(self, subcube):
        res = []
        for i in range(len(subcube.ls)):
            [x, y] = subcube.xys[i] # in image coordinates
            
            sky = subcube.wcs.pixel_to_world(x, y, subcube.ls[i])
            res.append(sky)

        return res 

#%% 
    ##### Function that creates a list centroids   #####
    ###############################################################################  
    # @subcube: Sub-channel for pixel center calculation. (SubeCube)       
    # @image: The data that we will use, with or without background subtraction based on user option. (np.array)
    ########### --> Returns [x,y]   ############################
    # @y: Center Y coordinate. (int)   
    # @x: Center X coordinate. (int)  
    ###############################################################################
    def getPixelCenters(self, subcube, image):
        
        c1 = SkyCoord(subcube.user_ra, subcube.user_dec, unit="deg")  #### TDS
        x,y,z = subcube.wcs.world_to_pixel(c1, subcube.ls[0]*u.um)   #### TDS
        res = []
        for i in range(image.shape[0]):
            res.append([x,y])
            
        return res    


    def getPixelCentersPerCube(self, subcube, image, toSky=False):
        
        c1 = SkyCoord(subcube.user_ra, subcube.user_dec, unit="deg")  #### TDS
        xx, yy, zz = subcube.wcs.world_to_pixel(c1, subcube.ls[0]*u.um)   #### TDS
        sliceIm = np.nansum(image[0:10,:,:], axis=0)
        x, y = self.userCentroid(sliceIm, xx, yy) #, hbox_x=9, hbox_y=9)
        #x, y = self.imageCentroid(sliceIm)
        res = []
        for i in range(image.shape[0]):
            res.append([x,y])
            
        return res    


        # %% Aperture Photometry
    def AperturePhotometry(self, subcube, image):
        aperture_sum = []
        all_apers = []
        aper_elem = []
        all_area = []
        all_error = []
        for i in range(image.shape[0]):
            # print('BAND: ',subcube.name_band , ' PLANE : ', i)
            sliceIm = image[i,:,:]
            jj, kk = subcube.xys[i]

            [photometries, errors, areas] = self.getApertureSum(jj, kk, i, subcube.rs[i], sliceIm, subcube.wcs,\
                                                               subcube.error_data[i,:,:], subcube.aperture_type, subcube.DQ)
            
            aperture_sum.append(photometries)  
            all_area.append(areas)
            all_error.append(errors) 

        return [aperture_sum, all_area,all_error]
    

    #
    def getApertureSum(self, x, y, z, r, data, w, error, aprture_type, DQ):
       
        apers = []
        dq_slice = []
        for i in range(len(r)):
            #get the rectangular or circular aperture
            if aprture_type == 0:
                aper = CircularAperture([x,y], r[i])
            else:    
                aper = RectangularAperture([x,y], r[i]*2., r[i]*2.)
            apers.append(aper)

        phot_table = aperture_photometry(data, apers, wcs=w, error=error)
        rs_photometry = []
        rs_error = []
        rs_area = []
        for i in range(len(r)):
            rs_photometry.append(phot_table["aperture_sum_"+str(i)][0])
            rs_error.append(phot_table["aperture_sum_err_"+str(i)][0])
            rs_area.append(apers[i])

        return [rs_photometry, rs_error, rs_area]
   
  
    
    #%% RIGHT NOW THIS FUNCTION IS NOT USED BECAUSE WE ALWAYS ASUME THERE ARE SOME COORDINATES AVAILABLE TO EXTRACT THE 11x11

    ##### Calculate the image Centroid using an 11x11 box at the 'midle' of the image  #####
    ###############################################################################  
    # @image: The data that we will use, with or without background subtraction based on user option. (np.array)
    # @xx: Cordinate of X-axe, used as sub-image center. (float)
    # @yy: Cordinate of Y-axe, used as sub-image center. (float)    
    ########### --> Returns [column, rows]   ############################
    # @column: Centroid Y coordinate. (int)   
    # @row: Centroid X coordinate. (int)  
    ###############################################################################
    def imageCentroid(self, image):
        NY = int(image.shape[0]/2)
        NX = int (image.shape[1]/2)
        
        #  # 25% to 75% sub Image
        img = image.copy()
        start_X = int(NX/2)
        start_Y = int (NY/2)
        subImg = img[start_Y:start_Y+NY, start_X:start_X+NX]
        
        xys = np.where(subImg == np.nanmax(subImg))
        yy = xys[0][0]
        xx = xys[1][0]
        
        zoom2img = subImg[yy-5:yy+6, xx-5:xx+6]
        
        #if there are NaNs within the sub-image do not center
        if np.isnan(np.sum(zoom2img)) or np.isinf(np.sum(zoom2img)) or np.sum(zoom2img) <= 0.: 
            
            return [xx, yy]                 
        
        else:
            #if np.ma.count(zoom2img) < 7: ipdb.set_trace()
            cx, cy = centroid_2dg(zoom2img) # task from phot utils will return x,y
            xx = start_X + xx - 5 + cx 
            yy = start_Y + yy - 5 + cy
            
            return [xx, yy]


   #%% Calculate the image Centroid using an 11x11 box at the coordinates provided 
   # @image: the 2D input image 
   # @xx:    coordinate of 11x11 of max flux
   # @yy:    coordinate of 11x11 of max flux
   ###
    def userCentroid(self, image, x, y, hbox_x=5, hbox_y=5):

        iy = int(y)
        ix = int(x)
        
        zoom2img = image[iy-hbox_y:iy+hbox_y+1, ix-hbox_x:ix+hbox_x+1]
        
        if np.isnan(np.sum(zoom2img)) or np.isinf(np.sum(zoom2img)) or np.sum(zoom2img) <= 0.: 
            
            print('The cetroid algorithm failed because there are some nans or infs or negative values in the images around the centering region')
            return [x, y]        
        
        else:
            cx, cy= centroid_2dg(zoom2img) 
            xx = ix - hbox_x + cx
            yy = iy - hbox_y + cy
            
            return [xx, yy]

#%%
    ##### Function for centering at a specific wavelength that user defines, using 3 slices from data array.     #####
    ###############################################################################  
    # @images: List of all avaliable sub-channels. (list of SubeCube)
    # @l_c: Wavelength used for centering. (float)
    # @RA: The user defined RA before centering. (arcsec)
    # @dec: The user defined dec before centering. (arcsec)    
    ########### --> Returns sky   ############################
    # @sky: The sky coordinates after centering at wavelenth l_c. (SkyCoord)   
    ###############################################################################      
    def lambdaBasedCentering(self, cubes, user_ra, user_dec, l_c, dxdy=False):
        res_cubes = []
        ls_min = []
        ls_max = []
        found = False
        for i in range(len(cubes)):
            ls_min.append(cubes[i].ls[0])
            ls_max.append(cubes[i].ls[-1])
            if l_c >= ls_min[i] and l_c<= ls_max[i]:
                res_cubes.append(i)
                found = True
                
        if len(res_cubes) == 0:
            raise ValueError('The wavelength for centering is not within the wavelength range of the cubes provided.')

        the_cube = cubes[res_cubes[0]] 
        
        c1 = SkyCoord(user_ra, user_dec, unit="deg")  # defaults to   
        x, y, z = the_cube.wcs.world_to_pixel(c1, l_c*u.um)
        
        if dxdy:
            the_cube = theCube
            z,y,x = the_cube.cube_before.shape
            x = x/2
            y = y/2
            sky_list = []
            res_cubes_all = []
            for i in range(len(the_cube.cube_before)):
                plane = the_cube.cube_before[i,:,:]
                jj, kk= self.userCentroid(plane,x,y)
                sky = the_cube.wcs.pixel_to_world(jj,kk,l_c)
                sky_list.append(sky)
                res_cubes_all.append( res_cubes[0] )
                
            return sky_list,res_cubes_all            
        else:    
            x = x.tolist()
            y = y.tolist()
            z = round(z.tolist())
            
        plane = np.nansum(the_cube.cube_before[np.max(((z-2),0)):np.min(((z+3),len(the_cube.cube_before))),:,:], axis=0)  #add one before and one plane after l_c
        
        xx, yy = self.userCentroid(plane, x, y)
        
        print("Centering around", the_cube.ls[z], "um in cube", the_cube.name_band)
        sky = the_cube.wcs.pixel_to_world(xx, yy, the_cube.ls[z]*u.um)
        
        return sky, the_cube.ls[z]


    #%%   
    def totalImageFlux(self, image):

        res = []
        flux = 0
        for i in range(len(image)):
            flux = np.nansum(image[i])
            res.append(flux)
            
        return res      


#%%% Calculate the PSF infinite aperture as the total of the elements 
# @image:    the 3-D image cube of PSF
###
    def PSFInfFlux(self, image, delta_factor):
        
        res = []
        flux = 0
        for i in range(len(image)): 
            flux = np.nansum(image[i])
            res.append(flux)
        # res = list(np.array(res) * 10**6 * (delta_factor/206265**2))    
        
        return res      
  
#  #%% Infinite Correction Point Source 
#    def PSFPointSourceCorrection(self, image):
#        psf_inf = self.PSFInfFlux(image)
        
        
#%%
    def subtractUserBackground(self, subcube, r_in, r_out):
        res = []
        annulus = []
        annulus_aperture_list = []
        annulus_centroid = []
        res_rout = []
        anImg = subcube.cube_before.copy()
        
        for z in range(len(anImg)):
           
            img = anImg[z,:,:]
            j,k= subcube.xys[z]
            annulus_aperture = CircularAnnulus([j,k], r_in=r_in[z], r_out=r_out[z])
            # if r_in[z] > subcube.rs[z]:
            #     print("=== WARNING ===  "+subcube.name_band+" [ lambda"+str(subcube.ls[z])+"] Annulus inner r("+str(r_in[z])+") is greater than aperture("+str(subcube.rs[z])+")")
            annulus_aperture_list.append(annulus_aperture)
            annulus_masks = annulus_aperture.to_mask(method='center')
            annulus_data = annulus_masks.multiply(img)
            
            ww = np.where(annulus_data != 0)
            annulus_data_1d = annulus_data[ww]
            mask2 = np.where(~np.isnan(annulus_data_1d))
            annulus_data_1d = annulus_data_1d[mask2] #exclude the NaN 
            mean, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            anImg[z,:,:] = anImg[z,:,:]  -  median_sigclip
            annulus.append(annulus_masks)
            annulus_centroid.append([j,k])
            res.append(median_sigclip)
            res_rout.append(r_out)

        return  [anImg,res,annulus,annulus_centroid,annulus_aperture_list,res_rout]             
        
    
#%%
    def addMaxValue(self,image, prece):

        for z in range(len(image)):
            maxV = np.nanmax(image[z])
            maxAdd = prece * maxV / 5
            image[z] = image[z]+maxAdd
            
        return image



  #%% Get lambdas overlappinf correction 
    #def getLambdasOverlappingCorrection(self, ch1_data,ch1_ls,ch2_data,ch2_ls,delta1,delta2):
    #    print('Stitching channels and bands')
    #    ch1_start = np.where(np.array(ch1_ls)>= ch2_ls[0] - delta2/2 )[0][0]
    #    ch2_stop =  np.where(np.array(ch2_ls)<= ch1_ls[len(ch1_ls)-1]+ delta1/2)[0]
    #    ch2_stop =ch2_stop[len(ch2_stop)-1]                                                                     
    #    ch1_overlapping_before = ch1_data[ch1_start:]
    #    ch1_over_ls= ch1_ls[ch1_start:]
    #    ch2_overlapping = ch2_data[:ch2_stop+1]
    #    # print(len(ch1_overlapping_before),len(ch2_overlapping))
    #
    #    
    #    ch1_mean = np.mean(ch1_overlapping_before)
    #    ch2_mean = np.mean(ch2_overlapping)
    #    ratio = ch2_mean /ch1_mean
    #    # print(ratio,ch1_mean ,ch2_mean)
    #    ch1_fixed = []
    #    
    #    ch2_over_ls =  ch2_ls[:ch2_stop+1]
    #    for i in range(len(ch1_data)):
    #        ch1_fixed.append(ch1_data[i]*ratio)
    #    ch1_overlapping = ch1_fixed[ch1_start:]  
    

        # print(ch1_over_ls[0])
        # print(ch2_over_ls[0])
        # plt.plot(ch1_over_ls,ch1_overlapping_before, 'o' ,markersize=1, color='black',label='1st Subchannel Overlapping Part Before Scaling')
        # plt.plot(ch1_over_ls,ch1_overlapping,  'o' ,markersize=1, color='red', label='1st Subchannel Overlapping Part After Scaling')
        # plt.plot(ch2_over_ls,ch2_overlapping,  'o' ,markersize=1, color='green', label='2nd Subchannel Overlapping Part')
        # plt.xlabel('λ(μm)')
        # plt.ylabel('Flux')
        # plt.legend()
        # plt.show()        
        # # print('MSE = ',MSE(ch1_over_ls,ch2_over_ls))
        # plt.plot(ch1_ls,ch1_data, 'o' ,markersize= 0.5, color='black',label='Before Scaling')
        # plt.plot(ch1_ls,ch1_fixed,  'o' ,markersize=0.5, color='red', label='After Scaling')
        # plt.plot(ch2_ls,ch2_data,  'o' ,markersize=0.5, color='green', label='Following Sub-Channel')
        # plt.xlabel('λ(μm)')
        # plt.ylabel('Flux')
        # plt.legend()
        # plt.show()

    #    res_data = []
    #    res_ls = []
        # if len(ch1_overlapping) == len(ch2_overlapping):
        # print('fiiix')
        # res_data.extend(ch1_fixed[:ch1_start])
        # res_ls.extend(ch1_ls[:ch1_start])
        # for i in range(len(ch1_overlapping)):
        #         res_data.append((ch1_overlapping[i]+ch2_overlapping[i])/2)
        #         res_ls.append(ch1_over_ls[i])
        # res_ls.extend(ch2_ls[ch2_stop+1:])    
        # res_data.extend(ch2_data[ch2_stop+1:])
        # return [res_ls,res_data]

        # else:
        
    #    import pandas as pd
    #    res_all = []
    #    for i in range(len(ch1_fixed)):
    #            res_all.append([ch1_ls[i],ch1_fixed[i]])
    #            
    #    for i in range(len(ch2_data)):
    #            res_all.append([ch2_ls[i],ch2_data[i]])            
    #    df = pd.DataFrame(res_all, columns = ['ls', 'Flux']) #add here everithing
    #    df = df.sort_values(by=['ls'])
    #
    #    return [list(df.ls),list(df.Flux),ratio]

  
    #def fixSpectrumLambdas(self, all_data, all_ls, all_delta):
    #    print('Stitching band and channels....')
    #    ch1_data = all_data[0]
    #    ch1_ls = all_ls[0]
    #    tanio = []
    #  
    #    for i in range(len(all_data)-1):
    #
    #        ch2_data = all_data[i+1]
    #        ch2_ls = all_ls[i+1]
    #        [ls,apers,ratio ] = self.getLambdasOverlappingCorrection(ch1_data, ch1_ls, ch2_data, ch2_ls, all_delta[i], all_delta[i+1])
    #        ch1_data = apers
    #        ch1_ls = ls
    #        tanio.append(ratio)
    #
    #    return [ls, apers, tanio]


    #%%LOAD 1D EXTRACTION
    def Load1DFile(self,filename):
        import math
        hdu_list = fits.open(filename)
        data = hdu_list['EXTRACT1D'].data
        lambdas = data[:]['WAVELENGTH']
        flux = data[:]['FLUX']
        error =  data[:]['ERROR']
        bright = data[:]['SURF_BRIGHT']
        background = data[:]['BACKGROUND']
        backgroundError =  data[:]['BERROR']
        area =  data[:]['NPIXELS']
        r = np.sqrt(area / (math.pi))
        # fits.close()

        return [lambdas,flux,error,bright,background,backgroundError,area,r]

    def LoadAll1D(self,files):
        data_all = []
        lambdas_all = []
        flux_all = []
        error_all = []
        bright_all = []
        background_all = []
        backgroundError_all = []
        area_all = []
        r_all = []
   
        for i in range(len(files)):
            [lambdas,flux,error,bright,background,backgroundError,area,r] = self.Load1DFile(files[i])
            lambdas_all.extend(lambdas)
            flux_all.extend(flux)
            error_all.extend(error)
            bright_all.extend(bright)
            background_all.extend(background)
            backgroundError_all.extend(backgroundError)
            area_all.extend(area)
            r_all.extend(r)
        
        return [lambdas_all,flux_all,error_all,bright_all,background_all,backgroundError_all,area_all,r_all]   
#%% 
    def getSubcubesAll(self, subcubes, background, aperture_correction):
        all_rs_arcsec = []
        all_ls = []
        all_apers = []
        all_xys = []
        all_area_pix = []
        all_bright = []
        all_error_spectrum =[]
        all_corrected_spectrum = []
        all_delta = []
        all_background = []
        all_names = []
        all_error_corrected = []
        all_unit_ratio = []
        all_r_in = []
        all_rs = []
        all_ps = []
        all_psc_flux = []
        all_psc_err= []
        for i in range(len(subcubes)):
            all_rs_arcsec.extend(subcubes[i].rs_arcsec)
            all_rs.extend(subcubes[i].rs)
            all_ls.extend(subcubes[i].ls)
            all_apers.extend(subcubes[i].apers)
            all_xys.extend(subcubes[i].xys)
            all_error_spectrum.extend(subcubes[i].error)
            all_corrected_spectrum.extend(subcubes[i].corrected_spectrum)
            
            
            all_delta.extend(subcubes[i].CDELT3L)
            all_names.extend(subcubes[i].nameL)
            all_unit_ratio.extend(subcubes[i].unit_ratio)
            
            ps_list = [subcubes[i].pixel_scale] * len(subcubes[i].rs)
            all_ps.extend(ps_list)
            
            if (background): 
                all_background.extend(subcubes[i].background_spectrum)
                all_r_in .extend(subcubes[i].bckg_rs)    
            if (aperture_correction):
                all_psc_flux.extend(subcubes[i].spectrum_PSF_corrected)
                all_psc_err.extend(subcubes[i].error_PSF_corrected)

        return [all_rs_arcsec, all_ls, all_apers, all_xys, all_area_pix, all_bright, all_error_spectrum,\
                all_corrected_spectrum, all_delta, all_names, all_unit_ratio, all_background, all_r_in, all_rs, all_ps, all_psc_flux, all_psc_err]    
    
    
    #%%    
    def getSubcubesAllAppended(self,subcubes,background):
        all_rs = []
        all_ls = []
        all_apers = []
        all_xys = []
        all_area_pix = []
        all_bright = []
        all_error_spectrum =[]
        all_corrected_spectrum = []
        all_delta = []
        all_background = []
        all_names = []
        
        for i in range(len(subcubes)):
            all_rs.append(subcubes[i].rs)
            all_ls.append(subcubes[i].ls)
            all_apers.append(subcubes[i].apers)
            all_xys.append(subcubes[i].xys)
            # all_area_pix.append(subcubes[i].area_pix)
            # all_bright.append(subcubes[i].bright)
            all_error_spectrum.append(subcubes[i].error)
            all_corrected_spectrum.append(subcubes[i].corrected_spectrum)
            
            all_delta.append(subcubes[i].CDELT3)
            all_names.extend(subcubes[i].name_band)
            if (background) : all_background.append(subcubes[i].background_spectrum)
            
        return [all_rs,all_ls,all_apers,all_xys,all_area_pix,all_bright,all_error_spectrum,all_corrected_spectrum,all_delta,all_names,all_background]     

    #%%
    def listMJSR2Jy(self,data, ratio):
        res = []
        for i in range(len(data)):
            res.append(data[i] ** 10** 6 * (ratio[i] / 206265**2))
            
        return res    
    

    #%%Grid in Arcseconds 
    def createGridInArcSec(self, user_ra, user_dec, gridPoints_dist, gridPointsX, gridPointsY, cube, r_ap, pointSource, l_ap):
        NX = np.arange(0,gridPointsX)
        NY = np.arange(0,gridPointsY)
        
        gridPoints_pix = gridPoints_dist / cube.pixel_scale
        if r_ap == -1:
            raise ValueError('For some reason the radius is not defined') 
            r_ap = gridPoints_dist/2
            r_pix = ((gridPoints_pix/2)) 
        else:    
            r_pix = r_ap / cube.pixel_scale
        c1 = SkyCoord(user_ra, user_dec, unit="deg")  # defaults to      
        user_x, user_y, user_z = cube.wcs.world_to_pixel(c1, cube.ls[0]*u.um)
        grids_xs = user_x + (NX - (gridPointsX-1)/2) * gridPoints_pix
        grids_ys = user_y + (NY - (gridPointsY-1)/2) * gridPoints_pix
        
        
        sky_list = []
        pixels_list = []
        #coord_grid = []
        names = []
        sky_ra = []
        sky_dec = []
        for i in range(len(grids_xs)):
            for j in range(len(grids_ys)):
                sky = cube.wcs.pixel_to_world(grids_xs[i], grids_ys[j], 0)
                #coord_grid.append(sky)   
                sky_list.append(sky)
                pixels_list.append([i,j])
                names.append(str(i)+"_"+str(j))
                sky_ra.append(sky[0].ra)
                sky_dec.append(sky[0].dec)
                
        # for i in range(len(subchannels)):                
        #     self.plotGridSubchanel( user_ra, user_dec, gridPoints_dist, gridPointsX, gridPointsY, subchannels[i], r)
        # params_path = current_path+"/Params"
        # self.delteFilesatPath(params_path)
        # self.writeParamsFiles(coord_grid,r,l_ap,pointSource)        
        
        return sky_list, pixels_list, names, sky_ra, sky_dec
        

#%%   
    def plotGrid(self, cube, user_ra, user_dec, gridPoints_dist, gridPointsX, gridPointsY, r_as, output_path, output_filebase_name):

        NX = np.arange(0,gridPointsX)
        NY = np.arange(0,gridPointsY)
        from matplotlib.patches import Rectangle
        
        gridPoints_pix = gridPoints_dist / cube.pixel_scale
        if r_as == -1:
            r_as = gridPoints_dist/2
            r_pix = ((gridPoints_pix/2)) 
            # print("EXOUME grid_points: ", gridPoints_pix, " r: ", r_pix)
        else:    
            r_pix = r_as / cube.pixel_scale
            # print("EXOUME grid_points: ", gridPoints_pix, " xeirokinhto r: ", r_pix)
        c1 = SkyCoord(user_ra, user_dec, unit="deg")  # defaults to      
        user_x, user_y, user_z = cube.wcs.world_to_pixel(c1, cube.ls[0]*u.um)
        grids_xs = user_x +(NX - (gridPointsX-1)/2) * gridPoints_pix
        grids_ys = user_y +(NY - (gridPointsY-1)/2) * gridPoints_pix
        
        sky_list = []
        pixels_list = []
        #coord_grid = []
        names = []
        for i in range(len(grids_xs)):
            for j in range(len(grids_ys)):
                sky = cube.wcs.pixel_to_world(grids_xs[i], grids_ys[j],0)
                #coord_grid.append(sky)   
                sky_list.append(sky)
                pixels_list.append([grids_xs[i], grids_ys[j]])
                names.append(str(i)+"_"+str(j))
        
        img = np.nansum(cube.cube_before[0:10], axis=0)
        #img = cube.cube_before[0,:,:]
        #for i in range(1,len(cube.cube_before)):
        #    img = img + cube.cube_before[i,:,:]
        
        plt.figure()
        plt.subplot(projection = cube.wcs.celestial)
        im = plt.imshow(img, origin='lower', norm=LogNorm()) #, origin='lower'
        plt.colorbar(im)
        plt.plot(user_x, user_y, 'o', color="red", label="User Input Centroid")

        for i in range(len(pixels_list)):
            # xx = pixels_list[i][0] - (r_pix/2)
            xx = pixels_list[i][0] - r_pix
            yy = pixels_list[i][1] - r_pix
            # yy = pixels_list[i][1] - (r_pix/2)
            plt.gca().add_patch(Rectangle([xx,yy], 2*r_pix, 2*r_pix, linewidth=1, edgecolor='r', facecolor='none'))
            plt.plot(pixels_list[i][0], pixels_list[i][1], 'bo', markersize=3)
            plt.title(cube.name_band)
        plt.legend()
        plt.savefig(output_path+output_filebase_name+'_'+cube.name_band+'.png')
        #plt.show()
        plt.close()
         
        return sky_list, pixels_list, names            
            
    
    #%%
    def writeParamsFiles(self,sky_list,user_r_ap,lambda_ap, pointSource):
       #print('Ok prepei na nai')
       #print(repr(sky_list[0]))
       for i in range(len(sky_list)):
            f = open("Params/params_"+str(i)+".csv", "w")
            f.write('user_r_ap = '+str(user_r_ap)+"\n" )
            f.write('user_ra = '+str(sky_list[i][0].ra) +"\n" ) 
            f.write('user_dec = '+str(sky_list[i][0].dec) +"\n" )
            f.write('point_source = '+str(pointSource)+"\n" )
            f.write('lambda_ap = '+str(lambda_ap)+"\n" )
            f.write('aperture_correction = '+str(False)+"\n" ) 
            f.write('centering = '+str(False)+"\n" ) 
            f.write('lambda_cent = '+str(4.89049986650)+"\n" ) 
            f.write('background_sub  = '+str(False)+"\n" )
            f.write('r_ann_in = '+str(1.23)+"\n" )
            f.write('ann_width = '+str(1.23)+"\n" )
            #f.write('PSFs_path  = C:/Users/roub/Desktop/finale/PSFs/'+"\n" )
            #f.write('data_path   = C:/Users/roub/Desktop/finale/data/'+"\n" )
            #f.write('output_path   =C:/Users/roub/Desktop/finale/extractions/'+"\n" )
            
    #%%
    def getApertureDQList(self,cube):
       # print('DQ List ')
       res = []
       for i in range(len(cube.DQ)):
               # for j in range(len(cube.area[i])):
                   aper = cube.area[i][0]
                   # print("DQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ AAAAAAAAAAAA A A A A ", cube.area[i])
                   # aperstats2 = ApertureStats(cube.DQ[i,:,:], aper)
                   mask = aper.to_mask()
                   dq_masked = mask.cutout(cube.DQ[i,:,:])
                   dqv = np.max(dq_masked)
                   # print(aperstats2.max) 
                   res.append(dqv)

       return res     
   #%%
   

   # THIS FUNCTION HAS BEEN INTEGRATED IN THE CALCULATESTITCHRATIOS FUNCTION

   ##### Function that calculates the stitching ratio between all posible sub - bands ##### 
   #@realData: list of data SubCube elements 
   #@aperture_correction: boolean value 
    def stitchingRatioCalculation(self, realData, aperture_correction, idx, grid):
        # 'G140H',
        #cubesNames = [ 'G140H', 'ch_1_SHORT','ch_1_MEDIUM', 'ch_1_LONG' ,\
        #              'ch_2_SHORT','ch_2_MEDIUM', 'ch_2_LONG' ,
        #              'ch_3_SHORT','ch_3_MEDIUM', 'ch_3_LONG' ,
        #              'ch_4_SHORT','ch_4_MEDIUM', 'ch_4_LONG' ]

        allRatio = []
        for i in range(len(realData)-1):

            ratio = self.calculateStitchRatio(realData[i], realData[i+1], aperture_correction, idx, grid)
            allRatio.append(ratio)

        allRatio.append(1.)

        return allRatio


    #    stitchRatio = []
    #    subbandExist = []
    #    dct = {}
    #    # print(len(realData))
    #    for i in range(len(realData)-1):
    #
    #        #print(realData[i].name_band)
    #        dct[realData[i].name_band] = realData[i]
    #    # print("to leksiko einai ", str(repr(dict)))
    #    for i in cubesNames:
    #        exists = False
    #        for cube in realData:
    #            if(cube.name_band == i):
    #                exists = True
    #        if(exists == True): #if sub-band exists
    #           subbandExist.append(1) #put 1
    #        else:
    #           subbandExist.append(0) #else put 0
    #
    #    #cube_idx = 0
    #    allRatio = []
    #    #print("KAI TO SUBBANDS EXISTS PERIEXEIIIIIIIIIIIIIIIIIIII ", str(subbandExist))
    #    for i in range(len(subbandExist)-1):
    #        if(subbandExist[i] == 1 and subbandExist[i+1] == 1) and dct[cubesNames[i]].ls[-1] > dct[cubesNames[i+1]].ls[0]:
    #            # self.calculateStitchRatio(realData[cube_idx],realData[cube_idx+1])
    #            
    #            ratio = self.calculateStitchRatio(dct[cubesNames[i]], dct[cubesNames[i+1]], aperture_correction, idx, grid)
    #            #cube_idx = cube_idx+1
    #            allRatio.append(ratio)
    #            print('Stitching', cubesNames[i], 'and', cubesNames[i+1])
    #        else:
    #            #print('There is no stitching ratio between', cubesNames[i], 'and', cubesNames[i+1])
    #            allRatio.append(np.NaN)
    #    print('Stitching ratios:', allRatio)
    #    
    #    return allRatio        

                
    def calculateStitchRatios(self, realData, aperture_correction, idx, grid):

        allRatio = []
        for i in range(len(realData)-1):
            
            chA = realData[i]
            chB = realData[i+1]
            
            if aperture_correction: 
                #print('aderfia edw eimaste complettt')
                if grid:
                    chA_data = np.array(chA.spectrum_PSF_corrected) # not tested
                    chB_data = np.array(chB.spectrum_PSF_corrected) # not tested
                else:
                    chA_data = np.array(chA.spectrum_PSF_corrected)[idx,:]
                    chB_data = np.array(chB.spectrum_PSF_corrected)[idx,:]
                    
            else:   
                if grid:
                    chA_data = np.array(chA.corrected_spectrum)[idx,:]
                    chB_data = np.array(chB.corrected_spectrum)[idx,:]
                else:    
                    chA_data = np.array(chA.corrected_spectrum)[:,idx]
                    chB_data = np.array(chB.corrected_spectrum)[:,idx]
                    
            chA_ls = chA.ls
            chB_ls = chB.ls
            delta1 = chA.CDELT3
            delta2 = chB.CDELT3
            chA_start = np.where(np.array(chA_ls) >= chB_ls[0] - delta2/2)[0]
            chB_stop =  np.where(np.array(chB_ls) <= chA_ls[len(chA_ls)-1] + delta1/2)[0]
            
            if len(chA_start) == 0 or len(chB_stop) == 0:
                ratio = 1.
            
            else:
                chA_overlapping_data = chA_data[chA_start[0]:]
                chA_over_ls= chA_ls[chA_start[0]:]
                chB_overlapping_data = chB_data[:chB_stop[-1]+1]
                
                chA_median = np.nanmedian(chA_overlapping_data)
                chB_median = np.nanmedian(chB_overlapping_data)
                
                if np.isnan(chA_median) or np.isnan(chB_median):
                    ratio = 1.
                else:
                    ratio = chB_median / chA_median 
                    #chA_mean = np.mean(chA_overlapping_data)
                    #chB_mean = np.mean(chB_overlapping_data)
                    #ratio = chB_mean / chA_mean 
            
            allRatio.append(ratio)

        allRatio.append(1.)

        return allRatio


    #%%
    ##### Function for grid extraction, multiple points with square apertures #####
    ###############################################################################    
    # @ratio_list:  RA of grid central point in arcsec. (float)
    # @idx: dec of grid central point in arcsec. (float) 
    # @spectrum: Number of points in X pixel coordinates. (int)    
    ########### --> Return [df_res,realData_all,spec1ds]   ############################
    # @stitched_spectrum: The spectrum 1D element. (Spectrum1D)           
    ###############################################################################    
    def stitchSpectrum(self, ratio_list, idx, spectrum):

        firstone = np.where(np.array(ratio_list[idx:]) == 1.)[0][0]

        ratio = np.prod(ratio_list[idx:firstone+idx+1])

        return np.array(spectrum) * ratio


        #if np.isnan(ratio_list[idx]): #if there is no spectrum there
        #
        #    return [np.NaN] * len(spectrum)
        #
        #else:
        #    
        #    if np.isnan(np.sum(ratio_list[idx+1:])): #if there is nan at some next point
        #
        #        firstnan = np.where(np.isnan(ratio_list[idx:]))[0][0]
        #        
        #        ratio = 1 
        #        for i in range(idx, firstnan+idx):
        #            ratio = ratio * ratio_list[i]
        #
        #        return np.array(spectrum) * ratio
        #
        #    else:
        #        ratio = 1 
        #        for i in range(idx, len(ratio_list)):
        #            ratio = ratio * ratio_list[i]
        #
        #        return np.array(spectrum) * ratio    
            
            
  #%% 
    def delteFilesatPath(self, path):
        print()         

        files = glob.glob(path+'/*')
        for f in files:
            os.remove(f)          
    
    
#    def readGridParamsFile(self, path, filename):
#        print('reading Grid Parameters')
    
    
