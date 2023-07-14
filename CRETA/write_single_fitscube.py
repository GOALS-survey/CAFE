def write_single_fitscube(file_name, output_name): 

       import numpy as np
       from astropy import units as u
       from specutils import Spectrum1D
       import astropy
       from astropy.nddata import StdDevUncertainty
       from specutils import SpectrumList
       import pandas as pd 
       from astropy.io import fits  
       
       if output_name is None: output_name = file_name.split('.fits')[0]+'_cube.fits'
       
       hdu_list = fits.open(file_name)
       res_spec1d = []
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
           Flux_ap = table["Flux_ap"] * u.Jy
           Err_ap = table["Err_ap"] * u.Jy
           Flux_ap_st = table["Flux_ap_st"] * u.Jy
           Err_ap_st = table["Err_ap_st"] * u.Jy
           if aperture_correction:
               Flux_ap_PSC = table['Flux_ap_PSC'] * u.Jy
               Flux_Err_ap_PCS = table['Flux_Err_ap_PCS'] * u.Jy
           DQ = table["DQ"]
           Band_Names = table["Band_Name"]
           dct['band_name'] = Band_Names

           fluxes = [Flux_ap[i], Flux_ap_st[i], DQ[i]]
           errors = [Err_ap[i],Err_ap_st[i]]
           if aperture_correction:
               fluxes.append(Flux_ap_PSC[i])
               errors.append(Flux_Err_ap_PCS[i])
           errors.append(len(DQ[i]) * [0])
           q = astropy.units.Quantity(np.array(fluxes), unit=u.Jy) 
           unc = StdDevUncertainty(np.array(errors))
           pec1d = Spectrum1D(spectral_axis=wave[i].T, flux=q, uncertainty=unc, meta=dct)
           res_spec1d.append(pec1d)
           res = SpectrumList(res_spec1d)    
  
           dct = {}   
           dct = res[0].meta

         
           fluxes = res[0].flux[0]
           fluxes_stitched =res[0].flux[1]
           errors = res[0].uncertainty.array[0]
           errors_stitched = res[0].uncertainty.array[1]
           DQ = res[0].flux[-1]
            
           if aperture_correction:
                fluxes_PSC =res[0].flux[2]
                errors_PSC= res[0].uncertainty.array[0]
            
            # for i in range(len(dct)):
            #     for j in range(len(dct[i])):
            #         fluxes[:,j,i] = dct[i][j].flux[0,:]
            #         fluxes_stitched[:,j,i] = dct[i][j].flux[1,:]
            #         DQ[:,j,i] = dct[i][j].flux[2,:]  
            #         errors[:,j,i] = dct[i][j].uncertainty.array[0,:]
            #         errors_stitched[:,j,i] = dct[i][j].uncertainty.array[1,:]
            #         if aperture_correction:
            #             fluxes_PSC[:,j,i] = dct[i][j].flux[2,:]
            #             errors_PSC= dct[i][j].uncertainty.array[2,:]
            #             DQ[:,j,i] = dct[i][j].flux[2,:] 
                        
                    
            
            
            #%%write FITS multicard
           keys = res[0].meta.keys()
           values = list(res[0].meta.values())
           NAXIS1, NAXIS2, NAXIS3 = 1, 1, len(fluxes)
            
           import astropy.io.fits as fits
           hdu = fits.PrimaryHDU()
            
           h_flux = fits.ImageHDU(fluxes.value, name='FLUX')
           header = h_flux.header  
           dictionary =res[0].meta
           header['PCOUNT'] = 0
           header['GCOUNT'] = 1
           header['EXTNAME'] = 'FLUX'
           header['SRCTYPE'] = 'EXTENDED'
           header['BUNIT'] = 'Jy/pix'
           header['WCSAXES'] = 3 
           header['CRVAL1'] = float(dictionary["'extraction_RA'"].split(" ")[1]) #Extraction RA
           header['CRVAL2'] = float(dictionary[" 'extraction_DEC'"].split(" ")[1])  #Extraction DEC
           header['CRVAL3'] = (1)
           header['CDELT1'] = (0)
           header['CDELT2'] = (0)
           header['CDELT3'] = float(0)
           header['CTYPE1'] = 'RA---TAN'
           header['CTYPE2'] = 'DEC---TAN'
           header['CTYPE3'] = 'WAVE'
           header['CUNIT1'] = 'deg'
           header['CUNIT2'] = 'deg'
           header['CUNIT3'] = 'um '
           header['PC1_1']  = -1
           header['PC1_2']  = 0.0
           header['PC1_3']  = 0
           header['PC2_1']  = 0.0
           header['PC2_2']  = 1.0
           header['PC2_3']  = 0
           header['PC3_1']  = 0
           header['PC3_2']  = 0
           header['PC3_3']  = 1
           values[1] = values[1].replace("'", "")
           header['EXTRTYPE'] = dictionary[" 'exrtaction_type'"]
           header['APRAD'] = float(dictionary[" 'r_ap'"].split("'")[1])
           #add GRCNTRA , dec
           from astropy.table import Table
           df_names = pd.DataFrame(res[0].meta['band_name'][0])
           df_names.columns = ['Band_name']
           t_names = Table.from_pandas(df_names)
           
           h_err= fits.ImageHDU(errors, name='Err')
           h_flux_stitched = fits.ImageHDU(fluxes_stitched.value, name='Flux_st')
           h_wave = fits.ImageHDU(res[0].spectral_axis.value, name='Wave')
           h_err_stitched = fits.ImageHDU(errors_stitched, name='Err_st')
           h_dq = fits.ImageHDU(DQ.value, name='DQ')
           
           if aperture_correction:
                h_flux_PSC= fits.ImageHDU(fluxes_PSC.value, name='Flux_PSC')
                h_err_PSC = fits.ImageHDU(errors_PSC.value, name='Err_PSC')
            
           names_array = np.array(list(df_names['Band_name']))
           col1 = fits.Column(name='Band_name', format='20A', array=names_array)
           coldefs = fits.ColDefs([col1])
           h_names = fits.BinTableHDU.from_columns(coldefs, name = "Band_names")
            
           if aperture_correction:
                hdulist = fits.HDUList([hdu, h_flux,h_err, h_flux_PSC,h_err_PSC, h_flux_stitched, h_err_stitched, h_dq,h_wave, h_names])
           else:
                hdulist = fits.HDUList([hdu, h_flux,h_err,                       h_flux_stitched, h_err_stitched, h_dq,h_wave, h_names]) 
                
           hdulist.writeto(output_name, overwrite=True)
