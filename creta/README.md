# CRETA: The CAFE Region Extraction Tool Automaton

## Introduction

CRETA is a spectral extraction tool for astronomical IFU cubes obtained with JWST. It has two different modes: single region (line of sight) and grid extractions. CRETA also provides a rich set of preprocessing options and extractions that can be customized via parameter files. 

 ![picture alt](https://github.com/roumpakis/CRETA/blob/main/Images/22.png?raw=true "CRETA")

### Files & Folders Description
| Filename    |Description |
|--------------|:-----|
| creta.py              | Contains main functionality of CRETA, the only object that user will interact with |   
| cube_handler.py       | Represents the information associated with each sub-band/cube |
| cube_preproc.py       | Class with functionalities |
| UserAPI.py            | Class for file handling |   
| mylmfit2dfun.py       | Function to perform fitting tasks  |   
| single_params.txt     | Text file that contains parameters for single extraction |
| grid_params.txt       | Text file that contains parameters for grid extraction |

| Folder    |Description |
|--------------|:-----|
| Data              | Contains data .fits files  |   
| PSFs              | Contains PSF fits files|
| PSF_infaps        | Infinite Flux per PSF sub band  |   
| PSF_centroids_sky | Centroids of PSF sub bands in sky coordinates |
| PSF_Centroids     | Centroids of PSF sub bands in pixel coordinates |
| extractions       | Folder where CRETA saves output files by default |

### How to run it
1. Create a ```cube_cp``` object that gives access to both extraction options:
```python 
c = creta()
```

2. Extraction with default options. Run a single region extraction with ```singleExtraction``` or a grid extraction with ```gridExtraction```. The parameters for the extractions are read from the parameter file(s):
```python 
c.singleExtraction()
c.gridExtraction()
```

### Parameter files

Single Extraction parameters:

```cubes```: NIRSpec/IFU and/or MIRI/MRS cubes to perform the extraction from.<br/>
```user_r_ap```: Radius of the circular aperture [arcsec].<br/>
```user_ra```: Center RA coordinates [hh:mm:ss or degrees].<br/>
```user_dec```: Center Dec coordinates [dd:mm:ss or degrees].<br/>
```point_source```: Point source (cone) or extended source (cylinder) extraction.<br/>
```lambda_ap```: When point source extractions, the reference wavelength for the aperture radius [um]. Ignored otherwise.<br/>
```apperture_correction```: Whether to perform aperture correction.<br/>
```centering```: Whether to perform a centroid around the user given coordinates using a 11x11 pixel box.<br/>
```lambda_cent```: When centering is requested, the reference wavelength at which it will be performed [um]. Ignored otherwise.<br/>
```background```: Whether to perform a background annulus subtraction prior to aperture photometry.<br/>
```r_ann_in```: When background subtraction is requested, the inner radius of the annulus [arcsec]. Ignored otherwise.<br/>
```width```: When background subtraction is requested, the width of the annulus [arcsec]. Ignored otherwise.<br/>
```aperture_type```:  Aperture type: 0 for Circular, 1 for Rectangular.<br/>
```convolve:```:  Not currently supported. Whether to onvolve the cubes to a given resolution (PSF size at a given wavelength).<br/>

Single extractions can be parametrized via command line. The user can define a subset or all parameter options based on the desired extraction.

```python 
c.singleExtraction(parameter_file=False, user_ra=273.66541, user_dec=34.0295, centering=True, lambda_cent=5.4, point_source=False)
```
---

Grid Extraction parameters:

```cubes```: NIRSpec/IFU and/or MIRI/MRS cubes to perform the extraction from.<br/>
```user_ra```: Center RA coordinates [degrees].<br/>
```user_dec```: Center Dec coordinates [degrees].<br/>
```user_center```: Use the user-defined center. Otherwise the center will be set to the FOV center of the last cube.<br/>
```centering```: Whether to perform a centroid around the user given coordinates using a 11x11 pixel box.<br/>
```lambda_cent```: When centering is requested, the reference wavelength at which it will be performed [um]. Ignored otherwise.<br/>
```nx_steps```: Number of grid points in X coordinate. Use -1 value for default option.<br/>
```ny_steps```: Number of grid points in Y coordinate. Use -1 value for default option. <br/>
```spax_size```: Size of the box side used for extraction at each grid point. Use -1 value for defalut option, which is the same size as the distance between two points.<br/>
```step_size:```: Distance between two grid points. Use -1 for default option, which is the pixel scale of the longest wavelength cube.<br/>
```convolve:```:  Not currently supported. Whether to onvolve the cubes to a given resolution (PSF size at a given wavelength).<br/>

Grid extraction can be parametrized via command line. The user can define a subset or all parameter options based on the desired extraction.

```python 
c.gridExtraction(parameter_file=False, user_ra=49.23411, user_dec=-23.2998, centering=True, lambda_cent=8.5, point_source=False)
```

---
#### Single Extraction Parameters
| Parameter    | Default Value | Data Type |
|--------------|------:|-----------:|
| cubes                |   None |    list|
| user_ra              |  0     |   float|
| user_dec             |  0     |   float|
| user_r_ap            | [0.25] |    list|
| point_source         | True   |    bool|
| lambda_ap            |  5     |   float|
| apperture_correction |  False |    bool|
| centering            |  False |    bool|
| lambda_cent          |      5 |   float|
| background           |  False |    bool|
| r_ann_in             |   None |   float|
| ann_width            |   None |   float|
| aperture_type        |      0 |     int|
| convolve             | False |     bool|


#### Grid Extraction Parameters
| Parameter    | Default Value | Data Type |
|--------------|------:|-----------:|
| cubes                | False |     bool|
| user_ra              |  0    |    float|
| user_dec             |  0    |    float|
| user_center          | True  |     bool|
| centering            | False |     bool|
| lambda_cent          |  None |    float|
| nx_steps             | -1    |      int|
| ny_steps             | -1    |      int|
| spax_size            | -1    |    float|
| step_size            | -1    |    float|
| convolve             | False |     bool|






