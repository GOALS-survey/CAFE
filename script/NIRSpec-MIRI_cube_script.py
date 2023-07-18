import sys
import astropy.units as u
import pdb, ipdb

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages

import cafe
import cafe_io
import cafe_lib
import cafe_helper

source = 'NGC7469'
#source = 'IIZw096'

# Read parameter file with extraction keywords
#pf = open('./CRETA/param_files/'+source+'_grid_params.txt','r')
#print(pf.read())
#pf.close()

# Load the extraction tool (CRETA)
#c = creta.creta()

# Perform the extraction
#c.gridExtraction(parameter_file=True, perband_cent=True, parfile_name=source+'_grid_params.txt', data_path='./CRETA/data/'+source+'/', output_filebase_name=source)

if source == 'NGC7469':
    z=0.01630
    tplt='AGN'
elif source == 'IIZw096':
    z=0.03645
    tplt='SB'

source_fd = './CRETA/extractions/'
source_fn = source+'_GridExt_3x3_s0.5as_cube.fits'
source_fnb = source_fn.split('.fits')[0].replace('.','')
inppar_fn = './inp_parfiles/inpars_jwst_nirspec-miri_'+tplt+'.ini'

optpar_fn = './opt_parfiles/default_opt.cafe'

s = cafe.cubemod()

s.read_cube(source_fn, file_dir=source_fd, z=z)

s.plot_cube_ini(1, 1, inppar_fn, optpar_fn)

#s.fit_cube(inppar_fn, optpar_fn)

#s.plot_cube_fit(1, 1, inppar_fn, optpar_fn)

# Load the parameter cube from drive
parcube_fd = '/Users/tanio/Sync/pywork/CAFE_dev/output/'+source_fnb+'/'
parcube_fn = source_fnb+'_parcube.fits'

#s.read_cube(source_fn, file_dir=source_fd, z=z)

#s.read_parcube_file(parcube_fn, file_dir=parcube_fd)

#s.plot_cube_fit(1, 1, inppar_fn, optpar_fn)
