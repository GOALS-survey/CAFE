import sys
import ipdb
# Include the path to the folder where CAFE and CRETA have been installed. Usually it's one level up from the notebook/ folder.
sys.path.insert(0, '../CRETA/')
sys.path.insert(0, '../CAFE/')

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# NAME OF GALAXY
gal_name = 'NGC7469'

# SPECTRAL EXTRACTION
import creta
creta_dir = '../CRETA/'

## Read parameter file with extraction keywords
#param_fn = gal_name+'_MIRI_single_params.txt'
#pf = open(creta_dir+'param_files/'+param_fn,'r')
#print(pf.read())
#pf.close()

## Load the extraction tool (CRETA)
#c = creta.creta(creta_dir)

## Perform the extraction
#c.singleExtraction(parameter_file=True, parfile_name=param_fn, data_path=creta_dir+'data/'+gal_name+'/', output_filebase_name=gal_name)

# SPECTRAL FITTING
import cafe_io
from cafe_io import *
import cafe_helper
import cafe
cafe_dir = '../CAFE/'

# Setup data directory and file name, and parameter files.
source_fd = creta_dir+'extractions/'
source_fn = gal_name+'_SingleExt_r0.3as_cube.fits'
source_fnb = source_fn.split('.fits')[0].replace('.','')

inppar_fn = cafe_dir+'inp_parfiles/inpars_jwst_miri_AGN.ini'
optpar_fn = cafe_dir+'opt_parfiles/default_opt.cafe'

z=0.01623

# Load CAFE
s = cafe.specmod(cafe_dir)

# Read the spectrum
s.read_spec(source_fn, file_dir=source_fd, z=z)

# Plot initial model
s.plot_spec_ini(inppar_fn, optpar_fn)

# Fit spectrum
s.fit_spec(inppar_fn, optpar_fn)

# Line and PAH averaged velocity gradient wrt z, in [km/s]
print(s.parcube['VALUE'].data[-1,0,0])
print(s.parcube['VALUE'].data[-1,0,0]/2.998e5)

# Plot fitted spectrum
s.plot_spec_fit(inppar_fn, optpar_fn)

########### Starting a CAFE session from scratch ###############

parcube_fd = '../CAFE/output/'+source_fnb+'/'
parcube_fn = source_fnb+'_parcube.fits'

# Load the the spectrum
s.read_spec(source_fn, file_dir=source_fd, z=z)

# Load the parameter cube from disk
s.read_parcube_file(parcube_fn, file_dir=parcube_fd)

# Plot the previous fit
s.plot_spec_fit(inppar_fn, optpar_fn)


ipdb.set_trace()

# Note this is technically unnecessary if the initalization is done with the same spectrum, since the parameter cube is already loaded in s.parcube, but for completeness:
ini = cafe.specmod(cafe_dir)
ini.read_parcube_file(parcube_fn, file_dir=parcube_fd)

# Plot the initialized spectrum (should be the same as the fitted spectrum above)
s.plot_spec_ini(inppar_fn, optpar_fn, ini_parcube=ini.parcube)


ipdb.set_trace()
