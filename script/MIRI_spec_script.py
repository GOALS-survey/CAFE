import sys
sys.path.insert(0, '../')

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages

#import creta
import cafe_io
import cafe_helper
import cafe

#source = 'NGC7469'
source = 'IIZw096'

# Read parameter file with extraction keywords
#pf = open('./CRETA/param_files/'+source+'_single_params.txt','r')
#print(pf.read())
#pf.close()

# Load the extraction tool (CRETA)
#c = creta()

# Perform the extraction
#c.singleExtraction(parameter_file=True, perband_cent=True, parfile_name=source+'_single_params.txt', data_path='./CRETA/data/'+source+'/', output_filebase_name=source)

if source == 'NGC7469':
    z=0.01630
    tplt='AGN'
elif source == 'IIZw096':
    z=0.03637
    tplt='SB'

source_fd = '../../CRETA/extractions/'
source_fn = source+'_SingleExt_r0.3as_cube.fits'
source_fnb = source_fn.split('.fits')[0].replace('.','')

cafe_dir = '../'
inppar_fn = cafe_dir+'inp_parfiles/inpars_jwst_miri_IIZw096.ini'
optpar_fn = cafe_dir+'opt_parfiles/default_opt.cafe'

s = cafe.specmod(cafe_dir)

s.read_spec(source_fn, file_dir=source_fd, z=z)
#s.plot_spec_ini(inppar_fn, optpar_fn)
s.fit_spec(inppar_fn, optpar_fn)

# Line and PAH averaged velocity gradient wrt z, in [km/s]
print(s.parcube['VALUE'].data[-1,0,0])
print(s.parcube['VALUE'].data[-1,0,0]/2.998e5)

#s.plot_spec_fit(inppar_fn, optpar_fn)

# Load the parameter cube from drive
parcube_fd = '../output/'+source_fnb+'/'
parcube_fn = source_fnb+'_parcube.fits'

s.read_spec(source_fn, file_dir=source_fd, z=z)
s.read_parcube_file(parcube_fn, file_dir=parcube_fd)

#s.plot_spec_ini(inppar_fn, optpar_fn)

ini = cafe.specmod(cafe_dir)
ini.read_parcube_file(parcube_fn, file_dir=parcube_fd)

#s.plot_spec_ini(inppar_fn, optpar_fn, ini_parcube=ini.parcube)

s.fit_spec(inppar_fn, optpar_fn, ini_parcube=ini.parcube)

# Line and PAH averaged velocity gradient wrt z, in [km/s]
print(s.parcube['VALUE'].data[-1,0,0])
print(s.parcube['VALUE'].data[-1,0,0]/2.998e5)

ini2 = cafe.specmod(cafe_dir)
ini2.read_parcube_file(parcube_fn, file_dir=parcube_fd)

s.fit_spec(inppar_fn, optpar_fn, ini_parcube=ini2.parcube)

#s.plot_spec_ini(inppar_fn, optpar_fn, ini_parcube=ini2.parcube)

# Line and PAH averaged velocity gradient wrt z, in [km/s]
print(s.parcube['VALUE'].data[-1,0,0])
print(s.parcube['VALUE'].data[-1,0,0]/2.998e5)

s.save_asdf(inppar_fn, optpar_fn, file_name=parcube_fn.split('.fits')[0])
