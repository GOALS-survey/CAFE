import sys
# Include the path to the folder where CAFE and CRETA have been installed. Usually it's one level up from the notebook/ folder.
sys.path.insert(0, '../CRETA/')
sys.path.insert(0, '../CAFE/')

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# SPECTRAL EXTRACTION
import creta
creta_dir = '../CRETA/'

# Read parameter file with extraction keywords
gal_name = 'NGC7469'
param_fn = gal_name+'_MIRI_single_params.txt'
pf = open(creta_dir+'param_files/'+param_fn,'r')
print(pf.read())
pf.close()

# Load the extraction tool (CRETA)
c = creta.creta(creta_dir)

# Perform the extraction
c.singleExtraction(parameter_file=True, parfile_name=param_fn, data_path=creta_dir+'data/'+gal_name+'/', output_filebase_name=gal_name)
