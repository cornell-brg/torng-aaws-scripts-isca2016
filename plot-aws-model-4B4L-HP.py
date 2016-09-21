#=========================================================================
# plot-aws-model-4B4L-HP
#=========================================================================
# This script generates Figure 3 in the conference paper.
#
# Date   : September 20, 2016
# Author : Christopher Torng
#

import matplotlib.pyplot as plt
import math
import sys
import os.path

import numpy as np

from AwstealModel       import AwstealModel
from plot_awsteal_model import plot_awsteal_model

#-------------------------------------------------------------------------
# Command line
#-------------------------------------------------------------------------

if len(sys.argv) == 1:
  level = 100
else:
  level = int(sys.argv[1])

#-------------------------------------------------------------------------
# Calculate figure size
#-------------------------------------------------------------------------
# We determine the fig_width_pt by using \showthe\columnwidth in LaTeX
# and copying the result into the script. Change the aspect ratio as
# necessary.

fig_width_pt  = 244.0
inches_per_pt = 1.0/72.27                     # convert pt to inch
aspect_ratio  = 0.9

fig_width     = fig_width_pt * inches_per_pt  # width in inches
fig_height    = fig_width * aspect_ratio      # height in inches
fig_size      = [ fig_width, fig_height ]

#-------------------------------------------------------------------------
# Configure matplotlib
#-------------------------------------------------------------------------

plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['font.size']          = 8
plt.rcParams['font.family']        = 'serif'
plt.rcParams['font.serif']         = ['Times']
plt.rcParams['figure.figsize']     = fig_size

#-------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------

model = AwstealModel()
model.N_B = 4.0
model.N_L = 4.0
N_BA      = 4.0
N_LA      = 4.0

model.N_BW           = model.N_B - N_BA
model.N_LW           = model.N_L - N_LA
model.beta           = 2.0
model.alpha          = 3.0
model.V_L            = np.arange(0.7,2.5,0.01)
model.print_table    = False
model.lambda_        = 0.5
model.run()

print "model.V_L_opt_  ", model.V_L_opt_
print "model.V_B_opt_  ", model.V_B_opt_
print "model.IPS_opt_  ", model.IPS_opt_

print "model.V_L_optr_ ", model.V_L_optr_
print "model.V_B_optr_ ", model.V_B_optr_
print "model.IPS_optr_ ", model.IPS_optr_

print "model.IPS_N     ", model.IPS_N
print "model.IPS_opt   ", model.IPS_opt
print "model.IPS_optr  ", model.IPS_optr

#-------------------------------------------------------------------------
# Create plot
#-------------------------------------------------------------------------

plot_awsteal_model( model )

#-------------------------------------------------------------------------
# Generate PDF
#-------------------------------------------------------------------------

input_basename = os.path.splitext( os.path.basename(sys.argv[0]) )[0]
if level == 100:
  output_filename = input_basename + '.py.pdf'
else:
  output_filename = input_basename + '_' + str(level) + '-split.py.pdf'

plt.savefig( output_filename, bbox_inches='tight' )

