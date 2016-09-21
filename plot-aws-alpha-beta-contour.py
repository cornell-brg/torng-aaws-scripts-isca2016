#=========================================================================
# plot-aws-alpha-beta-contour
#=========================================================================
# This script generates Figure 4 in the conference paper.
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
#plt.rcParams['font.usetex']        = True

#-------------------------------------------------------------------------
# Model
#-------------------------------------------------------------------------

IPC_values = np.arange(1.0,4.1,0.25)
E_values   = np.arange(1.0,4.1,0.25)

xi,yi      = np.meshgrid( IPC_values, E_values )
IPS_opt_  = np.empty([len(IPC_values),len(E_values)])
IPS_optr_ = np.empty([len(IPC_values),len(E_values)])

for i in range(len(IPC_values)):
  for j in range(len(E_values)):

    IPC_value = xi[i][j]
    E_value   = yi[i][j]

    # Debug printout
    # print IPC_value, E_value

    model = AwstealModel()
    model.N_B            = 4.0
    model.N_L            = 4.0
    model.N_BW           = 0.0
    model.N_LW           = 0.0
    model.beta           = IPC_value
    model.alpha          = E_value
    model.V_L            = np.arange(0.7,2.5,0.01)
    model.print_table    = False
    model.lambda_        = 0.5
    model.run()

    IPS_opt_[i][j]  = model.IPS_opt_
    IPS_optr_[i][j] = model.IPS_optr_

clines = np.arange(1.0,1.25,0.025)

ax = plt.subplot(2,2,1)
CS = plt.contour( IPC_values, E_values, IPS_opt_, clines, colors='red' )
plt.clabel( CS, inline=1, fontsize=8, inline_spacing=0, colors='red' )
plt.axis('equal')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.xlim(1,4)
plt.ylim(1,4)

# Turn off top and right border

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax = plt.subplot(2,2,2)
CS = plt.contour( IPC_values, E_values, IPS_optr_, clines, colors='red' )
plt.clabel( CS, inline=1, fontsize=8, inline_spacing=0, colors='red' )
plt.axis('equal')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.xlim(1,4)
plt.ylim(1,4)

# Turn off top and right border

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.tight_layout(w_pad=0)

#-------------------------------------------------------------------------
# Generate PDF
#-------------------------------------------------------------------------

input_basename = os.path.splitext( os.path.basename(sys.argv[0]) )[0]
output_filename = input_basename + '.py.pdf'

plt.savefig( output_filename, bbox_inches='tight' )

