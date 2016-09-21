#=========================================================================
# plot-aws-model-explore
#=========================================================================
# This script generates Figure 2 in the conference paper.
#
# Date   : September 20, 2016
# Author : Christopher Torng

import matplotlib.pyplot as plt
import math
import sys
import os.path

import numpy as np

from AwstealModel          import AwstealModel

from scipy.interpolate import splrep, splev
import matplotlib.patches as patches

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

fig_width_pt  = 112.0
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

#-----------------------------------------------------------------------
# plot
#-----------------------------------------------------------------------

def create_plot( beta, alpha, gamma ):

  V_Ls = np.arange(0.7,1.31,0.05)
  V_Bs = np.arange(0.7,1.2,0.05)

  labels = []
  EE     = []
  IPS    = []

  for V_L in V_Ls:
    for V_B in V_Bs:

      model = AwstealModel()
      model.N_B     = 4.0
      model.N_L     = 4.0
      N_BA          = 4.0
      N_LA          = 4.0
      model.N_BW    = model.N_B - N_BA
      model.N_LW    = model.N_L - N_LA
      model.beta    = beta
      model.alpha   = alpha
      model.gamma   = gamma
      model.lambda_ = 0.1
      model.init()

      P_L      = model.P_L( V_L )
      P_B      = model.P_B( V_B )
      IPS_L    = model.IPS_L( V_L )
      IPS_B    = model.IPS_B( V_B )

      P_tot    = N_LA * P_L + N_BA * P_B
      P_tot_   = P_tot / model.P_target

      IPS_tot  = ( N_LA * IPS_L + N_BA * IPS_B )
      IPS_tot_ = IPS_tot / model.IPS_N

      # power  = energy / time
      # energy = power * time
      # energy = power / perf
      # energy eff = perf / power

      EE_tot_  = IPS_tot_ / P_tot_

      EE.append(EE_tot_)
      IPS.append(IPS_tot_)
      labels.append( "{:3.1f}:{:3.1f}".format( V_L, V_B ) )

  plt.plot( IPS, EE, color='green', linestyle='',  marker='.', markersize=5 )

  # Draw isopower line

  plt.plot( [0.0, 2.0], [0.0, 2.0], linewidth=0.5, linestyle='-',
      color='b', zorder=1 )

  # Draw lines through (1,1)

  plt.axvline( x=1.0, linewidth=0.5, linestyle='-', color='r', zorder=1 )
  plt.axhline( y=1.0, linewidth=0.5, linestyle='-', color='r', zorder=1 )

  # Manually draw pareto optimal line

  # Pareto optimal line with lambda_ = 0.10
  pareto_x = [  0.8,   0.9,   0.95,   1.0, 1.05,   1.1, 1.125, 1.150, 1.175, 1.200, 1.25, 1.30 ]
  pareto_y = [ 1.36, 1.300, 1.2715, 1.233, 1.19, 1.145, 1.110, 1.090, 1.065, 1.040, 1.00, 0.95 ]

  # Pareto optimal line with lambda_ = 0.50
  #pareto_x = [  0.8,   0.9,   0.95,   1.0, 1.05,   1.1,  1.125, 1.15,  1.175, 1.2000 ]
  #pareto_y = [ 1.16, 1.157, 1.1515, 1.143, 1.13, 1.120,  1.110, 1.100, 1.090, 1.080 ]

  plt.plot( pareto_x, pareto_y, color='black', linestyle='--', linewidth=0.7 )
  plt.text( 0.98, 1.24, "Pareto Frontier", transform=plt.gca().transData,
      color='black' )
  plt.gca().add_patch( patches.Rectangle(
                         (0.98, 1.240),
                         0.44, 0.032,
                         color='white',
                         zorder=3 ) )

  # Draw goal point

  plt.plot( 1.11, 1.11, color='black', marker='.', markersize=12,
      markerfacecolor='white', markeredgewidth=1.0 )

  # Move gridlines behind plot

  plt.gca().set_axisbelow(True)

  #for label, ips, ee in zip( labels, IPS, EE ):
  #  plt.annotate( label, (ips, ee), fontsize=5 )

  plt.grid(True)
  plt.xlabel("Normalized IPS", fontsize=8 )
  plt.ylabel("Normalized Energy Efficiency", fontsize=8 )
  plt.xticks(np.arange(0.7, 1.31, 0.1))
  plt.yticks(np.arange(0.7, 1.31, 0.1))
  plt.xlim(0.7,1.3)
  plt.ylim(0.75,1.3)

  # Turn off top and right border

  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

  # Remove tick marks

  ax.tick_params( axis='both',
                  which='both',
                  left='off',
                  bottom='off' )

#-------------------------------------------------------------------------
# Generate plots
#-------------------------------------------------------------------------

plt.subplot(1,1,1)
create_plot( 2.0, 3.0, 0.25 )

#-------------------------------------------------------------------------
# Generate PDF
#-------------------------------------------------------------------------

input_basename = os.path.splitext( os.path.basename(sys.argv[0]) )[0]
if level == 100:
  output_filename = input_basename + '.py.pdf'
else:
  output_filename = input_basename + '_' + str(level) + '-split.py.pdf'

plt.savefig( output_filename, bbox_inches='tight' )

