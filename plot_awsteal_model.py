#=========================================================================
# plot_awsteal_model
#=========================================================================
# Plot the normalized power vs performance and the dP/dIPS vs voltage

import matplotlib.pyplot as plt
import numpy             as np

from AwstealModel import AwstealModel

def plot_awsteal_model( model, subplot_a_ylim = -1, subplot_b_xlim = -1 ):

  s = model
  plt.clf()

  #-----------------------------------------------------------------------
  # Plot P vs IPS
  #-----------------------------------------------------------------------

  ax = plt.subplot(2,2,1)

  Vrange = np.arange(0.7,1.3,0.002)

  plt.plot( s.IPS_L_(Vrange), s.P_L_(Vrange), color='green' )
  plt.plot( s.IPS_L_(s.V_N),  s.P_L_(s.V_N),
            color='green', fillstyle='none', marker='o', markersize=7, markeredgewidth=1 )

  plt.plot( s.IPS_B_(Vrange), s.P_B_(Vrange), color='blue' )
  plt.plot( s.IPS_B_(s.V_N),  s.P_B_(s.V_N),
            color='blue', fillstyle='none', marker='o', markersize=7, markeredgewidth=1 )

  plt.plot( s.IPS_L_(s.V_L_opt), s.P_L_(s.V_L_opt),
            color='black', marker='*', markersize=9 )

  plt.plot( s.IPS_B_(s.V_B_opt), s.P_B_(s.V_B_opt),
            color='black', marker='*', markersize=9 )

  plt.plot( s.IPS_L_(s.V_L_optr), s.P_L_(s.V_L_optr),
            color='black', marker='.', markersize=9 )

  plt.plot( s.IPS_B_(s.V_B_optr), s.P_B_(s.V_B_optr),
            color='black', marker='.', markersize=9 )

  plt.grid(True)
  plt.xlabel("Normalized IPS")
  plt.ylabel("Normalized Power")
  plt.xlim(0,3)

  # Cut off plot with a reasonable margin from the top

  ylim = max( s.P_L_(s.V_L_opt),
              s.P_B_(s.V_B_opt),
              s.P_L_(s.V_L_optr),
              s.P_B_(s.V_B_optr),
              s.P_L_(s.V_N),
              s.P_B_(s.V_N) )

  # Use explicit ylim for this subplot if specified

  if subplot_a_ylim == -1 : plt.ylim(0, ylim + 2.5)
  else                    : plt.ylim(0, subplot_a_ylim)

  # Turn off top and right border

  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

  #-----------------------------------------------------------------------
  # Plot dP/dIPS and total IPS vs voltage
  #-----------------------------------------------------------------------

  # Use explicit xlim for this subplot if specified

  if subplot_b_xlim == -1 : V_xlim = s.V_xlim
  else                    : V_xlim = subplot_b_xlim

  ax1 = plt.subplot(2,2,2)

  ax1.plot( s.V_L, s.IPS_tot_, color='red', linewidth=1.5)

  ax1.plot( s.V_L_opt_,  s.IPS_opt_,  color='black', marker='*', markersize=9 )
  ax1.plot( s.V_L_optr_, s.IPS_optr_, color='black', marker='.', markersize=9 )

  ax1.axvline( s.V_L_opt_, color='black', linestyle='--', linewidth=1 )

  ax1.set_ylim(0,s.IPS_opt_+0.2)
  ax1.set_xlim(s.V_min,V_xlim+0.01)

  ax1.set_ylabel("Normalized Total IPS")

  ax2 = ax1.twinx()

  ax2.plot( s.V_L, s.dP_dIPS_L )
  ax2.plot( s.V_L, s.dP_dIPS_B )

  # Extra markers to make sure the two x-axes are aligned with each other.
  # Uncomment this to plot two stars on the two x-axes at the same
  # x-position. They should line up vertically.

#  ax1.plot( s.V_L_opt_, 0.1,  color='g', marker='*', markersize=9 )
#  ax2.plot( s.V_L_opt_, 0.2,  color='r', marker='*', markersize=9 )

  # Tweak y_lim to something and bring the marginal utility crossover
  # point into view

  ax2.set_ylim(0,8)

  ax2.set_xlim(s.V_min,V_xlim+0.01)

  ax1.grid()
  ax2.grid()
  ax2.yaxis.grid(False)

  ax2.set_yticks([])

  xticks = np.arange(s.V_min,V_xlim+0.01,0.3)
  ax1.set_xticks(xticks)
  ax2.set_xticks(xticks)

  xticklabels = []
  for xtick in xticks:
    # Get index of this xtick in V_L (this assumes the xtick is in V_L)
    idx = (np.abs(s.V_L-xtick)).argmin()
    # Use it to index into the corresponding V_B
    v_b = s.V_B[idx]
    if v_b >= 0.1:
      xticklabels.append( "{:3.2f}\n{:3.2f}".format(xtick,v_b) )

  # idx = (np.abs(s.V_L-xticks[0])).argmin()
  # v_b = s.V_B[idx]
  # xticklabels[0] = "$V_L$: {:3.2f}    \n$V_B$: {:3.2f}    ".format(xtick,v_b)

  ax1.set_xticklabels( xticklabels, fontsize=8 )

  # ax1.set_xlabel("{} {}".format(s.IPS_opt_,s.IPS_optr_))

  # Turn off top and right border

  ax1.spines['right'].set_visible(False)
  ax1.spines['top'].set_visible(False)
  ax1.xaxis.set_ticks_position('bottom')
  ax1.yaxis.set_ticks_position('left')

  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax2.xaxis.set_ticks_position('bottom')
  ax2.yaxis.set_ticks_position('left')

  plt.tight_layout(w_pad=0)

