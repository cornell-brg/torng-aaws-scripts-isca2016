#=========================================================================
# AwstealModel
#=========================================================================
# This is our simple first-order model for our asymmetry-aware
# work-stealing scheduler. The idea is that you create a model object, set
# various parameters, call "run", and then query various parameters for
# display or plotting.
#
# Date    : September 20, 2016
# Authors : Christopher Torng, Moyang Wang, and Christopher Batten
#

import numpy          as np
import scipy.optimize as opt

class AwstealModel():

  #-----------------------------------------------------------------------
  # Constructor
  #-----------------------------------------------------------------------
  # We set default values for all of the parameters here. These parameter
  # names correspond to those used in the conference paper
  # "Asymmetry-Aware Work-Stealing Runtimes" (ISCA '16) in Section 2: "A
  # Marginal-Utility-Based Approach".
  #
  # V_N             : nominal voltage
  # V_t             : threshold voltage
  # V_min           : minimum voltage
  # V_max           : maximum voltage
  #
  # N_L             : total number of little cores
  # N_B             : total number of big cores
  # N_LA            : total number of active little cores
  # N_BA            : total number of active little cores
  # N_LW            : total number of waiting little cores
  # N_BW            : total number of waiting big cores
  #
  # s               : power law relationship between power and voltage
  # F_ratio         : frequency ratio of big core to little core at V_N
  #
  # alpha           : energy ratio of big to little core at nominal
  #                   voltage and frequency
  #
  # beta            : performance ratio of big to little core at nominal
  #                   voltage and frequency
  #
  # lambda_         : static power of the big core as a percentage of the
  #                   big core's total power at nominal voltage and
  #                   frequency
  #
  # gamma           : little core's leakage current relative to big core's
  #                   leakage current when both are at nominal voltage and
  #                   frequency
  #
  # V_L             : voltage sweep for little core
  # V_B             : solved big core voltage for each little core voltage
  #                   in V_L after solving the optimization problem
  #
  # I_L             : leakage current of little core at nominal voltage
  # I_B             : leakage current of big core at nominal voltage
  #
  # IPS_N           : performance in IPS at nominal voltage
  #
  # P_target        : target power, aggregate power of all cores running
  #                   at nominal voltage and frequency
  #
  # P_alloc         : total power available for allocation across active
  #                   cores

  def __init__( s ):

    # Nominal supply, threshold voltage, and resting voltage level

    s.V_N   = 1.0
    s.V_t   = 0.3 # used with some vf models
    s.V_min = 0.7 # min voltage
    s.V_max = 1.3 # max voltage

    # Total number of big and little cores

    s.N_B = 4.0
    s.N_L = 4.0

    # How many cores are waiting

    s.N_BW = 0.0
    s.N_LW = 0.0

    # s captures the power-law relationship between power and voltage.
    # Note that s does not include the fact that frequency is also a
    # function of voltage. So the classic cubic relationship between power
    # and voltage would mean setting s to 2.0

    s.s = 2.0

    # Frequency ratio. How much faster is the big core at nominal?

    s.F_ratio = 1.0

    # This parameter captures the relative performance benefit of a big
    # core compared to a little core at nominal voltage and frequency.

    s.beta = 2.0

    # This parameter captures the relative energy overhead of a big core
    # compared to a little core at nominal voltage and frequency.

    s.alpha = 3.0

    # The lambda_ parameter sets the static power of the big core to a
    # certain percentage of the big core's total power at nominal. For
    # example, a value of 0.50 sets the leakage to be 50% of the total
    # power at nominal.

    # The gamma parameter captures the little core's leakage current
    # relative to the big core's leakage current when both are at nominal
    # voltage and frequency.

    s.static_en = 'yes'
    s.lambda_   = 0.50
    s.gamma     = 0.25

    # Voltage/Frequency model
    # v1: frequency is linear function of voltage with zero y-intercept
    # v2: frequency is linear function of voltage with non-zero y-intercept
    # v3: frequency is non-linear function of voltage

    s.vf_model = 'v2'

    # Voltage range to run model across.

    s.V_L = np.arange(0.7,1.3,0.02)

    # Switch to print out table with numerical data when run

    s.print_table = False

  #-----------------------------------------------------------------------
  # Basic frequency, power, performance equations
  #-----------------------------------------------------------------------

  # Frequency as a function of voltage

  def f( s, V ):

    if s.vf_model == 'v1':
      return 1.0 * V

    elif s.vf_model == 'v2':
      return (738388868.6 * V - 405388868.6) / 1000000

    elif s.vf_model == 'v3':
      return ( V - s.V_t )**2 / V

  # Dynamic power for little and big cores as function of voltage

  def P_L_dyn( s, V ):
    return s.alpha_L * s.IPC_L * s.f(V) * V**s.s

  def P_B_dyn( s, V ):
    return s.alpha_B * s.IPC_B * s.F_ratio * s.f(V) * V**s.s

  # Static power for little and big cores as function of voltage

  def P_L_static( s, V ):
    if s.static_en == 'yes':
      return V * s.I_L
    else:
      return 0.0

  def P_B_static( s, V ):
    if s.static_en == 'yes':
      return V * s.I_B
    else:
      return 0.0

  # Total power for little and big cores as function of voltage

  def P_L( s, V ):
    return s.P_L_dyn(V) + s.P_L_static(V)

  def P_B( s, V ):
    return s.P_B_dyn(V) + s.P_B_static(V)

  # Normalized total power

  def P_L_( s, V ):
    return s.P_L(V)/s.P_L(s.V_N)

  def P_B_( s, V ):
    return s.P_B(V)/s.P_L(s.V_N)

  # Performance for little and big cores as function of voltage

  def IPS_L( s, V ):
    return s.IPC_L * s.f(V)

  def IPS_B( s, V ):
    return s.IPC_B * s.F_ratio * s.f(V)

  # Normalized performance

  def IPS_L_( s, V ):
    return s.IPS_L(V) / s.IPS_L(s.V_N)

  def IPS_B_( s, V ):
    return s.IPS_B(V) / s.IPS_L(s.V_N)

  #-----------------------------------------------------------------------
  # init
  #-----------------------------------------------------------------------

  def init( s ):

    # Calculate performance and power factors from given alpha and beta

    s.IPC_L   = 1.0
    s.IPC_B   = s.beta

    s.alpha_L = 1.0
    s.alpha_B = s.alpha

    # Figure out the leakage current, which we are assuming does not vary
    # with voltage to first order.

    # This is the algebra for the big core's static leakage current,
    # s.I_B.
    #
    # Given the static power P_s, the dynamic power P_d, and the
    # percentage lambda_ (the percentage that static power should
    # represent out of the total power), solve for P_s.
    #
    #   P_s / (P_s + P_d)     = lambda_
    #   P_s                   = lambda_ * (P_s + P_d)
    #   P_s                   = lambda_ * P_s + lambda_ * P_d
    #   P_s - lambda_ * P_s   =                 lambda_ * P_d
    #   P_s * ( 1 - lambda_ ) =                 lambda_ * P_d
    #   P_s                   = ( lambda_ * P_d ) / ( 1 - lambda_ )
    #
    # Then divide by nominal voltage V_N to calculate leakage current:
    #
    #   I_s                   = ( lambda_ * P_d ) / ( V_N - V_N * lambda_ )

    s.I_B = ( s.lambda_ * s.P_B_dyn(s.V_N) ) / ( s.V_N - s.V_N * s.lambda_ )
    s.I_L = s.I_B * s.gamma

    # Calculate number of active cores.

    s.N_LA = s.N_L - s.N_LW
    s.N_BA = s.N_B - s.N_BW

    # Total performance at nominal voltage is sum of performance of little
    # and big cores at nominal voltage.

    s.IPS_N  = s.N_LA * s.IPS_L(s.V_N) + s.N_BA * s.IPS_B(s.V_N)

    # Power constraint. Both active and inactive cores count towards the
    # power constraint.
    #
    # This is the total power available for _allocation_ among active
    # cores. Inactive cores must run at V_min, and this power is not
    # available for allocation to active cores.
    #
    # The power constraint is therefore calculated by summing the powers
    # of active cores at nominal voltage and frequency and adding the
    # power headroom available after resting inactive cores to V_min.

    s.P_alloc = s.N_LA * s.P_L(s.V_N) \
              + s.N_BA * s.P_B(s.V_N) \
              + s.N_LW * (s.P_L(s.V_N) - s.P_L(s.V_min)) \
              + s.N_BW * (s.P_B(s.V_N) - s.P_B(s.V_min))

    # Power target is the power with all cores at nominal voltage

    s.P_target = s.N_L * s.P_L(s.V_N) \
               + s.N_B * s.P_B(s.V_N)

  #-----------------------------------------------------------------------
  # run
  #-----------------------------------------------------------------------

  def run( s ):

    s.init()

    #---------------------------------------------------------------------
    # Case 1: Only little cores are active
    #---------------------------------------------------------------------
    # Handle case where only one core type is active. Give it all of the
    # power. Figure what voltage it can run at to use that power and the
    # corresponding performance.

    if s.N_LA > 0 and s.N_BA == 0:

      # Available power is evenly allocated to each active little core

      pwr = s.P_alloc / s.N_LA

      # Sweep the power for little core across the voltage range

      v_range = np.arange(0.7,5.0,0.002)
      P_L_values = s.P_L(v_range)

      # Pick the voltage that gives the closest power

      idx = (np.abs(P_L_values-pwr)).argmin()

      # Set this voltage as the optimum voltage

      s.V_L_opt    = v_range[idx]
      s.V_L_opt_   = s.V_L_opt / s.V_N

      # Calculate the IPS at this optimum voltage
      # This is the IPS with power evenly distributed across active cores

      s.IPS_L_opt  = s.IPS_L(s.V_L_opt)
      s.IPS_L_opt_ = s.IPS_L_opt / s.IPS_N

      # Restrict the voltage to V_max
      # The 'r' is for realistic

      if s.V_L_opt > s.V_max:
        s.V_L_optr    = s.V_max
        s.V_L_optr_   = s.V_L_optr / s.V_N
        s.IPS_L_optr  = s.IPS_L(s.V_max)
        s.IPS_L_optr_ = s.IPS_L_optr / s.IPS_N
      else:
        s.V_L_optr    = s.V_L_opt
        s.V_L_optr_   = s.V_L_opt_
        s.IPS_L_optr  = s.IPS_L_opt
        s.IPS_L_optr_ = s.IPS_L_opt_

      # Set big core voltages all to V_min, since they are all inactive and
      # resting in this case. Also set their IPS to zero since they are
      # not doing any useful work.

      s.V_B_opt     = s.V_min
      s.V_B_opt_    = s.V_min / s.V_N
      s.IPS_B_opt   = 0
      s.IPS_B_opt_  = 0

      s.V_B_optr    = s.V_min
      s.V_B_optr_   = s.V_min / s.V_N
      s.IPS_B_optr  = 0
      s.IPS_B_optr_ = 0

      # Compute the total IPS for both optimal and realistic scenarios
      # This is done by multiplying by the number of active cores

      s.IPS_opt     = s.N_LA * s.IPS_L_opt
      s.IPS_opt_    = s.IPS_opt  / s.IPS_N

      s.IPS_optr    = s.N_LA * s.IPS_L_optr
      s.IPS_optr_   = s.IPS_optr / s.IPS_N

      return

    #---------------------------------------------------------------------
    # Case 2: Only big cores are active
    #---------------------------------------------------------------------
    # Handle case where only one core type is active. Give it all of the
    # power. Figure what voltage it can run at to use that power and the
    # corresponding performance.

    if s.N_BA > 0 and s.N_LA == 0:

      # Available power is evenly allocated to each active big core

      pwr = s.P_alloc / s.N_BA

      # Sweep the power for big core across the voltage range

      v_range = np.arange(0.7,5.0,0.002)
      P_B_values = s.P_B(v_range)

      # Pick the voltage that gives the closest power

      idx = (np.abs(P_B_values-pwr)).argmin()

      # Set this voltage as the optimum voltage

      s.V_B_opt    = v_range[idx]
      s.V_B_opt_   = s.V_B_opt / s.V_N

      # Calculate the IPS at this optimum voltage
      # This is the IPS with power evenly distributed across active cores

      IPS_B_values = s.IPS_B(v_range)
      s.IPS_B_opt  = IPS_B_values[idx]
      s.IPS_B_opt_ = s.IPS_B_opt / s.IPS_N

      # Restrict the voltage to V_max
      # The 'r' is for realistic

      if s.V_B_opt > s.V_max:
        s.V_B_optr    = s.V_max
        s.V_B_optr_   = s.V_B_optr / s.V_N
        s.IPS_B_optr  = s.IPS_B(s.V_max)
        s.IPS_B_optr_ = s.IPS_B_optr / s.IPS_N
      else:
        s.V_B_optr    = s.V_B_opt
        s.V_B_optr_   = s.V_B_opt_
        s.IPS_B_optr  = s.IPS_B_opt
        s.IPS_B_optr_ = s.IPS_B_opt_

      # Set little core voltages all to V_min, since they are all inactive and
      # resting in this case. Also set their IPS to zero since they are
      # not doing any useful work.

      s.V_L_opt     = s.V_min
      s.V_L_opt_    = s.V_min / s.V_N
      s.IPS_L_opt   = 0
      s.IPS_L_opt_  = 0

      s.V_L_optr    = s.V_min
      s.V_L_optr_   = s.V_min / s.V_N
      s.IPS_L_optr  = 0
      s.IPS_L_optr_ = 0

      # Compute the total IPS for both optimal and realistic scenarios
      # This is done by multiplying by the number of active cores

      s.IPS_opt     = s.N_BA * s.IPS_B_opt
      s.IPS_opt_    = s.IPS_opt  / s.IPS_N

      s.IPS_optr    = s.N_BA * s.IPS_B_optr
      s.IPS_optr_   = s.IPS_optr / s.IPS_N

      return

    #---------------------------------------------------------------------
    # Case 3: A mix of big and little cores are active
    #---------------------------------------------------------------------
    # Imagine the power vs. performance curves (one curve for little cores
    # and one curve for big cores). Basically, in this case, we start at
    # the low end of the little core's curve and work our way up. At each
    # step, we calculate where we would be on the big core's curve
    # assuming we use up all available power.
    #
    # To do this, we sweep the range of V_L and calculate the
    # corresponding range of V_B, where V_B is calculated to use up the
    # remaining available power using a generic solver. Then we calculate
    # total IPS across the range and pinpoint where total IPS is
    # maximized.

    # Function to calculate v_B (voltage of big core) given v_L (voltage
    # of little core) under given power constraint. Uses a generic solver
    # so that this should hopefully work for any vf_model.
    #
    # This is the function we want to set to 0 and solve for. Basically we
    # are taking all the available power and then subtracting the power we
    # use for big and little cores. Solve to use up all available power
    # (i.e., set to 0).
    #
    # The `scipy.optimize.fsolve` function finds the roots of a non-linear
    # equation defined by func(x) = 0, given a starting estimate.

    def helper_func( x, y ):
      return s.P_alloc - s.N_BA * s.P_B(x) - s.N_LA * s.P_L(y)

    # For each V_L, solve for V_B (with starting estimate of 1.0 V). The
    # starting estimate just helps solve more quickly.

    s.V_B = np.empty(len(s.V_L))
    for i in range(len(s.V_L)):
      s.V_B[i] = opt.fsolve( helper_func, 1.0, args=(s.V_L[i]) )[0]

    # Mask off the voltages that are too low to remove noise from the
    # extreme ends of the plots later.

    V_B_mask = ( s.V_B > 0.35 )
    s.V_B = V_B_mask * s.V_B

    # The total IPS is the sum of IPS across cores using the calculated
    # V_L and V_B values. Inactive cores do not count towards the
    # performance.

    s.IPS_tot  = s.N_LA * s.IPS_L(s.V_L) + s.N_BA * s.IPS_B(s.V_B)
    s.IPS_tot_ = s.IPS_tot / s.IPS_N

    # For plotting, if the V_B voltage is too low, zero out the IPS_tot
    # entries

    s.IPS_tot  = V_B_mask * s.IPS_tot
    s.IPS_tot_ = V_B_mask * s.IPS_tot_

    # IPS_tot is what we are trying to maximize, so we go ahead and
    # figure out what the optimal V_L and V_B are at this point.

    idx          = np.argmax(s.IPS_tot)
    s.IPS_opt    = s.IPS_tot[idx]
    s.V_L_opt    = s.V_L[idx]
    s.V_B_opt    = s.V_B[idx]

    s.IPS_opt_   = s.IPS_opt  / s.IPS_N
    s.V_L_opt_   = s.V_L[idx] / s.V_N
    s.V_B_opt_   = s.V_B[idx] / s.V_N

    # Finding the realistic voltages
    #
    # At this point, we have the optimal voltages to maximize total IPS,
    # but we need to get the realistic voltages that respect V_min / V_max.

    # Calculate the first derivative of the marginal utility curves
    # These variables are used for plotting

    IPS_L_values = s.IPS_L(s.V_L)/s.IPS_L(s.V_N)
    IPS_B_values = s.IPS_B(s.V_B)/s.IPS_L(s.V_N)

    P_L_values   = s.P_L(s.V_L)/s.P_L(s.V_N)
    P_B_values   = s.P_B(s.V_B)/s.P_L(s.V_N)

    dP_dV_L      = np.gradient(P_L_values)   # first differential
    dIPS_dV_L    = np.gradient(IPS_L_values) # first differential
    s.dP_dIPS_L  = dP_dV_L / dIPS_dV_L       # element-wise division

    dP_dV_B      = np.gradient(P_B_values)   # first differential
    dIPS_dV_B    = np.gradient(IPS_B_values) # first differential
    s.dP_dIPS_B  = dP_dV_B / dIPS_dV_B       # element-wise division

    # Work backwards from the optimal point to find the first operating
    # point that is valid given V_min and V_max

    s.V_L_optr  = 0
    s.V_B_optr  = 0
    s.V_L_optr_ = 0
    s.V_B_optr_ = 0

    s.IPS_optr  = 0
    s.IPS_optr_ = 0

    for i in reversed(range(0,idx)):
      if     s.V_L[i] >= s.V_min and s.V_L[i] <= s.V_max \
         and s.V_B[i] >= s.V_min and s.V_B[i] <= s.V_max:
        s.V_L_optr  = s.V_L[i]
        s.V_B_optr  = s.V_B[i]
        s.V_L_optr_ = s.V_L[i] / s.V_N
        s.V_B_optr_ = s.V_B[i] / s.V_N

        s.IPS_optr  = s.IPS_tot[i]
        s.IPS_optr_ = s.IPS_tot[i] / s.IPS_N
        break

    # Now we have an operating point that respects V_min and V_max

    # This is just for plotting. Set the V_xlim to be the largest V_L
    # where both big and little core marginal utilities are positive.

    s.V_xlim = s.V_L[-1]
    for i, val in enumerate( s.V_L ):
      if val > 1.0: # check above 1.0 V
        if s.dP_dIPS_L[i] > 0.0 and s.dP_dIPS_B[i] > 0.0:
          s.V_xlim = val
        else:
          break

    # Remove potential bugs in plots

    if s.V_L_optr == 0:
      s.V_L_optr  = s.V_max
      s.V_B_optr  = s.V_max
      s.V_L_optr_ = s.V_max / s.V_N
      s.V_B_optr_ = s.V_max / s.V_N

      s.IPS_optr  = s.N_LA * s.IPS_L(s.V_max) + s.N_BA * s.IPS_B(s.V_max)
      s.IPS_optr_ = s.IPS_optr / s.IPS_N

    #-------------------------------------------------------------------------
    # Data Table Printout
    #-------------------------------------------------------------------------

    if s.print_table:

        # Calculations for table

        tmp_IPS_B     = s.IPS_B (s.V_B)
        tmp_P_B       = s.P_B   (s.V_B)
        tmp_dIPS_dV_B = [ tmp_IPS_B[i] - tmp_IPS_B[i-1] for i in range( len(tmp_IPS_B) ) ]
        tmp_dP_dV_B   = [ tmp_P_B  [i] - tmp_P_B  [i-1] for i in range( len(tmp_P_B)   ) ]
        tmp_dIPS_dP_B = [ ips / p for ips, p in zip( tmp_dIPS_dV_B, tmp_dP_dV_B ) ]

        tmp_IPS_L     = s.IPS_L (s.V_L)
        tmp_P_L       = s.P_L   (s.V_L)
        tmp_dIPS_dV_L = [ tmp_IPS_L[i] - tmp_IPS_L[i-1] for i in range( len(tmp_IPS_L) ) ]
        tmp_dP_dV_L   = [ tmp_P_L  [i] - tmp_P_L  [i-1] for i in range( len(tmp_P_L)   ) ]
        tmp_dIPS_dP_L = [ ips / p for ips, p in zip( tmp_dIPS_dV_L, tmp_dP_dV_L ) ]

        P_alloc = s.N_LA * s.P_L(s.V_L) \
                + s.N_BA * s.P_B(s.V_B) \
                + s.N_LW * s.P_L(s.V_min) \
                + s.N_BW * s.P_B(s.V_min)

        # Table data

        table = {

            'V_L'        : s.V_L,
            'P_L'        : s.P_L(s.V_L),
            'P_L_'       : P_L_values,
            'F_L'        : s.f(s.V_L),
            'IPS_L'      : s.IPS_L(s.V_L),
            'IPS_L_'     : IPS_L_values,
            'dIPS/dP_L'  : tmp_dIPS_dP_L,
            'dIPS/dP_L!' : s.dP_dIPS_L,

            'V_B'        : s.V_B,
            'P_B'        : s.P_B(s.V_B),
            'P_B_'       : P_B_values,
            'F_B'        : s.f(s.V_B),
            'IPS_B'      : s.IPS_B(s.V_B),
            'IPS_B_'     : IPS_B_values,
            'dIPS/dP_B'  : tmp_dIPS_dP_B,
            'dIPS/dP_B!' : s.dP_dIPS_B,

            'IPS_tot'    : s.IPS_tot,
            'IPS_tot_'   : s.IPS_tot_,
            'P_alloc'    : P_alloc,
            }

        # Order the columns for print out

        labels  = [
            'V_L', 'P_L', 'P_L_', 'F_L', 'IPS_L', 'IPS_L_', 'dIPS/dP_L', 'dIPS/dP_L!',
            'V_B', 'P_B', 'P_B_', 'F_B', 'IPS_B', 'IPS_B_', 'dIPS/dP_B', 'dIPS/dP_B!',
            'IPS_tot', 'IPS_tot_', 'P_alloc' ]

        # Get the columns into the same order as the labels

        data_columns = [ table[key] for key in labels ]

        # Print header

        header_str   = len(labels) * '''{:<10}  '''
        header_str   = header_str.format( *labels )
        dashes_str   = '-' * len( header_str )
        print header_str
        print dashes_str

        # Print data

        index_for_table_start = 30

        template_str = len(labels) * '''{:0<10.6}  '''
        for i in xrange(51):
          ind = index_for_table_start + i
          # Assemble the row data by accessing the same index in each column
          row_data = [ col[ind] for col in data_columns ]
          print template_str.format( *row_data )

