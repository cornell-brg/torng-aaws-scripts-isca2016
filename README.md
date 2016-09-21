Asymmetry-Aware Work-Stealing Runtimes: Analytical Model Scripts
================================================================

- Authors : Christopher Torng, Moyang Wang, and Christopher Batten
- Contact : clt67@cornell.edu

This is our simple first-order model for our asymmetry-aware work-stealing scheduler. The idea is that you create a model object, set various parameters, call "run", and then query various parameters for display or plotting.

The original conference paper is available here:

- http://www.csl.cornell.edu/~cbatten/pdfs/torng-aaws-isca2016.pdf

An leakage errata and addendum is available here:

- http://www.csl.cornell.edu/~ctorng/pdfs/torng-aaws-errata-isca2016.pdf

Various plotting scripts are available in this repo. Each script corresponds to one of the analytical model plots (i.e., Figures 2-5) in the conference paper.

- Figure 2: Pareto-Optimal Frontier for 4B4L System
    - plot-aws-model-explore.py
- Figure 3: 4B4L System w/ All Cores Active
    - plot-aws-model-4B4L-HP.py
- Figure 4: Theoretical Speedup for 4B4L System vs. alpha and beta
    - plot-aws-alpha-beta-contour.py
- Figure 5: 4B4L System w/ 2B2L Active
    - plot-aws-model-4B4L-LP.py

Please refer to the descriptions of each plot in the conference paper.

Run each plot script like this:

    % python plot-aws-model-explore.py
    % python plot-aws-model-4B4L-HP.py
    % python plot-aws-alpha-beta-contour.py
    % python plot-aws-model-4B4L-LP.py

A PDF file with the plot will then be generated in the current directory.

