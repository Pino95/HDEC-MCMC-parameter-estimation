# HDEC-MCMC-parameter-estimation
Running these scripts will allow you to obtain the inflationary prediction in the n_s-r plane for a non-canonical inflaton.

In this case it is applied to the Higgs-Dilaton model in Einstein-Cartan gravity (see arXiv:2202.04665)

The code "MCMC-Nieh-Yan(Dinesty).py" can scan the parameter space of your model to find out which subspace satisfy a specific duration of inflation, with the chosen accuracy.

The "Observables-generator.py" script finds the values of n_s and r for the sampled parameters.

"HD-ns-r-plotter.py" Plots the point in the n_s-r plane, color-code them with a chosen parameter, and overlap the constraints from BICEP/Keck18+Planck2018 and the combination of 
BICEP/Keck18+Planck2018+ACT+DESI Y1

"HD-fields-evolution.py" is a nice bonus to see the inflationary trajectories in field space
