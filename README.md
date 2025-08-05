# HDEC-MCMC-parameter-estimation
Running these scripts will allow you to obtain the inflationary prediction in the n_s-r plane for a non-canonical and non-minimally coupled inflaton.

In the file "model.py" you must include the analytical expressions for the field potential, conformal factor and kinetic function, together with the first two derivatives of such functions. Notice that you have to be consistent with frames. If you are in the Einstein frame you must set the conformal factor to 1.

The file "slow-roll.py" contains the definition of the slow roll parameters for an inflaton with generic potential, conformal factor and kinetic term. Notice that everything is expressed in the Einstein frame, so a Weyl redefinition is implicitly performed. Be careful as the number of e-folds is not frame-invariant. 

The code "MCMC-Dinesty.py" can scan the parameter space of your model to find out which subspace satisfy a specific duration of inflation, with the chosen accuracy.

The "Observables-generator.py" script finds the values of n_s and r for the sampled parameters. Here you can also define any other quantity you wish to compute.

"NsR-plotter.py" Plots the point in the n_s-r plane, color-code them with a chosen parameter, and overlap the constraints from BICEP/Keck18+Planck2018 and the combination of 
BICEP/Keck18+Planck2018+ACT+DESI Y1

