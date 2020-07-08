## MATLAB code to reproduce the simulations

- v1 runs the simulations for one particular value of r at a time,
- v1-5 is the same as v1 except that it does not compute the “Pure Synthetic Control” estimator (\lambda \to 0). It is made to run with a large value of n0 (typically n0=500 or 1,000) where computing the Delaunay triangulation necessary to obtain the pure synthetic control takes too much time,
- v2 is the same as v1 except that it runs the simulations for multiple values of r in one go. It also computes the Pure Synthetic Control as in v1.
- v2_f is the same as v2 but is a functionnal version.
- v2.5 also computes the different, optimal lambda (or M) for the bias-corrected estimators.
- v3 is the “v1-5 to v2” : it runs the simulations for multiple values of r in one go but does not compute the Pure Synthetic Control for computing time reasons.

In the paper: v2 has been used to compute Tables 1-3. v3 has been used to compute Table 4.
