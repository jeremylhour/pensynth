# pensynth

Codes corresponding to "A Penalized Synthetic Control Estimator for Disaggregated Data", Alberto Abadie and Jérémy L'Hour

Journal of the American Statistical Association,  December 2021, 116(536), 1817–1834.

ABSTRACT:

Synthetic control methods are commonly applied in empirical research to estimate the effects of treatments or interventions on aggregate outcomes. A synthetic control estimator compares the outcome of a treated unit to the outcome of a weighted average of untreated units that best resembles the characteristics of the treated unit before the intervention. When disaggregated data are available, constructing separate synthetic controls for each treated unit may help avoid interpolation biases. However, the problem of finding a synthetic control that best reproduces the characteristics of a treated unit may not have a unique solution. Multiplicity of solutions is a particularly daunting challenge when the data includes many treated and untreated units. To address this challenge, we propose a synthetic control estimator that penalizes the pairwise discrepancies between the characteristics of the treated units and the characteristics of the units that contribute to their synthetic controls. The penalization parameter trades off pairwise matching discrepancies with respect to the characteristics of each unit in the synthetic control against matching discrepancies with respect to the characteristics of the synthetic control unit as a whole. We study the properties of this estimator and propose data-driven choices of the penalization parameter. 

[Available here.](https://economics.mit.edu/sites/default/files/publications/A%20Penalized%20Synthetic%20Control%20Estimator%20for%20Disagg.pdf)

Code: [https://github.com/jeremylhour/pensynth](https://github.com/jeremylhour/pensynth)

LIST OF RELEVANT FILES :

- Simulations : [simulations/Main_MonteCarlo_v2.m](simulations/Main_MonteCarlo_v2.m) has been used to compute Tables 1-3. [simulations/Main_MonteCarlo_v3.m](simulations/Main_MonteCarlo_v3.m) has been used to compute Table 4.
- Empirical application : [examples/DelaunayExampleforPaper.R](examples/DelaunayExampleforPaper.R), and [incremental_algo_puresynth/main.py](incremental_algo_puresynth/main.py) is used to compute the pure synthetic control.

WARNING :

Regarding the files contained in the [examples](examples/) folder, except for [DelaunayExampleforPaper.R](DelaunayExampleforPaper.R) and [EXB_Lalonde.R](EXB_Lalonde.R) that are used in the paper, there are no guarantee and there might remain bugs, coding mistakes, etc. Please check it carefully before applying.
