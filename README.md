This code implements a simple least square method for the following problem. We have measures of abundances of some elements (Ar, Ca, Fe...), with their errors : the measure comes from the ICM (intra-cluster medium), or any medium where we can consider that the enrichment comes from a great amount of nucleosynthesis sources (such as supernovae and AGB). This code implements many models of supernovae and AGB, and their yields (in term of abundances). This code allows to determine the best combination of models explaining the observed abundances : allowing for example to provide informations about some privilegied models of SN1A, SNCC, AGBs... The list of the implemented models is not fixed, and can be improved by the user (it'll be explained at the end of this README).

### Fit

If you want to fit abundances over supernovae models, you'll first need to present your abundances in a json file, in the following way : 

`{`

`  "Si": [0.717, 0.101],`

`...`

`  "Fe": [0.769, 0.0151],`

`  "Ni": [0.64, 0.0845]`

`}`
Here, the first element of each list is the abundance value, the second being the error.
Then, you can change the DATA in `abunfit.py` to specify your file.
After that, you can perform a fit, in the following way : 

`a = AbunFit(DATA,   ['A22S03_0',"Ba06_DDTd","Iw99_W7new"] )`
Here, I choose three models : we try to fit the abundances with a linear combination of those 3 models (1 SNCC, 2 SN1A models here).
Then : 

`a.fit()`, `a.plot_fit()` and `a.corner()` to fit and display the results.

Finally, you can try to perform a mcmc, to estimate the errors around your results :


`a.run_mcmc(`

`n_walkers    = 64,     # > 2 × n_models`

`n_steps      = 3000,   # increase if tau is big`

`n_burn       = 500,    # burn-in to ignore`

`perturbation = 1e-3,   # initial dispersion around the fit`

`ci           = 68.27,  # CI (1σ)`

`)`


### Fitting and comparing many models

In the multifit part, you can make combinations over lists (it makes al the possible combinations between two or more sets of models).
Then, the code can tell you which combinations are the best.

You just have to call a `Multifit` object, in the following way :

`fit = Multifit([set1,set2,...],data_dir = DATA)`

Here, data_dir is the direction of the folder models. `[set1,set2,set3...]` is a list of all the sets of models.
A set of models can simply be a list of models : `[Ch04_2E-2,No13_SNe_0,Iw99_WDD3]`. It can more importantly be a keyword. The list of the keywords is :
- SNIA : the set of all the SNIA models ;
- SNCC : the set of all the SNCC models ;
- AGB  : the set of all the AGB models  ;
- DD : all the Delayed-Detonation (SNIA) available models ;
- DETONATION : all the Detonation (among them there are the DD for example) available models ;
- D2 : all the Double-Detonation (SNIA) av. models ;
- D6 : all the Dynamically-Driven Double-Degenerate Double-Detonation (SNIA) models ;
- HY : all the hypernovae models (SNCC) ;
- NMCH : all the Near-Chandrasekhar available models ;
- SMCH : all the Sub-Chandrasekhar available models ;
For example, if you want to search the fits for all the combinations of Delayed-Detonation SNIa with SNCCs, you just do : 
`fit = Multifit(["DD","SNCC"],data_dir = DATA)`

Then, to perform the fit, and display the results : 
`b.multifit()`
`b.plot_combo_map()`
`b.display_best_combos()`
