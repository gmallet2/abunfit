### Fit

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

For example, if you want to search the fits for all the combinations of Delayed-Detonation SNIa with SNCCs, you just do : 
`fit = Multifit(["DD","SNCC"],data_dir = DATA)`

Then, to perform the fit, and display the results : 
`b.multifit()`
`b.plot_combo_map()`
`b.display_best_combos()`
