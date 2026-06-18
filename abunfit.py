import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import json
import itertools
from models import Model
import corner
from MCMC import MCMC
import csv

AGB_DIR = "data/models/AGB/"
SNIA_DIR = "data/models/SNIa/"
SNCC_DIR = "data/models/SNcc/"
SOLAR_ABUN_TABLES     = "data/solar_tables/lodders09.txt"
DATA                  = "data/abundancies_results/Abell2199_2T.json"
PERIODIC_TABLE        = "data/periodic_table.json"
AVAILABLE_MODELS      = "data/models/models.csv"
ALPHA_SALPETER = -2.35

class Tools:
    @staticmethod
    def build_periodic_table() :
        """
        From the file "periodic_table.json" and a file of solar abundancies ("solar_table/lodders09.txt"), returns a periodic table as a dictionnary.
        Output :
            - periodic_table : a dictionnary, where an entry is :   "H": {"Z": 1, "M": 1.008 ,"solar_val" : 2.59E+10},
        """
        with open(PERIODIC_TABLE, "r") as f:
            periodic_table = json.load(f)
        abun_file = np.loadtxt(SOLAR_ABUN_TABLES)
        for info in periodic_table.values():
            try:
                info["solar_val"] = abun_file[info["Z"] - 1][1]
            except (IndexError, TypeError):
                info["solar_val"] = None
        return periodic_table
    @staticmethod
    def plot_models_compar(L_models,elements,alpha=-2.35) :
        """
        Compare models with publication-quality error bars.

        Inputs : 
            - L_abund (list)  : List of models names ["Le25-A22S03_0","Le18_300-0-c3"]
            - elements (list) : List of elements (["Ca","Ni","Fe"])
        """
        # Style
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.dpi": 150
        })
        models = {}
        periodic_table = Tools.build_periodic_table()

        with open(AVAILABLE_MODELS, "r", newline="", encoding="utf-8") as f:
            l_available_models = list(csv.reader(f))
        for model_name in L_models : 
            for i in l_available_models :
                if i[0]==model_name :
                    if i[1] == "AGB" :
                        models[model_name] = Model(model_name, elements, AGB_DIR,periodic_table, alpha)
                    elif i[1] == "SN1A" :
                        models[model_name] = Model(model_name, elements, SNIA_DIR,periodic_table)
                    elif i[1] == "SNCC" :
                        models[model_name] = Model(model_name, elements, SNCC_DIR,periodic_table, alpha)
        x = np.arange(len(elements))
        _, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.tab10.colors
        n_models = len(models)
        width = 0.15

        for i, (model_name, data) in enumerate(models.items()):
            offset = (i - (n_models -1)/2) * width
            ax.bar(x + offset,data.x,width=width,color=colors[i % len(colors)],label=model_name,alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(elements)

        ax.set_xlabel("Element")
        ax.set_ylabel("Abundance")
        ax.set_title("Comparison of abundance models", pad=15)

        ax.grid(True, linestyle='--', alpha=0.3)

        leg = ax.legend(loc='upper left',bbox_to_anchor=(0.09, 1),frameon=True,fancybox=True,shadow=False,borderaxespad=0.)

        leg.get_frame().set_alpha(0.95)
        print("TRY")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_abundance_compar(L_abund):
        """
        Compare abundance models with publication-quality error bars.

        Inputs : 
            - L_abund (list) List of JSON files containing abundances : {"Si": [value, error], ...}
        """
        # Style
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.dpi": 150
        })
        models = {}

        for file in L_abund:
            with open(file, "r") as f:
                models[file.split("/")[-1].replace(".json", "")] = json.load(f)

        elements = list(next(iter(models.values())).keys())
        x = np.arange(len(elements))
        _, ax = plt.subplots(figsize=(12, 7))

        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        colors = plt.cm.tab10.colors

        n_models = len(models)
        width = 0.15

        for i, (model_name, data) in enumerate(models.items()):

            values = [data[el][0] for el in elements]
            errors = [data[el][1] for el in elements]

            offset = (i - (n_models -1)/2) * width
            ax.errorbar(
                x + offset,
                values,
                yerr=errors,
                fmt=markers[i % len(markers)],
                markersize=8,
                color=colors[i % len(colors)],
                capsize=5,
                capthick=1.5,
                elinewidth=2,
                linewidth=2,
                label=model_name
            )
        ax.set_xticks(x)
        ax.set_xticklabels(elements)

        ax.set_xlabel("Element")
        ax.set_ylabel("Abundance")
        ax.set_title("Comparison of abundance models", pad=15)

        ax.grid(True, linestyle='--', alpha=0.3)

        leg = ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.09, 1),
            frameon=True,
            fancybox=True,
            shadow=False,
            borderaxespad=0.
        )

        leg.get_frame().set_alpha(0.95)

        plt.tight_layout()
        plt.show()


class Fit:
    """
    This class contains all the fit methods, so we can choose any of them to fit the abundances.
    """
    @staticmethod
    def fitfunc(parameters, modmatrix,data) :
        errors = np.where(data[:, 1] > 0, data[:, 1], 1.0)
        return (data[:,0] - np.dot(parameters,modmatrix))/errors




class AbunFit:
    """
    Class used to perform a simple fit : we fit abundancies of a set of elements, using a combination of chosen supernovae/AGB models.
    """
    def __init__(self, f_input , l_models , alpha  = ALPHA_SALPETER):
        """
        Inputs :
            - f_input (str) : a .json abundances file, with the classical dictionnary   {"Si": [0.768, 0.0964], ...} : it's the abundances we want to fit ;
            - l_models (list) : a list of the models we want to use for the fit. Those models have to be available in the database (see the list_models.txt files). Example : ['Le18_300-0-c3', 'A22S03_0']
            - alpha (float) : if we use SnCC and/or AGB models, we shall integrate IMFs, using salpeter models : we need the alpha of Salpeter
        """
        self.data = None

        with open(AVAILABLE_MODELS, "r", newline="", encoding="utf-8") as f:
            self.l_available_models = list(csv.reader(f))

        with open(f_input, "r") as f:
            self.data = json.load(f)

        self.periodic_table = Tools.build_periodic_table()
        self.elements = list(self.data.keys())
        self.alpha    = alpha
        self.models   = {}

        for name in l_models:
            if not self._model_from_name(name) :
                print(f"----- WARNING : model '{name}' can't be found -----")

        # Results fields
        self.fit_results            = None
        self.fit_results_abundances = None
        self.chi2                   = None
        self.reduced_chi2           = None
        self.flat_samples           = None   # chaînes MCMC post burn-in
        self.mcmc_median            = None
        self.mcmc_lo                = None   # borne basse 1σ
        self.mcmc_hi                = None   # borne haute 1σ

    def _model_from_name(self,model_name) :
        """
        (Internal) Load a model from the name of the model : it checks if the model is available, and its type (SNIA, SNCC, AGB).
        Inputs :
            - model_name (str) : the name of the model to load ;
        Outputs :
            - (bool) : True if the model has been found, or False ;
        """
        for i in self.l_available_models :
            if i[0]==model_name :
                if i[1] == "AGB" :
                    self.models[model_name] = Model(model_name, self.elements, AGB_DIR,self.periodic_table, self.alpha)
                elif i[1] == "SN1A" :
                    self.models[model_name] = Model(model_name, self.elements, SNIA_DIR,self.periodic_table)
                elif i[1] == "SNCC" :
                    self.models[model_name] = Model(model_name, self.elements,SNCC_DIR, self.periodic_table, self.alpha)
                return True
        return False
    
    def fit(self, verbose = True,fitfunc = Fit.fitfunc):
        """
        Perform a fit of the abundancies with the given models of supernovae/AGB.
        Inputs :
            - verbose (bool) : if True, it'll display lots of informations during the fit ;
            - fitfunc (Fit.function) : the fit function we want to use : see the class Fit.
        """
        self.modmatrix = np.vstack([self.models[i].x for i in self.models])
        x0     = np.ones(self.modmatrix.shape[0])
        l_data = np.array([self.data[el] for el in self.elements], dtype=float)

        result = optimization.least_squares(fitfunc, x0,args=(self.modmatrix, l_data),bounds=(0, np.inf))
        self.fit_results            = result.x
        self.fit_results_abundances = np.dot(self.fit_results, self.modmatrix)
        self._data_array            = l_data 

        errors = np.where(l_data[:, 1] > 0, l_data[:, 1], 1.0)
        self.chi2 = np.sum(
            ((l_data[:, 0] - self.fit_results_abundances) / errors) ** 2
        )
        dof = max(1, len(self.elements)-len(self.fit_results))
        self.reduced_chi2 = self.chi2 / dof

        if verbose:
            print("\n----- FIT least squares -----")
            for name, c in zip(self.models, self.fit_results):
                print(f"  {name:30s} : {c:.6g}")
            print(f"  chi2         = {self.chi2:.4f}")
            print(f"  chi2 réduit  = {self.reduced_chi2:.4f}")

    def run_mcmc(self,n_walkers = 64,n_steps = 3000,n_burn = 500, perturbation = 1e-3, ci = 68.27,progress = True):
        """
        MCMC used to estimate, after the fit, the errors around the coefficients ;
        Inputs :
            n_walkers   (int) : number of emcee walkers (has to be > 2*n_models)
            n_steps      : number of steps per walker
            n_burn       : burn-in (first steps to ignore)
            perturbation : initial dispersion qround the best-fit
            ci           : required confidence intervall (%)
            progress     : do we want a progress bar
        """
        if self.fit_results is None:
            raise RuntimeError("Call fit() and then you can use run_mcmc().")
        
        print(f"\n----- MCMC  ({n_walkers} walkers × {n_steps} steps, burn-in {n_burn}) -----")

        self.flat_samples, self.sampler = MCMC.run(self.modmatrix,self._data_array,self.fit_results,
            n_walkers = n_walkers,n_steps = n_steps,n_burn = n_burn, perturbation = perturbation,progress = progress)

        self.mcmc_median = np.percentile(self.flat_samples, 50, axis=0)
        self.mcmc_lo     = np.percentile(self.flat_samples, (100.0- ci) / 2.0, axis=0)
        self.mcmc_hi     = np.percentile(self.flat_samples, 100 - (100.0- ci)/2.0 , axis=0)

        print(f"\n  Results for the MCMC ({ci:.2f}% CI) :")
        for i, name in enumerate(self.models):
            med = self.mcmc_median[i]
            lo  = med - self.mcmc_lo[i]
            hi  = self.mcmc_hi[i] - med
            print(f"  {name:30s} : {med:.4g}  -{lo:.3g}  +{hi:.3g}")

    def plot_fit(self, show_mcmc: bool = True):
        """
        Result plot of the fit (bar charts)
        Inputs :
            - show_mcmc (bool) : Do we display the errors, calculated with the MCMC ?
        """
        xaxis  = np.arange(len(self.elements))
        data   = np.array([self.data[el][0] for el in self.elements])
        err    = np.array([self.data[el][1] for el in self.elements])
        colors = plt.cm.tab10(np.arange(self.modmatrix.shape[0]))

        _, ax = plt.subplots(figsize=(10, 5))
        bottom     = np.zeros(len(self.elements))
        bottom_lo  = np.zeros(len(self.elements))
        bottom_hi  = np.zeros(len(self.elements))

        coeffs = (self.mcmc_median
                  if (show_mcmc and self.mcmc_median is not None)
                  else self.fit_results)

        for i, name in enumerate(self.models):
            contrib = coeffs[i] * self.modmatrix[i]
            ax.bar(xaxis, contrib, bottom=bottom, color=colors[i],
                   label=name, alpha=0.8)

            if show_mcmc and self.mcmc_lo is not None:
                contrib_lo = self.mcmc_lo[i] * self.modmatrix[i]
                contrib_hi = self.mcmc_hi[i] * self.modmatrix[i]
                total_lo   = bottom_lo + contrib_lo
                total_hi   = bottom_hi + contrib_hi
                ax.errorbar(xaxis,bottom + contrib,yerr=[bottom + contrib - total_lo,total_hi - bottom - contrib],
                    fmt="none",ecolor=colors[i],elinewidth=1.2,capsize=3,alpha=0.7)
                bottom_lo += contrib_lo
                bottom_hi += contrib_hi
            bottom += contrib

        ax.errorbar(xaxis, data, yerr=err, fmt="ko", label="Data", zorder=10)
        ax.set_xticks(xaxis)
        ax.set_xticklabels(self.elements)
        ax.set_ylabel("Abundance (solar)")
        ax.set_xlabel("Element")
        ax.legend(fontsize=9)
        chi2_str = (f"$\\chi^2$/d.o.f. = {self.chi2:.1f} / "
                    f"{len(self.elements) - len(self.fit_results)}")
        ax.set_title(f"Abundance Fit  —  {chi2_str}")
        plt.tight_layout()
        plt.show()

    def plot_corner(self, **corner_kwargs):
        """
        Corner diagramm, only used after we ran a MCMC obviously ;
        Inputs :
            - n_show (int)
        """
        if self.flat_samples is None:
            raise RuntimeError("call run_mcmc() first ;")

        labels = list(self.models.keys())
        defaults = dict(
            labels = labels,show_titles = True, title_fmt = ".3g",title_kwargs = {"fontsize": 9},label_kwargs = {"fontsize": 9},
            quantiles = [0.159, 0.5, 0.841], truths = self.fit_results, truth_color = "tomato", smooth = 1.0)
        defaults.update(corner_kwargs)
        corner.corner(self.flat_samples, **defaults)
        plt.suptitle("MCMC posterior distributions", y=1.01, fontsize=11)
        plt.show()

class MultiFit:
    """
    Class used to perform multiple fits, over many combinations of models : then, we can compare thoses combinations and find the best ones.
    """
    def __init__(self, l_models, data_dir = DATA):
        self.l_models = []
        with open(AVAILABLE_MODELS, "r", newline="", encoding="utf-8") as f:
            l_available_models = list(csv.reader(f))
        for model in l_models :
            if model in ["AGB","SN1A","SNCC","DD","DETONATION","DEFLAGRATION","HY","D2","D6","SMCH","NMCH"] : #To add : possibility of choosing only DD, Deflagration etc...
                l = [] 
                for j in l_available_models :
                    if model in j:
                        l.append(j[0])
                self.l_models.append(l)
            else :
                self.l_models.append(model)

        self.chi2_results = []
        self.data_dir = data_dir
        print(self.l_models)
    def _fit_one(self,combo,alpha = ALPHA_SALPETER) :
        """
        (Internal) Used to fit one specifit combo
        Inputs :
            - combo (list) : a list of models names (combo)
            - alpha : salpeter mass function alpha
        """
        a = AbunFit(self.data_dir, list(combo), alpha=alpha)
        a.fit(verbose=False)
        self.chi2_results.append([list(combo) + [str(alpha)], a.reduced_chi2, a.fit_results])

    def multifit(self, alpha  = []):
        """
        This is the fit function, which perform a fit over all the possible conbinations of the models of the object.
        Inputs :
            - alpha (list) : we can make a list of alphas, allowing to test combinations of models with different alphas.
        """
        for combo in itertools.product(*self.l_models) :
            print(combo)
            try:
                if len(alpha) > 0:
                    for a_val in alpha:
                        self._fit_one(combo,a_val)
                else:
                    self._fit_one(combo)
            except FileNotFoundError :
                print("Missing file — ignored ;")
            except Exception as e:
                print(f"Error : {e} — ignored ;")

        if not self.chi2_results:
            print("No results ;")
            return
        
    def plot_combo_map(self) :
        chi2_vals, fractions = [], []
        for _, chi2, params in self.chi2_results:
            total = np.sum(params)
            chi2_vals.append(chi2)
            fractions.append(params[0]/total if total > 0 else np.nan)

        plt.figure()
        sc = plt.scatter(chi2_vals, fractions, c=chi2_vals,
                         cmap="viridis", alpha=0.8)
        plt.colorbar(sc, label="Reduced chi_2")
        plt.xlabel("Reduced chi_2")
        plt.ylabel("Fraction SNIa  c₀ / Σcᵢ")
        plt.title("Results multifit")
        plt.xscale("log")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        
    def display_best_combos(self,nb_best_combos = 10) :
        chi2_vals, l_names, l_params = [], [], []
        for names, chi2, params in self.chi2_results:
            l_names.append(names)
            chi2_vals.append(chi2)
            l_params.append(params)
        print("\n--- Top {indices} ---")
        for idx in np.argsort(np.array(chi2_vals))[:nb_best_combos]:
            print(f"  {l_names[idx]}  →  chi2r = {np.array(chi2_vals)[idx]:.4f}")
            print(l_params[idx])

if __name__ == "__main__":
    #Tools.plot_abundance_compar([DATA,"data/abundancies_results/Abell2199_bvvapec.json","data/abundancies_results/Abell2199_bvvgadem.json"])
    #Tools.plot_models_compar(['Ch04_0',"Ch04_1E-3","No06_0.001"],["Si","S","Ar","Ca","Fe","Cr","Mn","Ni"])
    #Tools.plot_models_compar(['Le25_A22S03_0',"Ch04_1E-3","No06_0.001"],["Si","S","Ar","Ca","Fe","Cr","Mn","Ni"])
    #a = AbunFit(DATA,['Le25_A22S03_0',"Le18_300-0-c3"])
    #a = AbunFit(DATA,['Ch04_0',"Le18_300-0-c3"])
    b = MultiFit(["SN1A","SNCC"],data_dir=DATA)
    b.multifit()
    b.plot_combo_map()
    b.display_best_combos()
    # 1. Fit moindres carrés (rapide, donne le point de départ)
    #a.fit()

    # 2. MCMC 
    #a.run_mcmc(
    #    n_walkers    = 64,     # > 2 × n_modèles
    #n_steps      = 3000,   # augmenter si tau est grand
    #    n_burn       = 500,    # burn-in à ignorer
    #    perturbation = 1e-3,   # dispersion initiale autour du best-fit
    #   ci           = 68.27,  # intervalle de confiance (1σ)
    #)

    # 3. Visualisations
    #a.plot_fit()                     # barres empilées avec erreurs MCMC
    #a.plot_corner()                  # corrélations entre paramètres