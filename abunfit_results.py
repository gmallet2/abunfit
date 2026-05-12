import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
RESULTS_DIR = "results/"

class AbunFitResults() :
    def __init__(self,name):
        self.name=name
        self.data = None # The entry data... [entry filename]
        self.results = None #Structure : [[model, fit_results, abundancies_results, chi_2,chi_2_reduced],...]
    def results_from_fit(self,data_filename,model,fit_results,abundancies_result,chi_2,chi_2_reduced) :
        self.results = [model,fit_results,abundancies_result,chi_2,chi_2_reduced]
        self.data = [data_filename]
    def save_results(self,f_name) :
        np.save(RESULTS_DIR+f_name, [self.data, self.results])
    def load_results(self,f_name) :
        self.data,self.results = np.load(f_name, allow_pickle=True)

    def plot_1(self,model):
        for i in self.results : 
            same = True
            for j in range(len(i[0])) :
                if i[0][j] != model[0][j] :
                    same = False
            if same :
                xaxis = np.arange(len(i[2]))

                data = np.array([self.data[el][0] for el in self.elements])
                err = np.array([self.data[el][1] for el in self.elements])

                colors = plt.cm.tab10(np.arange(self.modmatrix.shape[0]))
                bottom = np.zeros(len(self.elements))

                for i, name in enumerate(self.models.keys()):
                    contrib = self.fit_results[i] * self.modmatrix[i]
                    plt.bar(xaxis,contrib,bottom=bottom,color=colors[i],label=name,alpha=0.8)
                    bottom += contrib
                #plt.annotate(r'$\chi^2$/d.o.f. = '+str(np.round(float(self.chi2),1))+"/"+str((len(self.elements)- self.modmatrix.shape[0])), xy= ,horizontalalignment='right', color='black', fontsize=14)

                plt.errorbar(xaxis,data,yerr=err,fmt='ko',label="Data",zorder=10)
                plt.xticks(xaxis, self.elements)
                plt.ylabel("Abundance")
                plt.xlabel("Element")
                plt.legend()
                plt.title("Abundance Fit")
                plt.tight_layout()
                plt.show()
    def plot_all(self) :
