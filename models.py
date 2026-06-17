"""
Implementation of models of supernovae and AGB, used for fitting abundancies ;
"""
import numpy as np
import matplotlib.pyplot as plt
import json

AGB_DIR = "data/models/AGB/"
SNIA_DIR = "data/models/SNIa/"
SNCC_DIR = "data/models/SNcc/"


class Model() :
    """
    A simple standard model of supernovae/AGB, loaded from file.
    """
    def __init__(self,model_name,elements,dir,periodic_table,alpha=-2.35):
        """
        Inputs :
            - model_name : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - dir : the direction where we find the model ;
        """
        self.alpha=alpha
        self.model = None
        self.periodic_table = periodic_table
        self.elements=elements
        self.model_name=model_name
        self.dir=dir
        self.y=np.zeros(len(self.elements))
        self.x=np.zeros(len(self.elements))
        with open(dir+model_name+".json", "r") as f:   
            self.data = json.load(f)

        self._integrate_over_mass()
        self._normalize()

    def _integrate_over_mass(self) : 
        """
        (Internal) For some models, self.m is a list of mass, over which integrate the IMF. That's what this function does.
        """
        self.m = self.data["Mass"]

        for n, el in enumerate(self.elements):
            if self.m == None : 
                data_mod = np.zeros_like(np.array([0]),dtype=float)
            else : 
                data_mod = np.zeros_like(np.array(self.m),dtype=float)
            for i in list(self.data.keys()) : 
                j = i.split("_")
                if j[0] == el :
                    data_mod += np.array(self.data[i])
            if len(data_mod.shape)==0:
                yiel=np.array([data_mod])
            elif len(data_mod.shape)==1:
                yiel=np.array(data_mod)
            else:
                yiel=np.array(np.sum(data_mod.T, axis=1))  
                 
            if self.m == None :
                self.y[n] = data_mod[0]
            else : 
                if ("Ro10" in self.model_name or "K10_AGB" in self.model_name): # special cases
                    self.y[n] = np.sum(yiel*self.m**self.alpha) / np.sum(self.m**self.alpha)
                else:
                    self.y[n] = np.trapezoid(yiel*np.power(self.m,self.alpha), x=self.m) / np.trapezoid(np.power(self.m,self.alpha), x=self.m)
            
    def _normalize(self) :
        """
        (Internal) Normalization of the abundancies ;
        """
        for el in range(len(self.elements)) : 
            self.x[el]=self.y[el]/(self.periodic_table[self.elements[el]]["M"]*self.periodic_table[self.elements[el]]["solar_val"])

    def plot_model(self) :
        """
        Used to plot the abundances values of a model ;
        """
        color = plt.cm.tab10(np.arange(1))[0]
        bottom = np.zeros(len(self.elements))
        xaxis = np.arange(len(self.elements))
        contrib = self.x
        plt.bar(xaxis,contrib,bottom=bottom,color=color,alpha=0.8)
        plt.xticks(xaxis, self.elements)
        plt.ylabel("Abundance for the model "+self.model_name)
        plt.xlabel("Element")
        plt.title("Abundance Fit for the model "+self.model_name)
        plt.tight_layout()
        plt.show()
        

class AGBModel(Model) :
    """
    Heritage of Model, used for AGB models only ;
    """
    def __init__(self, model_name,elements,periodic_table,alpha):
        """
        Inputs :
            - model_name : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - periodic_table : a dictionnary of the periodic table, where each entry is : "H": {"Z": 1, "M": 1.008 ,"solar_val" : 1	2.59E+10}. Can be loaded for example using, in abunfit.py, 
            - alpha : as, for AGB models, we need to integrate the IMF, we need the alpha value for the Salpeter function.
        the Tool.build_periodic_table() function ;
        """
        self.alpha = alpha
        super().__init__(model_name,elements,AGB_DIR,periodic_table)
    def _manage_model(self) :
        """
        (Internal) : Used to transform the raw data from the model into usables self.y and self.x results
        """
        if "K10_AGB" in self.model_name:
            self.m=np.array([1])
        else:
            self.m=np.array([0.9,1.0,1.25,1.5,1.75,1.9,2.0,2.25,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5])
        self._integrate_over_mass()

if __name__=="__main__" :
    from abunfit import Tools
