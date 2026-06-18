"""
Implementation of models of supernovae and AGB, used for fitting abundancies ;
"""
import numpy as np
import matplotlib.pyplot as plt
import json

class Model() :
    """
    A simple standard model of supernovae/AGB, loaded from file.
    """
    def __init__(self,model,elements,dir,periodic_table,alpha=-2.35):
        """
        Inputs :
            - model : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - dir : the direction where we find the model ;
            - periodic_table : the periodic table as a dictionnary (built using for example Tools.build_periodic_table() in abunfit.py) ;
            - alpha : the alpha for the Salpeter IMF, used only for SNCC and AGB (not SN1A) ;
        """
        self.alpha=alpha
        self.model = model
        self.periodic_table = periodic_table
        self.elements=elements
        self.y=np.zeros(len(self.elements))
        self.x=np.zeros(len(self.elements))
        with open(dir+self.model+".json", "r") as f:   
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

            yiel=np.array(data_mod)
  
            if self.m == None :
                self.y[n] = data_mod[0]
            else : 
                if len(self.m)==1 :
                    self.y[n] = np.sum(yiel*self.m[0]**self.alpha) / np.sum(self.m[0]**self.alpha)
                else:
                    self.y[n] = np.trapezoid(yiel*np.power(self.m,self.alpha), x=self.m)/np.trapezoid(np.power(self.m,self.alpha), x=self.m)
            
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
        plt.ylabel("Abundance for the model "+self.model)
        plt.xlabel("Element")
        plt.title("Abundance Fit for the model "+self.model)
        plt.tight_layout()
        plt.show()
