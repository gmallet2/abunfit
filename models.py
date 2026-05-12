"""
Implementation of models of supernovae and AGB, used for fitting abundancies ;
"""
import numpy as np
import matplotlib.pyplot as plt

AGB_DIR = "AGB/"
SNIA_DIR = "SNIa/"
SNCC_DIR = "SNcc/"


class Model() :
    """
    A simple standard model of supernovae/AGB, loaded from file.
    """
    def __init__(self,model_name,elements,dir):
        """
        Inputs :
            - model_name : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - dir : the direction where we find the model ;
        """
        self.model = None
        self.elements=elements
        self.model_name=model_name
        self.dir=dir
        self.y=np.zeros(len(self.elements))
        self.x=np.zeros(len(self.elements))

    def _integrate_over_mass(self) : 
        """
        (Internal) For some models, self.m is a list of mass, over which integrate the IMF. That's what this function does.
        """
        model, val = self.model_name.rsplit("_", 1)
        for n, el in enumerate(self.elements):
            try :
                data_mod = np.loadtxt(self.dir+model+"/"+val+"/"+str(el)+".txt")
            except(FileNotFoundError) :
                data_mod = np.zeros(data_mod.shape)
            if len(data_mod.shape)==0:
                yiel=np.array([data_mod])
            elif len(data_mod.shape)==1:
                yiel=np.array(data_mod)
            else:
                yiel=np.array(np.sum(data_mod.T, axis=1))   

            if ("Ro10" in self.model_name or "K10_AGB" in self.model_name): # special cases
                self.y[n] = np.sum(yiel*self.m**self.alpha) / np.sum(self.m**self.alpha)
            else:
                self.y[n] = np.trapezoid(yiel*np.power(self.m,self.alpha), x=self.m) / np.trapezoid(np.power(self.m,self.alpha), x=self.m)

    def _normalize(self,periodic_table) :
        """
        (Internal) Normalization of the abundancies ;
        Inputs :
            - periodic_table : a dictionnary of the periodic table, where each entry is : "H": {"Z": 1, "M": 1.008 ,"solar_val" : 1	2.59E+10}. Can be loaded for example using, in abunfit.py, 
        the Tool.build_periodic_table() function ;
        """
        for el in range(len(self.elements)) : 
            self.x[el]=self.y[el]/(periodic_table[self.elements[el]]["M"]*periodic_table[self.elements[el]]["solar_val"])
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

class SNIaModel(Model) :
    """
    Heritage of Model, used for SNIa models only ;
    """
    def __init__(self, model_name,elements,periodic_table,):
        """
        Inputs :
            - model_name : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - periodic_table : a dictionnary of the periodic table, where each entry is : "H": {"Z": 1, "M": 1.008 ,"solar_val" : 1	2.59E+10}. Can be loaded for example using, in abunfit.py, 
        the Tool.build_periodic_table() function ;
        """
        super().__init__(model_name,elements,SNIA_DIR)
        self.d_model=np.loadtxt(self.dir+model_name+".txt")
        self._manage_model(periodic_table)
        self._normalize(periodic_table)

    def _manage_model(self,periodic_table) :
        """
        (Internal) : Used to transform the raw data from the model into usables self.y and self.x results
        """
        for el in range(len(self.elements)) : 
            for i in range(len(self.d_model)) :
                if self.d_model[i,0] == float(periodic_table[self.elements[el]]["Z"]):
                    self.y[el] += self.d_model[i,2]


class SNccModel(Model) :
    """
    Heritage of Model, used for SNcc models only ;
    """
    def __init__(self, model_name,elements,periodic_table,alpha):
        """
        Inputs :
            - model_name : the name of the model : it has to ba available in the database.
            - elements : a list of the elements (H,He...) to be extracted and used (it's not necessary to use all the elements from the model of the database ;)
            - periodic_table : a dictionnary of the periodic table, where each entry is : "H": {"Z": 1, "M": 1.008 ,"solar_val" : 1	2.59E+10}. Can be loaded for example using, in abunfit.py, 
            - alpha : as, for SnCC models, we nedd to integrate the IMF, we need the alpha value for the Salpeter function.
        the Tool.build_periodic_table() function ;
        """
        super().__init__(model_name,elements,SNCC_DIR)
        self.alpha=alpha
        self._manage_model()
        self._normalize(periodic_table)

    def _manage_model(self) :
        """
        (Internal) : Used to transform the raw data from the model into usables self.y and self.x results
        """
        m = self.model_name.split("_")
        model,type = m[0],m[1]
        if model == "A22S03" :
            self.m = np.array([15,20,25,40])
        if model == "No06" :
            self.m=np.array([13,15,18,20,25,30,40])
        elif model == "Ch04" :
            self.m=np.array([13,15,20,25,30,35])
        elif model == "No13" :
            if type == "SNcc" :
                if self.model_name== "No13_SNcc_0" :
                    self.m=np.array([11,13,15,18,20,25,30,40,100,140])
                else :
                    self.m=np.array([13,15,18,20,25,30,40])
            elif type == "PISNe" :
                self.m=np.array([140,150,170,200,270,300])
            elif type == "SNe" :
                self.m=np.array([11,13,15,18,20,25,30,40,100,140,150,170,200,270,300])
            elif type == "HNe" :
                if self.model_name == "No13_HNe_0" :
                    self.m=np.array([20,25,30,40,100,140])
                else :
                    self.m=np.array([20,25,30,40])
        elif (self.model_name == "He0210_SNcc_0"):
            self.m=np.array([10,12,15,20,25,35,50,75,100])
        elif (self.model_name == "He0210_PISNe_0"):
            self.m=np.array([140,150,158,168,177,186,195,205,214,223,232,242,251,260])
        elif (self.model_name == "He0210_SNe_0"):
            self.m=np.array([10,12,15,20,25,35,50,75,100,140,150,158,168,177,186,195,205,214,223,232,242,251,260])
        elif (self.model_name == "Su16_N20"):
            self.m=np.array([12.25, 12.5, 12.75, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.2, 15.7, 15.8, 
            15.9, 16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17.0, 17.1, 17.3, 17.4, 17.5, 17.6, 17.7, 17.9, 18.0, 18.1, 18.2, 18.3, 18.4, 18.5, 18.7, 
            18.8, 18.9, 19.0, 19.1, 19.2, 19.3, 19.4, 19.7, 19.8, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.8, 21.0, 21.1, 21.2, 21.5, 21.6, 21.7, 25.2, 25.3, 25.4, 25.5, 
            25.6, 25.7, 25.8, 25.9, 26.0, 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7, 26.8, 26.9, 27.0, 27.1, 27.2, 27.3, 27.4, 29.0, 29.1, 29.2, 29.6, 60, 80, 100, 120])
        elif (self.model_name == "Su16_W18"):
            self.m=np.array([12.25, 12.5, 12.75, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.2, 15.7, 15.8, 
            16.0, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17.0, 17.1, 17.3, 17.4, 17.5, 17.6, 17.9, 18.1, 18.2, 18.3, 18.4, 18.5, 19.2, 19.3, 19.7, 19.8, 
            20.1, 20.2, 20.3, 20.4, 20.5, 20.8, 21.0, 21.1, 21.2, 21.5, 21.6, 25.2, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9, 26.0, 26.1, 26.2, 26.3, 26.4, 26.5, 27.0, 27.1, 
            27.2, 27.3, 60, 120])
        elif model == "Ro10" : 
            self.m=np.array([1.])
        self._integrate_over_mass()


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
        super().__init__(model_name,elements,AGB_DIR)
        self.alpha=alpha
        self._manage_model()
        self._normalize(periodic_table)
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
    a = SNIaModel('Ba06_DDTa',["O","Ne","Mg","Si","S","Ar","Ca","Cr","Mn","Fe","Ni"],Tools._build_periodic_table())
    a.plot_model()