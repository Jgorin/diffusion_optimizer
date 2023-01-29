import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plotModelResults(fwdModel,expData):
    # Calculate the temp in 10000/TK (Standard plotting units for this field)
    expData["10000/TK"] = 10000/(expData["TC"]+273.15)
    fwdModel["10000/TK"] = 10000/(fwdModel["TC"]+273.15)
    fwdModel["MDDFi_Cum"] = np.cumsum(fwdModel["Fi_MDD"])
    expData["Fi_Cum"] = np.cumsum(expData["Fi"])
    
    plt.figure();
    plt.subplot(1,2,1)
    sns.scatterplot(data = expData, x = "10000/TK", y = "ln(D/a^2)")
    sns.scatterplot(data = fwdModel, x = "10000/TK",y="ln(D/a^2)_MDD")
    plt.legend(["Experiment","Model"])
    plt.subplot(1,2,2)
    sns.scatterplot(data = expData, x =expData.index, y = "Fi")
    sns.scatterplot(data = fwdModel, x =fwdModel.index, y="Fi_MDD")
    plt.legend(["Experiment","Model"])