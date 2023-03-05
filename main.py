
from scipy.optimize import differential_evolution
from numdifftools.core import Hessian
from numpy.linalg import eigvals
import pandas as pd
import numpy as np
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective import DiffusionObjective
import torch as torch
from diffusion_optimizer.utils.utils import forwardModelKinetics
import matplotlib.pyplot as plt

def plot_results(params,dataset,objective):
    
    #params = torch.tensor(optimizer.sample_manager._elites[0]._param)
    #params = torch.tensor([1.06393577e+02, 2.12389770e+01, 1.86098740e+01, 8.15511327e+00, 9.69495713e-01, 2.95042649e-02])
    data = forwardModelKinetics(params,(torch.tensor(dataset.TC), torch.tensor(dataset.thr),dataset.np_lnDaa,torch.tensor(dataset.Fi)),objective.lookup_table)
  
    T_plot = 10000/(dataset["TC"]+273.15)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(T_plot,dataset["ln(D/a^2)"],'bo',markersize=10)
    plt.plot(T_plot,data[-2],'ko',markersize=7)
    plt.ylabel("ln(D/a^2)")
    plt.xlabel("10000/T (K)")
    plt.subplot(2,1,2)



    Fi_MDD = np.array(data[-3])
    temp = Fi_MDD[1:]-Fi_MDD[0:-1]
    Fi_MDD =np.insert(temp,0,Fi_MDD[0])

    Fi = np.array(dataset.Fi) 
    temp = Fi[1:]-Fi[0:-1]
    Fi = np.insert(temp,0,Fi[0])
    plt.plot(range(0,len(T_plot)),Fi,'b-o',markersize=5)
    plt.plot(range(0,len(T_plot)),Fi_MDD,'k-o',markersize=3)
    plt.xlabel("step number")
    plt.ylabel("Fractional Release (%)")
    plt.show()
def get_errors(objective, result, norm_dist=1.96, step=1e-5):
    hessian = Hessian(objective, step=step, full_output=False)(result)
    # check if the matrix is positive definite
    if np.all(eigvals(hessian) > 0):
        # get the covariance matrix
        cov = np.linalg.inv(hessian)
        # get the standard errors of the parameters
        se = np.sqrt(np.diag(cov))
        return np.multiply(se, norm_dist)
    else:
        print("Hessian matrix is not positive definite.")
        return None

dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
objective = DiffusionObjective(dataset, [],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")


result = differential_evolution(
    objective, 
    [
        (70, 110), 
        (0, 25), 
        (0, 25), 
        (0, 25), 
        (0, 25),
        (0,25),
        (0,25),
        (0.001, 1), 
        (0.001, 1),
        (0.001,1),
        (0.001,1),
        (0.001,1)
    ], 
    disp=True, 
    tol=0.0000001,
    updating="deferred",
    init="latinhypercube",
    maxiter=10000
)

print(result.x)

plot_results(torch.tensor(result.x),dataset,objective)
norm_dist_error = get_errors(objective, result.x)
print("\n\n")
print(["{0:0.6f}".format(i) for i in norm_dist_error])
print(["{0:0.6f}".format(i) for i in result.x])
print("\n\n")
print(norm_dist_error)
print(result.x)
