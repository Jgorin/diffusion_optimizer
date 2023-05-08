
from scipy.optimize import differential_evolution
from numdifftools.core import Hessian
from numpy.linalg import eigvals
import pandas as pd
import numpy as np
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective_3HeParam import DiffusionObjective
import torch as torch
from diffusion_optimizer.utils.utils_3HeParam import forwardModelKinetics
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from conHe_Param import conHe_Param


def plot_results(params,dataset,objective):
    tot_moles = params[0]
    params = params[1:]

  
    data = forwardModelKinetics(params,(torch.tensor(dataset.TC), torch.tensor(dataset.thr),dataset.np_lnDaa,torch.tensor(dataset.Fi)),objective.lookup_table)
    print(data)
    T_plot = 10000/(dataset["TC"]+273.15)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(T_plot,dataset["ln(D/a^2)"],'bo',markersize=10)
    plt.plot(T_plot,data[-2],'ko',markersize=7)
    plt.ylabel("ln(D/a^2)")
    plt.xlabel("10000/T (K)")
    plt.subplot(3,1,2)



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

    plt.subplot(3,1,3)
    plt.plot(range(0,len(T_plot)),dataset.M,'b-o',markersize=5)
    plt.plot(range(0,len(T_plot)),tot_moles*Fi_MDD,'k-o',markersize=3)
    plt.xlabel("step number")
    plt.ylabel("Atoms Released at Each Step")
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

#nlc = NonlinearConstraint(con,lb =[0,-np.inf],ub = [np.inf,0])
nlc = NonlinearConstraint(conHe_Param,lb =[0,0],ub = [np.inf,np.inf])


# There is always one Ea (70,110)
# There are always num_domain # of LnD0aa (0,25)
# There are always num_domain-1 # of Fracs (0.001,1)
result = differential_evolution(
    objective, 
    [
        (2510948713- 3*12772561.78 ,  2510948713+ 3*12772561.78  ),
        (50, 150), 
        (-15, 25), 
        (-15,25),
        (-15,25),

        (0.00,1),
        (0.00,1),


 









    ], 
    disp=True, 
    tol=0.001, #4 zeros seems like a good number from testing. slow, but useful.
    popsize=25,
    recombination = 0.6,
    mutation = (0.6,1),
    maxiter = 100000000,
    constraints = nlc
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
