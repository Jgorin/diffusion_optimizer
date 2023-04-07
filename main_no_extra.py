
from scipy.optimize import differential_evolution
from numdifftools.core import Hessian
from numpy.linalg import eigvals
import pandas as pd
import numpy as np
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective_no_extra import DiffusionObjective_no_extra
import torch as torch
from diffusion_optimizer.utils.utils_no_extra import forwardModelKinetics
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from con import con


def plot_results(params,dataset,objective):
    #params = torch.tensor([98.4302288,17.1535987,17.1473468,16.0422343,14.6013811,11.6164845,6.53169108,0.218205613,0.106816304,0.586360973,0.0438093845,0.0261573749])


    data = forwardModelKinetics(params,(torch.tensor(dataset.TC), torch.tensor(dataset.thr),dataset.np_lnDaa,torch.tensor(dataset.Fi)),objective.lookup_table)
    print(data)
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
objective = DiffusionObjective_no_extra(dataset, [1,2,3,4,5],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")
# used 1-7 for km95-28, used 1-5 for km95-6, and 1-6 for km96-20db,1-7 for km95-15, 1-5 for km95-6-db, 1-5 for km95-15-De
#nlc = NonlinearConstraint(con,lb =[0,-np.inf],ub = [np.inf,0]
nlc = NonlinearConstraint(con,lb =[0,0],ub = [np.inf,np.inf])


# There is always one Ea (70,110)
# There are always num_domain # of LnD0aa (0,25)
# There are always num_domain-1 # of Fracs (0.001,1)
result = differential_evolution(
    objective, 
    [
        (0,110), 
        (0,25),
        (0,25),
        (0,25),
    


        (0.001,1),
        (0.001,1)










    ], 
    disp=True, 
    tol=0.00001,
    maxiter=100000000,
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
