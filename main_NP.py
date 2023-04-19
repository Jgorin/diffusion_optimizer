
from scipy.optimize import differential_evolution
from numdifftools.core import Hessian
from numpy.linalg import eigvals
import pandas as pd
import numpy as np
from diffusion_optimizer.neighborhood.dataset_np import Dataset
from diffusion_optimizer.diffusion_objective_np import DiffusionObjective
import torch as torch
from diffusion_optimizer.utils.utils import forwardModelKinetics
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint
from con import con
from jax import jit
from jax import grad
import random

def plot_results(params,dataset,objective):
    
    #params = torch.tensor([92.8597496,19.3075859,16.7392942,13.448396,12.0199871,7.57389364,0.928117128,0.012069709,0.0364852194,0.0225384812])

    data = forwardModelKinetics(params,(np.array(dataset.TC), np.array(dataset.thr),dataset.np_lnDaa,np.array(dataset.Fi)),objective.lookup_table)
   
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
objective = DiffusionObjective(dataset, [],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")

#nlc = NonlinearConstraint(con,lb =[0,-np.inf],ub = [np.inf,0])
nlc = NonlinearConstraint(con,lb =[0,0],ub = [np.inf,np.inf])

grad_fn = grad(objective)
low_bounds = np.array([ 60,0,0,0,0.001,0.001])
upp_bounds = np.array([150,25,25,25,1,1])


# start by generating random initial guess
initial_guess = np.array([random.random() for _ in range(len(low_bounds))])
initial_guess = initial_guess*(upp_bounds-low_bounds) +low_bounds


for i in range(10000):
    gradient = grad_fn(initial_guess)
    #gradient = np.array([-0.001*gradient[0]])
    step = -0.05*np.array(gradient)
            
    initial_guess+=step
    print(objective(initial_guess))



# There is always one Ea (70,110)
# There are always num_domain # of LnD0aa (0,25)
# # There are always num_domain-1 # of Fracs (0.001,1)
# result = differential_evolution(
#     objective, 
#     [

#         (60, 150), 
#         (0, 25), 
#         (0,25),
#         (0,25),
#         (0.001,1),
#         (0.001,1),


#     ], 
#     disp=True, 
#     tol=0.01,

    
#     maxiter=100000000000000000,
#     constraints = nlc
# )

# print(result.x)

# plot_results(torch.tensor(result.x),dataset,objective)
# norm_dist_error = get_errors(objective, result.x)
# print("\n\n")
# print(["{0:0.6f}".format(i) for i in norm_dist_error])
# print(["{0:0.6f}".format(i) for i in result.x])
# print("\n\n")
# print(norm_dist_error)
# print(result.x)
