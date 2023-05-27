
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
from diffusion_optimizer.utils.utils_3HeParam import calc_arrhenius



def plot_results(params,dataset,objective, reference_law = []):
    # Params is a vector X of the input parameters
    # dataset is the dataset class with your data
    # objective is the objective you used
    # reference_law is an array with values [Ea, lnd0aa]

    R = 0.008314
    tot_moles = params[0]
    params = params[1:]
        # Infer the number of domains from input
    if len(params) <= 3:
        ndom = 1
    else:
        ndom = (len(params))//2

   


    # Reconstruct the time-added and temp-added inputs
    time = torch.tensor(dataset.thr*3600)
    TC = torch.tensor(dataset.TC)
    tsec = torch.cat([objective.time_add,time])
    TC = torch.cat([objective.temp_add,TC])



    data = calc_arrhenius(params,objective.lookup_table,tsec,TC)
    print(data)

    T_plot = 10000/(dataset["TC"]+273.15)
    if len(reference_law) == 0:
        n_plots = 3
    else:
        n_plots = 4

    fracs = params[ndom+1:]
    fracs = torch.concat((fracs,1-torch.sum(fracs,axis=0,keepdim=True)),axis=-1)

    fig,axes = plt.subplots(ncols = 2, nrows = 2,layout = "constrained",figsize=(10,10))


    errors_for_plot = np.array(pd.concat([dataset["ln(D/a^2)"]-dataset["ln(D/a^2)-del"],dataset["ln(D/a^2)+del"]-dataset["ln(D/a^2)"]],axis=1).T)        

    #plt.subplot(n_plots,1,1)
    frac_weights = (fracs-torch.min(fracs)/(torch.max(fracs)-torch.min(fracs)))*1.5+1.3
    for i in range(ndom):
        D = np.log(np.exp(params[i+1])*np.exp((-params[0])/(R*(dataset["TC"]+273.15))))
        axes[0,0].plot(np.linspace(min(T_plot),max(T_plot),1000), np.linspace(max(D),min(D),1000), '--k',linewidth = frac_weights[i],zorder=0)
   
    axes[0,0].errorbar(T_plot,dataset["ln(D/a^2)"],yerr= errors_for_plot,fmt = 'bo',markersize=10,zorder=5)
    axes[0,0].plot(T_plot,data[1],'ko',markersize=7,zorder = 10)
    
    
    #Normalize Fractions for plotting weights
   
    if n_plots == 4:
        ref = np.log(np.exp(reference_law[1])*np.exp((-reference_law[0])/(R*(dataset["TC"]+273.15))))
    #     breakpoint()
    #     plt.plot(np.linspace(min(T_plot),max(T_plot),1000), np.linspace(max(ref),min(ref),1000), '--k')
    axes[0,0].set_ylabel("ln(D/a^2)")
    axes[0,0].set_xlabel("10000/T (K)")
    #axes[0].xlim([min(T_plot)-2,max(T_plot)+2])
    #axes[0].ylim([min(dataset["ln(D/a^2)"]-2),min(dataset["ln(D/a^2)"]+2)])
    axes[0,0].set_box_aspect(1)




    #plt.subplot(n_plots,1,2)
    Fi_MDD = np.array(data[0])
    temp = Fi_MDD[1:]-Fi_MDD[0:-1]
    Fi_MDD =np.insert(temp,0,Fi_MDD[0])
    Fi = np.array(dataset.Fi) 
    temp = Fi[1:]-Fi[0:-1]
    Fi = np.insert(temp,0,Fi[0])


    axes[1,0].errorbar(range(0,len(T_plot)),Fi,yerr = dataset["Fi uncertainty"],fmt ='b-o',markersize=5,zorder=5)
    axes[1,0].plot(range(0,len(T_plot)),Fi_MDD,'k-o',markersize=3,zorder=10)
    axes[1,0].set_xlabel("step number")
    axes[1,0].set_ylabel("Fractional Release (%)")
    #axes[1].axis('square')
    axes[1,0].set_box_aspect(1)


    #axes[2].subplot(n_plots,1,3)
    axes[1,1].errorbar(range(0,len(T_plot)),dataset["M"],yerr = dataset["delM"], fmt ='b-o',markersize=5,zorder=5)
    axes[1,1].plot(range(0,len(T_plot)),tot_moles*Fi_MDD,'k-o',markersize=3,zorder=10)
    axes[1,1].set_xlabel("step number")
    axes[1,1].set_ylabel("Atoms Released at Each Step")
    axes[1,1].set_box_aspect(1)
    #axes[2].axis('square')


    if n_plots == 4:
        #Calculate reference law results
        
        r_r0 = 0.5*(ref-dataset["ln(D/a^2)"])
        #axes[3].subplot(n_plots,1,n_plots)
        axes[0,1].plot(data[0]*100,r_r0, 'b-o', markersize=4)
        axes[0,1].set_xlabel("Cumulative 3He Release (%)")
        axes[0,1].set_ylabel("log(r/r_0)")
        axes[0,1].set_box_aspect(1)     
    plt.tight_layout
    plt.show()


dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
objective = DiffusionObjective(dataset, time_add = torch.tensor([300*60,2003040*60]), temp_add = torch.tensor([40,21.111111111111]), 
                               pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl",
                               omitValueIndices= []#[range(18,33)])
                                )

#nlc = NonlinearConstraint(con,lb =[0,-np.inf],ub = [np.inf,0])
nlc = NonlinearConstraint(conHe_Param,lb =[0,0],ub = [np.inf,np.inf])


# There is always one Ea (70,110)
# There are always num_domain # of LnD0aa (0,25)
# There are always num_domain-1 # of Fracs (0.001,1)
result = differential_evolution(
    objective, 
    [
        (3996895114 - 3996895114*0.01 , 3996895114 +3996895114*0.01 ),
        (0.0001,150),
        (-10, 30), 
        (-10,30),
        (-10,30),
        (0.00001,1),
        (0.00001,1),

    ], 
    disp=True, 
    tol=0.0001, #4 zeros seems like a good number from testing. slow, but useful.
    popsize = 100,
    maxiter = 100000000,
    constraints = nlc
)

print(result.x)


plot_results(torch.tensor(result.x),dataset,objective)
#reference_law = np.array([result.x[1],19])
print("\n\n")
print(["{0:0.6f}".format(i) for i in norm_dist_error])
print(["{0:0.6f}".format(i) for i in result.x])
print("\n\n")
print(norm_dist_error)
print(result.x)
