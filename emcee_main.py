if __name__ == '__main__':

    import numpy as  np
    from diffusion_optimizer.neighborhood.dataset import Dataset
    from diffusion_optimizer.diffusion_objective_3HeParam_emcee import DiffusionObjective
    import pandas as pd
    import torch as torch
    import emcee
    import matplotlib.pyplot as plt
    import corner
    from multiprocessing import Pool
    import os
    import cProfile

    os.environ["OMP_NUM_THREADS"] = "1"


    #from generate_samples import create_samples
    #from diffusion_optimizer.diffusion_optimization_v2.diffusion_optimizer.src.diffusion_optimizer.optimizers import DiffusionOptimizer, DiffusionOptimizerOptions

    # def create_walkers(input_csv,limits,lookup_table,args):
    #     objective = DiffusionObjective(dataset, [],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")
    #     opt = DiffusionOptimizer(objective,limits,lookup_table,args)
    #     count = len(data)
    #     while opt.iter<100
    #         count = opt.step(count)
    #         best = opt.get_best()
    #         print(f"Best: {best}")
    #         print(f"number of scores with 0: {count}")
    #     for sample in opt._samples:
    #         if sample._score == 0 and sample._params not in data.values:
    #             data.loc[len(data)] = [sample._params.numpy()]

    def fracs_prior(fracs):
        if np.sum(fracs) <= 1 and np.sum(fracs) == np.sum(np.abs(fracs)):
            return 0
        else:
            return -np.inf
        
    def moles_prior(moles):

        # if 4.0000013380000000000E+09-24943848.04*3 < moles < 4.0000013380000000000E+09+24943848.04*3:
        #     return 0
        # else:
        #     return -np.inf
        return -(1/2)*((moles-2510948713)**2/(12772561.78**2))
        
    def Ea_prior(Ea):
        if 30 < Ea < 150:
            return 0
        else:
            return -np.inf
    def lnd0aa_prior(lnD0aa):
        if len(lnD0aa)>1: #If we have greater than one domain
            diff = lnD0aa[0:-1]-lnD0aa[1:] # Take difference and 
            if sum(diff<0) > 0:
                return -np.inf
            elif all(-10 <= x <= 35 for x in lnD0aa):
                return 0
            else:
                return -np.inf
        else:
            return 0




    def log_prior(theta,objective):
        #dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
        #objective = DiffusionObjective(dataset, [],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")
        #flags = objective.check_flags(theta)
        total_moles = theta[0]
        moles_P = moles_prior(total_moles)
        theta = theta[1:] # Shorten X to remove this parameter
        Ea = theta[0]
        
        # Get the Ea prior
        Ea_P = Ea_prior(Ea)

        # Unpack the parameters
        if len(theta) <= 3:
            ndom = 1
        else:
            ndom = (len(theta))//2

        # Grab the other parameters from the input
        temp = theta[1:]
        lnD0aa = temp[0:ndom]
        # Get lnD0aa prior
        lnD0aa_P = lnd0aa_prior(lnD0aa)

        # Get fracs prior
        fracs = temp[ndom:]
        fracs_P = fracs_prior(fracs)

        return moles_P+Ea_P+lnD0aa_P+fracs_P#+flags

    def log_likelihood(theta,objective):

        #dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
        
        ans = (objective.__call__(theta)).numpy()

        # Find out why this is happening in the objective function if this fix works..


        if ans.size > 1:
            ans = np.sum(ans)
        return -(1/2)*ans #Check if this should be 1/2

    def log_probability(theta,objective):

        lp = log_prior(theta,objective)
        if not np.isfinite(lp):
            return -np.inf

        return lp+log_likelihood(theta,objective)



    # Define the irradiation and lab storage steps, if necessary
    seconds_since_irrad = torch.tensor(110073600)  # seconds
    irrad_duration_sec = torch.tensor(5*3600) # seconds
    irrad_T = torch.tensor(40) # C
    storage_T = torch.tensor(21.1111111) # in C
    
    # Make a tensor with these two extra heating steps in order.
    time_add = torch.tensor([irrad_duration_sec,seconds_since_irrad])
    temp_add = torch.tensor([irrad_T, storage_T])


    dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
    objective = DiffusionObjective(dataset, 
                                time_add = time_add,
                                temp_add =temp_add,
                                pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl",
                                )
    

    #x = np.array([2.49328712e+09, 7.86268769e+01, 1.16024686e+01, 1.03789158e+01, 6.83687948e+00, 2.71839371e+00, 6.55804826e-01, 3.26860029e-01, 9.64080513e-03])

    x = np.array([3.49328712e+09, 5.86268769e+01, 2.16024686e+01, 2.03789158e+01,9.83687948e+00, 5.71839371e+00, 7.55804826e-01, 1.26860029e-01, 9.64080513e-03])


    nwalkers = 25
    ndim = 9


    #pos = create_samples(nwalkers, (torch.tensor([ 50.0,    0,   0,   0,   0.001, 0.001]),torch.tensor([ 150.0,  35.0,  35.0,  35.0,  1.0,   1.0  ])))
    multiplier = np.concatenate((np.random.uniform(low=-0.01, high=0.01, size=[nwalkers, 1]), np.random.uniform(low=-0.01, high=0.01, size=[nwalkers, ndim-1])), axis=1)
    pos = np.tile(x, (nwalkers, 1)) * (1 + multiplier) 


    # pos = np.tile(x,(nwalkers,1))+ np.tile(x,(nwalkers,1))*np.random.uniform(low= -0.05,high =0.05,size=[nwalkers,ndim])
    # pos = np.row_stack((x,pos))



    # moles_add = np.random.normal(4e+09, 24943848.04, size=(nwalkers, 1))

    # pos = np.column_stack([moles_add,pos])
    filename = "snooker_moves_largeRun.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim,log_probability, args = [objective],backend = backend,
        moves=[
        (emcee.moves.DEMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ] 
    )
    sampler.run_mcmc(pos, 2000000, progress=True)


    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["Moles","Ea","LnD0aa1","LnD0aa2","LnD0aa3","LnD0aa4","Frac1","Frac2","Frac3"]

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");



    #tau = sampler.get_autocorr_time()
    #print(tau)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)



    fig1 = corner.corner(flat_samples, labels=labels)
    plt.show()

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [50-16, 50, 50+16])
        print(mcmc)

    tau = sampler.get_autocorr_time()
    print(tau)
