# I wrote this gross while loop to try to test out the possible values of the model
import time
from diffusion_optimizer.diffusion_objective import DiffusionObjective
from diffusion_optimizer.neighborhood.optimizer import Optimizer
from diffusion_optimizer.neighborhood.dataset import Dataset
import numpy as np
import pandas as pd
import os

file_path = os.path.dirname(os.path.realpath(__file__))
output_path = f"{file_path}/../../../main/output"

def tune_parameters(
    nameOfInputCSVFile="3Domains.csv", 
    nameOfExperimentalResultsFile="data4Optimizer.csv", 
    threshold=0.01, 
    num_sampToTry=range(3,21), 
    num_ResampToTry=range(3,20), 
    num_epsilonsToTry=[0.1, 0.15, 0.2, 0.25],
    misfitVals=np.ones([101,101])*10*11, 
    numIterations4Save=np.ones([100,101])*10*11, 
    durations=np.ones([101,101])*10*11, 
    names=["Ea","LnD0aa1","LnD0aa2","LnD0aa3","Frac1","Frac2"],
    limits=[(70,110),(0,25),(0,25),(0,25),(10**(-10),1),(10**(-10),1)],
    output_dir=output_path
):
    
    objective = DiffusionObjective(Dataset(pd.read_csv(nameOfExperimentalResultsFile)))
    for i in num_ResampToTry:
        for j in num_sampToTry:
            for x in num_epsilonsToTry:
                misfit = 10**11
                print(f"num_samp: {j}")
                print(f"num_resamp: {i}")
                print(f"epsilon threshold: {x}")
                srch = Optimizer(
                        objective=objective,
                        names = names,
                        limits=limits, 
                        num_samp=j, #number of random samples taken at each iteration
                        num_resamp=i, #number of best Voronoi polygons sampled at each iteration-- must be smaller than num_samp
                        epsilon_threshold=x,
                        maximize=False,
                        verbose=True
                        )
                start_time = time.time()
                didNotconverge = 0
                while (misfit >= threshold) and (didNotconverge == 0): # While misfit is below the threshold and we haven't failed to converge
                    srch.update(100)
                    misfit = srch.sample_manager._elites[0]._res
                    if misfit > threshold:
                        didNotConverge = 1
                if didNotconverge != 1: #This means we converged
                    misfitVals[i,j] = misfit
                    durations[i,j] = (time.time() - start_time)
                    numIterations4Save[i,j] = srch._iter
                
    np.savetxt(f"{output_dir}/misfitVals.csv", misfitVals,delimiter = ',')
    np.savetxt(f"{output_dir}/numiters.csv", numIterations4Save, delimiter = ',')
    np.savetxt(f"{output_dir}/durations.csv", durations, delimiter = ',')