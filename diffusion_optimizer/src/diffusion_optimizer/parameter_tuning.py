# I wrote this gross while loop to try to test out the possible values of the model
import time
from diffusion_optimizer.diffusion_objective import DiffusionObjective
from diffusion_optimizer.neighborhood.optimizer import Optimizer
from diffusion_optimizer.neighborhood.dataset import Dataset
import numpy as np
import pandas as pd

def tune_parameters(
    nameOfInputCSVFile="3Domains.csv", 
    nameOfExperimentalResultsFile="data4Optimizer.csv", 
    threshold=0.01, 
    num_sampToTry=range(3,21), 
    num_ResampToTry=range(3,20), 
    misfitVals=np.ones([101,101])*10*11, 
    numIterations4Save=np.ones([100,101])*10*11, 
    durations=np.ones([101,101])*10*11, 
    names=["Ea","LnD0aa1","LnD0aa2","LnD0aa3","Frac1","Frac2"],
    limits=[(70,110),(0,25),(0,25),(0,25),(10**(-10),1),(10**(-10),1)]
):
    
    objective = DiffusionObjective(Dataset(pd.read_csv(nameOfExperimentalResultsFile)))
    for i in num_sampToTry:
        j=i
        while j <=100:

            misfit = 10**11
            counter = 0
            print("num_samp = " +str(j))
            print("num_resamp = " +str(i))
            srch = Optimizer(
                    objective=objective,
                    names = names,
                    limits=limits, 
                    num_samp=8, #number of random samples taken at each iteration
                    num_resamp=4, #number of best Voronoi polygons sampled at each iteration-- must be smaller than num_samp
                    maximize=False,
                    verbose=True
                    )
            start_time = time.time()
            didNotconverge = 0
            numIters = 0
            while (misfit >= threshold) and (didNotconverge == 0): # While misfit is below the threshold and we haven't failed to converge
                srch.update(100)
                misfit = srch.sample[0]["result"]
                numIters = numIters+1
                if numIters > 2000:
                    didNotconverge = 1
                elif (numIters > 100) and (srch.sample[0]["result"] == srch.sample[75]["result"]):
                    didNotConverge = 1
            if didNotconverge != 1: #This means we converged
                misfitVals[i,j] = misfit
                durations[i,j] = (time.time() - start_time)
                numIterations4Save[i,j] = numIters
                j=j+1

    np.savetxt("misfitVals.csv", misfitVals,delimiter = ',')
    np.savetxt("numiters.csv", numIterations4Save, delimiter = ',')
    np.savetxt("durations.csv", durations, delimiter = ',')