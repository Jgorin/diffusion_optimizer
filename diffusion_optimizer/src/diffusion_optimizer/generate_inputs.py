import pandas as pd
from diffusion_optimizer.utils.utils import D0calc_MonteCarloErrors

def generate_inputs(nameOfInputCSVFile="3Domains.csv", nameOfExperimentalResultsFile="data4Optimizer.csv"):
    #Code written by Marissa Tremblay and transcribed into Python/modified by Drew Gorin. Last modified 1.2023
    #This m-file is used to fit an MDD model to stepwise degassing diffusion
    #experiment data. It is currently set up for only one isotope. The number
    #of domains is allowed to vary. The activation energy is assumed to be the
    #same across domains while the pre-exponential factor (D0/a^2) and the
    #fraction of gas in each domain varies. Needs the companion functions
    #D0calc_MonteCarloErros.m and TremblayMDD.m.

    expData = pd.read_csv(nameOfInputCSVFile,header=None)
        
    #If extra columns get read in, trim them down to just 3
    if expData.shape[1] >=4:
        expData = expData.loc[:,1:4]
        
    # Name the columsn of the iput data
    expData.columns = ["TC", "thr","M", "delM"]

    # Calculate Daa from experimental results
    expResults = D0calc_MonteCarloErrors(expData)

    # Combine the diffusion parameters with the experimental setup (T, thr, M, delM)
    # to get a final dataframe that will be passed into the optimizer
    diffusionExperimentResults = expData.join(expResults)

    # Write dataframe to a .csv file
    diffusionExperimentResults.to_csv(nameOfExperimentalResultsFile)