import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import os 
import pandas as pd
import yaml
from diffusion_optimizer.optimizers import DiffusionOptimizer, DiffusionOptimizerOptions
from diffusion_optimizer.optimization.diffusion_objective import DiffusionObjective
import time
import torch
import pickle

def list2tuple(list):
    return (list[0], list[1])

# create a dataset to add new samples to
DATASET_PATH = "/Users/josh/repos/drew_nn_opt/params-2.parquet"
# if dataset doesnt exist, create it
if not os.path.exists(DATASET_PATH):
    data = pd.DataFrame(columns=["x"])
    data.to_parquet(DATASET_PATH)
else:
    data = pd.read_parquet(DATASET_PATH)

def single_optimize(input_csv, limits, lookup_table, args):     
    objective = DiffusionObjective(pd.read_csv(input_csv))
    opt = DiffusionOptimizer(objective, limits, lookup_table, args)
    count = len(data)
    while opt._iter < 100:
        count = opt.step(count)
        best = opt.get_best()
        print(f"Best: {best}")
        print(f"number of scores with 0: {count}")
        
    # add all samples that are 0 to the dataset if they are not already in the dataset
    for sample in opt._samples:
        if sample._score == 0 and sample._params not in data.values:
            data.loc[len(data)] = [sample._params.numpy()]
    
    # save dataset
    print(len(data))
    data.to_parquet(DATASET_PATH)
    

if __name__ == '__main__':
    config_path = "/Users/josh/repos/diffusion_optimization_v2/demos/single_run/config.yaml"
    input_csv = "/Users/josh/Downloads/data4Optimizer.csv"
    # optimize_async("/Users/josh/repos/diffusion_optimization_v2/demos/single_run/data4Optimizer.csv", "/Users/josh/repos/diffusion_optimization_v2/demos/single_run/config.yaml", "/Users/josh/repos/diffusion_optimization_v2/demos/single_run")
    # # load config
    config = yaml.safe_load(open(config_path))

    # create limit tuple array
    limitDict = config["limits"]
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # load lookup table as pkl
    with open("/Users/josh/repos/diffusion_optimization_v2/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl", "rb") as f:
        lookup_table = pickle.load(f)

    # create options struct
    options = DiffusionOptimizerOptions.from_yaml(config_path)
    for i in range(1000):
        single_optimize(input_csv, limits, lookup_table, options)
    