from ray import tune
import numpy as np
import yaml
from diffusion_optimizer.neighborhood.dataset import Dataset
from ray.tune.search.bayesopt import BayesOptSearch
import os
import pandas as pd
from ray.tune.search.hyperopt import HyperOptSearch

from diffusion_optimizer.parameter_tuning.diffusion_optimizer_fitness import generate_trainable
from ray import air


def list2range(list):
    return range(list[0], list[1])

def list2tuple(list):
    return (list[0], list[1])

def tune_parameters(config_path, dataset_csv_path, output_path):
    config = yaml.safe_load(open(config_path))
    #num_samp_min = config["num_samp_min"]
    num_samp_ratio = config["num_samp_ratio"]
    num_samp_max = config["num_samp_max"]
    num_samp_decay = config["num_samp_decay"]
    resamp_to_samp_ratio = config["resamp_to_samp_ratio"]
    #epsilon_min = config["epsilon_min"]
    epsilon_ratio = config["epsilon_ratio"]
    epsilon_max = config["epsilon_max"]
    epsilon_growth = config["epsilon_growth"]
    omit_value_incides = config["omitValueIndices"]

    search_space = {
        #"num_samp_min": tune.randint(num_samp_min[0], num_samp_min[1]),
        # make sure sample max is larger than sample min
        "num_samp_max": tune.randint(num_samp_max[0], num_samp_max[1]), 
        "num_samp_ratio": tune.uniform(num_samp_ratio[0],num_samp_ratio[1]),
        "num_samp_decay": tune.uniform(num_samp_decay[0], num_samp_decay[1]),
        "resamp_to_samp_ratio": tune.uniform(resamp_to_samp_ratio[0], resamp_to_samp_ratio[1]),
        #"epsilon_min": tune.uniform(epsilon_min[0], epsilon_min[1]),
        # make sure epsilon max is larger than epsilon min
        "epsilon_max": tune.uniform(epsilon_max[0],epsilon_max[1]),
        "epsilon_ratio": tune.uniform(epsilon_ratio[0],epsilon_ratio[1]),
        "epsilon_growth": tune.uniform(epsilon_growth[0], epsilon_growth[1])
    }
    
    # convert arrays to tuples and limits in config
    limitDict = config["limits"]
    names = list(limitDict.keys())
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # create objective
    df = pd.read_csv(dataset_csv_path)
    dataset = Dataset(df)
    objective = generate_trainable(dataset, limits, names, config["max_iters"], config["threshold"],config["omitValueIndices"])
    
    trainable_with_cpu_gpu = tune.with_resources(objective, {"cpu": 12,"gpu":2}) #, "gpu": 1


    hyperopt_search = HyperOptSearch(
        metric="score",
        mode="min",
    )


    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=12,
            search_alg=hyperopt_search,
        ),
        run_config=air.RunConfig(stop={"training_iteration":150}),
    )
    results=tuner.fit()
    df = results.get_dataframe()
    df.to_parquet(output_path + "/results.parquet")
    return results