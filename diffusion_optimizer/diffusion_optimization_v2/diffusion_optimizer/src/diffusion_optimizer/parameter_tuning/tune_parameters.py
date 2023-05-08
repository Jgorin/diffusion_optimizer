from ray import tune
import yaml
import pandas as pd
from ray.tune.search.hyperopt import HyperOptSearch
from ray import air
from diffusion_optimizer.optimization.diffusion_objective import DiffusionDataset
from diffusion_optimizer.parameter_tuning.hyperoptimizer import generate_trainable
from diffusion_optimizer.parameter_tuning.search_space import SearchSpace


def list2range(list):
    return range(list[0], list[1])

def list2tuple(list):
    return (list[0], list[1])

def tune_parameters(config_path, dataset_csv_path, output_path):
    config = yaml.safe_load(open(config_path))
    omit_value_incides = config["omit_value_indices"]

    ss = SearchSpace(config)
    
    # convert arrays to tuples and limits in config
    limitDict = config["limits"]
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # create objective
    df = pd.read_csv(dataset_csv_path)
    dataset = DiffusionDataset(df)
    objective = generate_trainable(dataset, limits, config["max_iters"], config["threshold"], config["omit_value_indices"])
    
    trainable_with_cpu_gpu = tune.with_resources(objective, {"cpu": 6}) #, "gpu": 1

    hyperopt_search = HyperOptSearch(
        metric="score",
        mode="min",
    )

    tuner = tune.Tuner(
        objective,
        param_space=ss._search_space,
        tune_config=tune.TuneConfig(
            num_samples=1,
            search_alg=hyperopt_search
        ),
        run_config=air.RunConfig(stop={"training_iteration":1}),
    )
    
    results=tuner.fit()
    df = results.get_dataframe()
    df.to_parquet(output_path + "/results.parquet")
    
    return results