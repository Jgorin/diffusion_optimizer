import os 
import pandas as pd
import yaml
from diffusion_optimizer.optimizers import DiffusionOptimizer, DiffusionOptimizerOptions
from diffusion_optimizer.optimization.diffusion_objective import DiffusionObjective

FILE_PATH = os.path.abspath("../../")
CONFIG_PATH = F"{FILE_PATH}/demos/single_run/config.yaml"
INPUT_CSV = f"{FILE_PATH}/demos/single_run/data4Optimizer.csv"

def list2tuple(list):
    return (list[0], list[1])


# load config
config = yaml.safe_load(open(CONFIG_PATH))

# create limit tuple array
limitDict = config["limits"]
limits = [list2tuple(limit) for limit in limitDict.values()]

# create options struct
options = DiffusionOptimizerOptions.from_yaml(CONFIG_PATH)

# pass dataset into objective class
objective = DiffusionObjective(pd.read_csv(INPUT_CSV), config["omit_value_indices"])

# instantiate optimizer
opt = DiffusionOptimizer(objective, limits=limits, args=options)
best_fit = opt.fit(eps=100)

# df = opt.plot()
print(best_fit)
