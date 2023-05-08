from diffusion_optimizer.parameter_tuning.tune_parameters import tune_parameters
import os

FILE_PATH = os.path.abspath("../../")
CONFIG_PATH = f"{FILE_PATH}/demos/parameter_tuning/search_space.yaml"
INPUT_CSV = f"{FILE_PATH}/demos/single_run/data4Optimizer.csv"

tune_parameters(CONFIG_PATH, INPUT_CSV, f"{FILE_PATH}/demos/parameter_tuning")