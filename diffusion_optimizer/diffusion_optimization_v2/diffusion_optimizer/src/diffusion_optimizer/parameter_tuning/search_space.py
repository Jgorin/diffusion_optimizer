import yaml
from ray import tune

class SearchSpace:
    def __init__(self, config):
        self._config = config
        
        num_samp_ratio = self._config["num_samp_ratio"]
        num_samp_max = self._config["num_samp_max"]
        num_samp_decay = self._config["num_samp_decay"]
        resamp_to_samp_ratio = self._config["resamp_to_samp_ratio"]
        epsilon_ratio = self._config["epsilon_ratio"]
        epsilon_max = self._config["epsilon_max"]
        epsilon_growth = self._config["epsilon_growth"]
        
        self._search_space = {
            "num_samp_max": tune.randint(num_samp_max[0], num_samp_max[1]), 
            "num_samp_ratio": tune.uniform(num_samp_ratio[0],num_samp_ratio[1]),
            "num_samp_decay": tune.uniform(num_samp_decay[0], num_samp_decay[1]),
            "resamp_to_samp_ratio": tune.uniform(resamp_to_samp_ratio[0], resamp_to_samp_ratio[1]),
            "epsilon_max": tune.uniform(epsilon_max[0],epsilon_max[1]),
            "epsilon_ratio": tune.uniform(epsilon_ratio[0],epsilon_ratio[1]),
            "epsilon_growth": tune.uniform(epsilon_growth[0], epsilon_growth[1])
        }
    
    @staticmethod
    def from_config_path(config_path):
        config = yaml.safe_load(open(config_path))
        return SearchSpace(config)
