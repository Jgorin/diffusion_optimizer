from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.neighborhood.optimizer import Optimizer
import time

from diffusion_optimizer.neighborhood.optimizer import OptimizerOptions
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective import DiffusionObjective
from ray.tune.trainable import Trainable
import os
import torch

def generate_trainable(dataset:Dataset, limits, names, max_iters, threshold):
    class DiffusionHyperparameterObjective(Trainable):
        def __init__(self, config, logger_creator=None, remote_checkpoint_dir=None, custom_syncer=None, sync_timeout=None):
            super().__init__(config, logger_creator, remote_checkpoint_dir, custom_syncer, sync_timeout)
            self._config = config
            self._dataset = dataset
            self._limits = limits
            self._names = names
            self._max_iters = max_iters
            self._threshold = threshold
        
        def step(self):
            ds = Dataset(self._dataset)
            objective = DiffusionObjective(ds)
            config = self._config
            num_samp_range = [config["num_samp_min"], config["num_samp_max"]]
            num_samp_decay = config["num_samp_decay"]
            resamp_ratio = config["resamp_to_samp_ratio"]
            epsilon_range = [config["epsilon_min"], config["epsilon_max"]]
            epsilon_growth = config["epsilon_growth"]
            options = OptimizerOptions(
                num_samp_range, 
                num_samp_decay, 
                resamp_ratio, 
                epsilon_range, 
                epsilon_growth
            )
            self.optimizer = Optimizer(objective, self._limits, self._names, options, maximize=False, verbose=False)
            t0 = time.time()
            comparator = max if self.optimizer._maximize else min
            
            # keep running epochs until threshold is met or the max iterations is hit
            while self.optimizer._iter == 0 or (self.optimizer._iter < self._max_iters and comparator(self.optimizer.get_best()._res, self._threshold) == self._threshold):
                self.optimizer.update(num_iter=1)
                
                if self.optimizer._iter >= self._max_iters and comparator(self.optimizer.get_best()._res, self._threshold) == self._threshold:
                    return {"score": 10**10}
            
            t1 = time.time()
            return {"score": len(self.optimizer.sample_manager._samples)}
    
        def save_checkpoint(self, tmp_checkpoint_dir):
            print(tmp_checkpoint_dir)
            checkpoint_path = os.path.join(tmp_checkpoint_dir, "idk.json")
            self.optimizer.get_best().save_as_json(checkpoint_path)
            return tmp_checkpoint_dir
        
    return DiffusionHyperparameterObjective