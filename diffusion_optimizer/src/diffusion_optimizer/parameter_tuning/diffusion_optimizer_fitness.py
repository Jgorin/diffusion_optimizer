from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.neighborhood.optimizer import Optimizer
import time

from diffusion_optimizer.neighborhood.optimizer import OptimizerOptions
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective import DiffusionObjective
from ray.tune.trainable import Trainable
import os
import torch
import numpy as np

def generate_trainable(dataset:Dataset, limits, names, max_iters, threshold,omit_value_indices):
    class DiffusionHyperparameterObjective(Trainable):
        def __init__(self, config, logger_creator=None, remote_checkpoint_dir=None, custom_syncer=None, sync_timeout=None):
            super().__init__(config, logger_creator, remote_checkpoint_dir, custom_syncer, sync_timeout)
            self._config = config
            self._dataset = dataset
            self._limits = limits
            self._names = names
            self._max_iters = max_iters
            self._threshold = threshold
            self._omit_value_indices = omit_value_indices
            
        
        def step(self):
            ds = Dataset(self._dataset)
            config = self._config
            #num_samp_range = [config["num_samp_min"], config["num_samp_max"]]
            num_samp_decay = config["num_samp_decay"]
            resamp_ratio = config["resamp_to_samp_ratio"]
            #epsilon_range = [config["epsilon_min"], config["epsilon_max"]]
            epsilon_growth = config["epsilon_growth"]
            num_samp_max = config["num_samp_max"]
            epsilon_ratio = config["epsilon_ratio"]
            num_samp_ratio = config["num_samp_ratio"]
            epsilon_max = config["epsilon_max"]

            objective = DiffusionObjective(ds,self._omit_value_indices)
            
            options = OptimizerOptions(
                num_samp_decay,
                resamp_ratio,
                epsilon_growth,
                num_samp_max,
                num_samp_ratio,
                epsilon_ratio,
                epsilon_max

            )
            self.optimizer = Optimizer(objective, self._limits, self._names, options, maximize=False, verbose=False)
 
            comparator = max if self.optimizer._maximize else min
            


            # keep running epochs until threshold is met or the max iterations is hit
            max_samps = 35000

            scoreNum = 0
            for i in range(0,10):
                exit_flag = 0
                while (self.optimizer._iter == 0 or (self.optimizer._iter < self._max_iters and comparator(self.optimizer.get_best()._res, self._threshold) == self._threshold and self.optimizer.sample_manager.get_std() > 10**(-5))) and exit_flag ==0 :
                    self.optimizer.update(num_iter=1)             
                    if len(self.optimizer.sample_manager._samples) >max_samps:
                        scoreNum = scoreNum + 10**10
                        exit_flag = 1

                        #return {"score": 10**10}

                    if self.optimizer._iter >= self._max_iters and comparator(self.optimizer.get_best()._res, self._threshold) == self._threshold:
                        scoreNum = scoreNum + 10**10
                        exit_flag = 1

                        #return {"score": 10**10}


                scoreNum += self.optimizer.get_best()._res + 0.2*len(self.optimizer.sample_manager._samples)/max_samps
            #return {"score": len(self.optimizer.sample_manager._samples)/max_iters + self.optimizer.get_best()._res}
            return{"score": scoreNum}
    
        def save_checkpoint(self, tmp_checkpoint_dir):
            print(tmp_checkpoint_dir)
            checkpoint_path = os.path.join(tmp_checkpoint_dir, "idk.json")
            self.optimizer.get_best().save_as_json(checkpoint_path)
            return tmp_checkpoint_dir
        
    return DiffusionHyperparameterObjective