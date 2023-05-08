from diffusion_optimizer.optimization.diffusion_objective import DiffusionDataset, DiffusionObjective
from diffusion_optimizer.optimizers import GEDOptimizer, GEDOptions
from ray.tune.trainable import Trainable
import os

def generate_trainable(dataset:DiffusionDataset, limits, max_iters, threshold, omit_value_indices):
    class DiffusionHyperparameterObjective(Trainable):
        def __init__(self, config, logger_creator=None, remote_checkpoint_dir=None, custom_syncer=None, sync_timeout=None):
            super().__init__(config, logger_creator, remote_checkpoint_dir, custom_syncer, sync_timeout)
            self._config = config
            self._dataset = dataset
            self._limits = limits
            self._max_iters = max_iters
            self._threshold = threshold
            self._omit_value_indices = omit_value_indices
            
        
        def step(self):
            ds = DiffusionDataset(self._dataset)
            config = self._config
            num_samp_decay = config["num_samp_decay"]
            resamp_ratio = config["resamp_to_samp_ratio"]
            epsilon_growth = config["epsilon_growth"]
            num_samp_max = config["num_samp_max"]
            epsilon_ratio = config["epsilon_ratio"]
            num_samp_ratio = config["num_samp_ratio"]
            epsilon_max = config["epsilon_max"]
            max_sample_cutoff = config["max_sample_cutoff"]
            elite_stddev_convergence_threshold = config["elite_stddev_convergence_threshold"]

            objective = DiffusionObjective(ds, self._omit_value_indices)
            
            options = GEDOptions(
                num_samp_max=num_samp_max,
                num_samp_ratio=num_samp_ratio,
                num_samp_decay=num_samp_decay,
                resamp_ratio=resamp_ratio,
                epsilon_max=epsilon_max,
                epsilon_ratio=epsilon_ratio,
                epsilon_growth=epsilon_growth,
                maximize=False,
                verbose=False
            )
            
            self.optimizer = GEDOptimizer(objective, self._limits, options)
 
            comparator = max if self.optimizer.args._maximize else min
            while self.optimizer._iter == 0 or \
                    (self.optimizer._iter < self._max_iters and \
                    len(self.optimizer._normalized_params) < max_sample_cutoff and \
                    comparator(self.optimizer.get_best()._score, self._threshold) == self._threshold and \
                    self.optimizer.get_elite_stddev() > elite_stddev_convergence_threshold):
                        
                self.optimizer.step()
            print(self.optimizer._iter)
                             
            return self.get_fitness()
        
        
        def get_fitness(self):
            # check max sample count condition
            if len(self.optimizer._normalized_params) > 15000:
                return {"score": 10**10}
            
            # check if didn't converge
            comparator = max if self.optimizer.args._maximize else min
            if comparator(self.optimizer.get_best()._score, self._threshold) == self._threshold:
                return {"score": 10**10}
            
            return { "score": len(self.optimizer._normalized_params)*self.optimizer.get_best()._score }
            
    
        def save_checkpoint(self, tmp_checkpoint_dir):
            checkpoint_path = os.path.join(tmp_checkpoint_dir, "idk.json")
            self.optimizer.get_best().save_as_json(checkpoint_path)
            return tmp_checkpoint_dir
        
    return DiffusionHyperparameterObjective