from diffusion_optimizer.optimization.base_optimizer import BaseOptimizer, BaseOptimizerOptions
from random import uniform
from diffusion_optimizer.optimization.calc import exponential_interpolation
import yaml


class DiffusionOptimizerOptions(BaseOptimizerOptions):
    def __init__(
        self,
        num_samp_max:float=1000,
        num_samp_ratio:float=0.025,
        num_samp_decay:float=20,
        resamp_ratio:float=0.5,
        maximize:bool=False,
        verbose:bool=False
    ):
        """
        This class contains a set of hyperparameters and settings for input into the DiffusionOptimizer class

        Args:
            num_samp_max (float, optional): The maximum number of samples per epoch. Defaults to 1000.
            num_samp_ratio (float, optional): The ratio between the maximum number of samples and the minimum. Defaults to 0.025.
            num_samp_decay (float, optional): The rate of decay for the number of samples taken per epoch. Defaults to 20.
            resamp_ratio (float, optional): The ratio between the number of samples and the number of resamples. Defaults to 0.5.
            maximize (bool, optional): Determines if trying to maximize the score or minimize. Defaults to False.
            verbose (bool, optional): Turns on/off logs to console. Defaults to False.
        """
        
        super().__init__(num_samp_max, resamp_ratio, maximize, verbose)
        self._num_samp_max = num_samp_max
        self._num_samp_ratio = num_samp_ratio
        self._num_samp_decay = num_samp_decay
        self._num_samp_range = range(int(self._num_samp_max * self._num_samp_ratio), self._num_samp_max)
    
    @staticmethod
    def from_yaml(config_path):
        config = yaml.safe_load(open(config_path))
        return DiffusionOptimizerOptions(
            config["num_samp_max"], 
            config["num_samp_ratio"], 
            config["num_samp_decay"], 
            config["resamp_ratio"],
            config["maximize"],
            config["verbose"],
        )


class DiffusionOptimizer(BaseOptimizer):
    def __init__(
        self, 
        objective:callable,
        limits:list[tuple], 
        lookup_table,
        args:DiffusionOptimizerOptions
    ):
        """
        This class describes a voronoi optimizer with adaptive sample counts.

        Args:
            • objective (callable): A callable that evaluates a single set of parameters and returns its fitness.
            • limits (list[tuple]): A list of tuples containing the ranges of each parameter's search space.
            • args (DiffusionOptimizerOptions): A set of arguments describing the optimizer's behaviour.
        """
        super().__init__(
            objective, 
            limits, 
            lookup_table,
            args
        )
        self.args = args
        
    # def step(self, count):
    #     super().step(count)
    #     # best_score = self.get_best()._score
    #     # self.args.num_samp = min(self.args._num_samp_max, int(exponential_interpolation([self.args._num_samp_range[0], self.args._num_samp_max],self.args._num_samp_decay, best_score))) 
    #     # num_resamp = int(self.args.num_samp * self.args.resamp_ratio)
    #     # self._elites = self._elites[:num_resamp]
        
    def __repr__(self):
        out = super().__repr__()
        out += f"  {self.args.num_samp}  {self.args.num_samp * self.args.resamp_ratio}"
        return out