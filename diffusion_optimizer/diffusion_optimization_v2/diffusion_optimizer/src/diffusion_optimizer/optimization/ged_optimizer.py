from diffusion_optimizer.optimization.diffusion_optimizer import DiffusionOptimizer, DiffusionOptimizerOptions
from random import uniform
from diffusion_optimizer.optimization.calc import exponential_interpolation
import yaml

class GEDOptions(DiffusionOptimizerOptions):
    def __init__(
        self,
        num_samp_max:float=1000,
        num_samp_ratio:float=0.025,
        num_samp_decay:float=20,
        resamp_ratio:float=0.5,
        epsilon_max:float=0.001,
        epsilon_ratio:float=0.01,
        epsilon_growth:float=0.5,
        maximize:bool=False,
        verbose:bool=False
    ):
        """
        This class contains a set of hyperparameters and settings for input into the GEDOptimizer class

        Args:
            num_samp_max (float, optional): The maximum number of samples per epoch. Defaults to 1000.
            num_samp_ratio (float, optional): The ratio between the maximum number of samples and the minimum. Defaults to 0.025.
            num_samp_decay (float, optional): The rate of decay for the number of samples taken per epoch. Defaults to 20.
            resamp_ratio (float, optional): The ratio between the number of samples and the number of resamples. Defaults to 0.5.
            epsilon_max (float, optional): The maximum value for epsilon. Defaults to 0.001.
            epsilon_ratio (float, optional): The ratio between the maximum and minimum value for epsilon. Defaults to 0.01.
            epsilon_growth (float, optional): The rate of growth for epsilon. Defaults to 0.5.
            maximize (bool, optional): Determines if trying to maximize the score or minimize. Defaults to False.
            verbose (bool, optional): Turns on/off logs to console. Defaults to False.
        """
        
        super().__init__(
            num_samp_max, 
            num_samp_ratio, 
            num_samp_decay,
            resamp_ratio,
            maximize, 
            verbose
        )
        
        self._epsilon_max = epsilon_max
        self._epsilon_ratio = epsilon_ratio
        self._epsilon_growth = epsilon_growth
        self._epsilon = epsilon_max * epsilon_ratio
        self._num_epsilon_range = [self._epsilon_max * self._epsilon_ratio, self._epsilon_max]
    
    @staticmethod
    def from_yaml(yaml_path):
        config = yaml.safe_load(open(yaml_path))
        return GEDOptions(
            config["num_samp_max"], 
            config["num_samp_ratio"], 
            config["num_samp_decay"], 
            config["resamp_ratio"],
            config["epsilon_max"],
            config["epsilon_ratio"],
            config["epsilon_growth"],
            config["maximize"],
            config["verbose"]
        )
        

class GEDOptimizer(DiffusionOptimizer):
    def __init__(
        self, 
        objective:callable,
        limits:list[tuple], 
        args:GEDOptions
    ):
        """
        This class describes a voronoi optimizer with adaptive sample counts and a greedy epsilon. Greedy epsilon is applied per population member.

        Args:
            • objective (callable): A callable that evaluates a single set of parameters and returns its fitness.
            • limits (list[tuple]): A list of tuples containing the ranges of each parameter's search space.
            • args (GEDOptions): A set of arguments describing the optimizer's behaviour.
        """
        
        super().__init__(
            objective, 
            limits, 
            args
        )
        self.args = args
        
    def step(self):
        # adds adaptive epsilon value based on best score
        super().step()
        best_score = self.get_best()._score
        self.args._epsilon = exponential_interpolation(self.args._num_epsilon_range, self.args._epsilon_growth, best_score)
    
    def _neighborhood_sample(self, elite_index):
        # adds random sampling chance based on current epsilon value
        epsilon = uniform(0, 1)
        if epsilon <= self.args._epsilon:
            self._queue.append(self._random_sample())
        else:
            super()._neighborhood_sample(elite_index)