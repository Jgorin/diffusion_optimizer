import numpy as np
from random import uniform
import bisect
import torch
import copy
import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt


class BaseOptimizerSample:
    """
    this class represents a single sample in the optimization process
    inputs:
        • params (torch.tensor): the parameters of the sample
        • score (float): the score of the sample
        • iter (int): the iteration the sample was created
        • index (int): the index of the sample in the sample list
    """
    def __init__(self, params, score, iter, index):
        self._params = params
        self._score = score
        self._iter = iter
        self._index = index
    
    def to_dict(self):
        return { "score": self._score, "iter": self._iter, "params": self._params.tolist() }
    
    def save_as_json(self, path):
        """
        this function saves the sample as a json file
        inputs:
            • path (str): the path to save the file to
        """
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4, sort_keys=True)
            
    def __repr__(self):
        out = '{}(\n    params={},\n    score={:.4f},\n    iter={},\n    index={}\n)'.format(
            self.__class__.__name__,
            torch.round(self._params, decimals=4).tolist(),
            self._score,
            self._iter,
            self._index)
        return out
        

class BaseOptimizerOptions:
    def __init__(
        self,
        num_samp:int=10,
        resamp_ratio:float=0.5,
        maximize:bool=False,
        verbose:bool=False
    ):
        """
        This class contains a set of hyperparameters and settings for input into the BaseOptimizer class

        Args:
            • num_samp (float, optional): The number of samples per epoch. Defaults to 10.
            • resamp_ratio (float, optional): The ratio between the number of samples and resamples. Defaults to 0.5.
            • maximize (bool, optional): Determines if trying to maximize the score or minimize. Defaults to False.
            • verbose (bool, optional): Turns on/off logs to console. Defaults to False.
        """
        
        assert num_samp != None and num_samp > 0, "Number of samples must be greater than zero! got {num_samp}."
        assert resamp_ratio != None and resamp_ratio > 0 and resamp_ratio <= 1, f"Resamp ratio {resamp_ratio} out of range."
        
        self.num_samp = num_samp
        self.resamp_ratio = resamp_ratio
        self.verbose = verbose
        self._maximize = maximize
        
    def __repr__(self):
        return 'Arguments: (num_samp={}, resamp_ratio={}, maximize={}, verbose={})'.format(
            self.num_samp,
            self.resamp_ratio,
            self._maximize,
            self.verbose
        )
        
    @staticmethod
    def from_yaml(yaml_path):
        """
        this function creates a BaseOptimizerOptions object from a yaml file
        inputs:
            • yaml_path (str): the path to the yaml file
        returns:
            • BaseOptimizerOptions: the options object
        """
        config = yaml.safe_load(open(yaml_path))
        return BaseOptimizerOptions(
            config["num_samp_max"], 
            config["resamp_ratio"], 
            config["maximize"], 
            config["verbose"]
        )
        

class BaseOptimizer:
    """
    This class describes a voronoi optimizer.
    inputs:
        • objective (callable): A callable that evaluates a single set of parameters and returns its fitness.
        • limits (list[tuple]): A list of tuples containing the ranges of each parameter's search space.
        • args (BaseOptimizerOptions): A set of arguments describing the optimizer's behaviour.
    """
    def __init__(
        self, 
        objective:callable,
        limits:list[tuple], 
        lookup_table,
        args:BaseOptimizerOptions
    ):
        
        assert objective != None, "objective must be provided."
        assert limits != None, "Limits must be provided."
        self._objective = objective
        self._limits = limits
        self.args = args
        self._iter = 0
        self._queue = []
        self._elites = []
        self._samples = []
        self._normalized_params = torch.tensor([])
        self._num_dim = len(limits)
        self._param_min = torch.tensor([x[0] for x in limits])
        self._param_max = torch.tensor([x[1] for x in limits])
        self._param_rng = torch.tensor([x[1]-x[0] for x in limits])
        self._comparator = max if self.args._maximize else min
        self.lookup_table = lookup_table
    
    
    def get_best(self):
        """
        this function returns the best sample
        returns:
            • BaseOptimizerSample: the best sample
        """
        return self._elites[0]
    
    
    def step(self, count):
        """
        this function performs a single step of the optimization process
        inputs:
            • count (int): the number of evaluations performed so far
        returns:
            • int: the number of evaluations performed so far
        """
        if len(self._normalized_params) == 0:
            self._random_population()
        else:
            self._neighborhood_population()
        while self._queue:
            params = self._queue.pop()
            count = self._evaluate_params(params, count)
        if self.args.verbose:
            print(self)
        self._iter += 1
        return count
        
    
    def fit(self, eps=10, cls=True):
        """
        this function performs the optimization process for a given number of epochs
        inputs:
            • eps (int, optional): the number of epochs to run. Defaults to 10.
            • cls (bool, optional): clears the console output. Defaults to True.
        returns:
            • BaseOptimizerSample: the best sample
        """
        # clear command line output
        if cls:
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")

        # log initial params
        print(f"{'='*100}")
        print(f"Started training loop training loop... epoch: {self._iter}")
        print(f"{self.args}")
        print(f"{'-'*100}\n")
        
        # calls step for eps iterations
        for _ in range(eps):
            self.step()
            
        print(f"\n{'='*100}\n")
        best = self.get_best()
        return best
            
    
    def _evaluate_params(self, params, count):
        """
        this function evaluates a set of parameters
        inputs:
            • params (torch.tensor): the parameters to evaluate
            • count (int): the number of evaluations performed so far
        returns:
            • int: the number of evaluations performed so far
        """
        score = self._objective(params, self.lookup_table)
        new_sample = self._create_new_sample(params, score, self._iter,len(self._samples))
        self._add_to_samples(new_sample)
        num_resamp = int(self.args.num_samp * self.args.resamp_ratio)
        # if better than worst elite, add to elite set
        if len(self._elites) == 0 or self._comparator(score, self._elites[-1]._score) == score or len(self._elites) < num_resamp:
            self._add_to_elites(new_sample)
        if score == 0:
            count += 1
        return count
        
    
    def _add_to_samples(self, new_sample):
        """
        this function adds a sample to the sample list
        inputs:
            • new_sample (BaseOptimizerSample): the sample to add
        """
        assert issubclass(new_sample.__class__, BaseOptimizerSample), "New sample must derive from BaseOptimizerSample..."
        # add to normalized parameters
        if len(self._normalized_params) > 0:
            self._normalized_params = torch.vstack((self._normalized_params, (new_sample._params - self._param_min) / self._param_rng))
        else:
            self._normalized_params = ((new_sample._params - self._param_min) / self._param_rng).view(1, self._num_dim)
        self._samples.append(new_sample)
        
    
    def _create_new_sample(self, params, score, iter, index):
        """
        this function creates a new sample. Used to conveniently set what type of sample is used.
        inputs:
            • params (torch.tensor): the parameters of the sample
            • score (float): the score of the sample
            • iter (int): the iteration the sample was created
            • index (int): the index of the sample
        returns:
            • BaseOptimizerSample: the new sample
        """
        return BaseOptimizerSample(params, score, iter, index)
        
    
    def _add_to_elites(self, sample):
        """
        this function adds a sample to the elite list
        inputs:
            • sample (BaseOptimizerSample): the sample to add
        """
        resamps = self.args.num_samp * self.args.resamp_ratio
        # keep elites at length self._num_samp
        if len(self._elites) >= resamps:
            self._elites.pop(-1)
            
        # append to elites list
        bisect.insort_left(self._elites, sample, key=lambda x: -1 * x._score if self.args._maximize else x._score)
        
    
    def _random_population(self):
        """
        this function generates a random population of samples
        """
        for _ in range(self.args.num_samp):
            self._queue.append(self._random_sample()) 
            
    
    def _neighborhood_population(self):
        """
        this function generates a neighborhood population of samples
        """
        num_resamp = int(self.args.num_samp * self.args.resamp_ratio)
        for i in range(self.args.num_samp):
            index = i % num_resamp
            self._neighborhood_sample(index)
            
    
    def _random_sample(self):
        """
        this function generates a random sample
        returns:
            • torch.tensor: the random sample
        """
        # generates a random sample in the samplespace
        return torch.rand(self._num_dim)*self._param_rng + self._param_min 
    
    
    def _neighborhood_sample(self, elite_index):
        """
        this function generates a neighborhood sample and adds it to the queue
        inputs:
            • elite_index (int): the index of the elite to generate a neighborhood sample from
        """
        kk = elite_index
        curr_elite = self._elites[kk]
        sample_index = curr_elite._index
        vk = self._normalized_params[sample_index]
        vj = torch.vstack((self._normalized_params[:sample_index,:], self._normalized_params[sample_index+1:,:]))
        xx = copy.deepcopy(vk)
        # get initial distance to ith-axis (where i == 0)
        d2ki = 0.0
        d2ji = torch.sum(torch.square(vj[:,1:] - xx[1:]), axis=1)
        # random step within voronoi polygon in each dimension
        for ii in range(self._num_dim):
            # find limits of voronoi polygon
            xji = 0.5*(vk[ii] + vj[:,ii] + (d2ki - d2ji)/(vk[ii] - vj[:,ii]))
            low_filter = xji[xji <= xx[ii]]
            high_filter = xji[xji >= xx[ii]]
            if len(low_filter) > 0:
                low = max(0.0, torch.max(low_filter))
            else: 
                low = 0.0
            if len(high_filter) > 0:
                high = min(1.0, torch.min(high_filter))
            else: 
                high = 1.0
            # random move within voronoi polygon
            xx[ii] = uniform(low, high)
            # update distance to next axis
            if ii < (self._num_dim - 1):
                d2ki += (torch.square(vk[ii  ] - xx[ii  ]) - 
                        torch.square(vk[ii+1] - xx[ii+1]))
                d2ji += (torch.square(vj[:,ii  ] - xx[ii  ]) - 
                        torch.square(vj[:,ii+1] - xx[ii+1]))
        
        xx = xx * self._param_rng + self._param_min # un-normalize
        self._queue.append(xx)

    
    def get_elite_stddev(self):
        """
        this function returns the standard deviation of the elite scores
        """
        return np.std([elite._score for elite in self._elites])
    
    
    def get_sample_dataframe(self):
        """
        this function returns a pandas dataframe of the samples
        """
        return pd.DataFrame.from_records([sample.to_dict() for sample in self._samples])
    
    
    def plot(self):
        """
        this function plots the samples
        """
        df = self.get_sample_dataframe()
        sorted = df.sort_values("score", ascending=False if self.args._maximize else True)
        best = sorted.drop_duplicates(subset=["iter"], ignore_index=True)
        best = best.reset_index()
        plt.plot(best["iter"], best["score"])
        plt.show()
        
    
    def __repr__(self):
        try:
            out = '{}(iteration={}, samples={}, best={:.20f})'.format(
            self.__class__.__name__,
            self._iter,
            len(self._normalized_params),
            self.get_best()._score)
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out
