# Code written by Josh Gorin to allow for an additional input into the neighborhood optimizer function
from re import S
import neighborhood as nbr
from diffusion_optimizer.neighborhood.objective import Objective
import numpy as np
from random import sample, uniform
from diffusion_optimizer.neighborhood.neighborhood_sampler import SampleManager, Sample
import math

def adaptive_decay_greedy_epsilon(error, error_decay_rate, initial_epsilon):
    return initial_epsilon * np.exp(-error_decay_rate * error)

def adaptive_sample_count(current_error, rate, intercept):
    return int(min(1000, intercept + (math.exp(rate * current_error) - 1)))

def exponential_interpolation(min_val, max_val, rate, value):
    if value > 3:

        return max_val


    #interpolated_value = np.max([1,max_val*np.exp(-rate*(1/value)) ])
    interpolated_value = min_val * (max_val / min_val) ** (rate * value)

    return interpolated_value

class OptimizerOptions:
    def __init__(
        self,
        #num_samp_range:range,
        num_samp_decay:float,
        resamp_to_samp_ratio:float,
        #epsilon_range:range,
        epsilon_growth:float,
        num_samp_max:int,
        num_samp_ratio:float,
        epsilon_range_ratio: float,
        epsilon_max:float

    ):
        self._num_samp_range = [int(num_samp_max*num_samp_ratio), num_samp_max]
        self._epsilon_range = [epsilon_max*epsilon_range_ratio,epsilon_max]
        self._num_samp_decay = num_samp_decay
        self._resamp_to_samp_ratio = resamp_to_samp_ratio
        #self._epsilon_range = epsilon_range
        self._epsilon_growth = epsilon_growth
        


class Optimizer(nbr.Searcher):
    def __init__(
        self, 
        objective:Objective, 
        limits:list[tuple], 
        names:list[str],
        options:OptimizerOptions,
        maximize:bool=False,
        verbose:bool=True
    ):
        # set initial parameter values
        if names == None:
            names = []
        initial_samp_num = options._num_samp_range[1]
        
        # call abstract abstantiation
        super().__init__(
            objective, 
            limits, 
            initial_samp_num, 
            int(initial_samp_num * options._resamp_to_samp_ratio), 
            names=names, 
            maximize=maximize, 
            verbose=verbose
        )
        
        self._objective = objective
        self._options = options
        self.sample_manager = SampleManager(num_samples=initial_samp_num, maximize=maximize)
        self.param_list = []
        self.std = []
 
    def update(self, num_iter=10):
        """
        tweaked from original codebase to pass in training data into objective function
        """
        
        for ii in range(num_iter):

            # If the first iteration take a random sample
            if len(self.sample_manager._samples) == 0:
                # set initial samples/resamples
                self.curr_num_samp = self._options._num_samp_range[1]
                self.curr_num_resamp = int(self.curr_num_samp * self._options._resamp_to_samp_ratio)
                
                # update elite size in samplemanager
                self.sample_manager.set_num_samples(self.curr_num_samp)
                
                # set initial epsilon
                self.current_epsilon_threshold = self._options._epsilon_range[0]
                self._random_sample(self.curr_num_samp)
            
            else:
                self.current_epsilon_threshold = exponential_interpolation(self._options._epsilon_range[0], self._options._epsilon_range[1], -1 * self._options._epsilon_growth, self.get_best()._res)
                self._neighborhood_sample(self.current_epsilon_threshold, self.curr_num_samp, self.curr_num_resamp)
                        
            # execute forward model for all samples in queue
            while self._queue:
                param = self._queue.pop()
                result = self._objective.evaluate(param)
                self.sample_manager.add_sample(Sample(result, self._iter, param, len(self.sample_manager._samples)))
                
            self.curr_num_samp = min(self._options._num_samp_range[1], int(exponential_interpolation(self._options._num_samp_range[0], self._options._num_samp_range[1], self._options._num_samp_decay, self.get_best()._res)))
            self.curr_num_resamp = max(1, int(self._options._resamp_to_samp_ratio * self.curr_num_samp))
            self.sample_manager.set_num_samples(self.curr_num_samp)
            self._iter += 1
            #param_list = self.param_list.append([self.sample_manager._elites[0]._param])
            self.std.append(self.sample_manager.get_std())
            if self._verbose:
                print(self)

    
    def _random_single_sample(self):
        return np.random.rand(self._num_dim)*self._param_rng + self._param_min
    
    def _random_sample(self, curr_num_samp):
        """Generate uniform random sample for initial iteration"""
        for ii in range(curr_num_samp):
            pt  = np.random.rand(self._num_dim)*self._param_rng + self._param_min
            self._queue.append(pt)
                
    def _neighborhood_sample(self, epsilon_threshold, curr_num_samp, curr_num_resamp):
        try:
            all_samps = np.array([x._param for x in self.sample_manager._samples])
            all_samps = (all_samps - self._param_min)/self._param_rng
            
            elites = self.sample_manager._elites
            elite_samps = np.array([e._param for e in elites])
            elite_samps = (elite_samps - self._param_min)/self._param_rng

            for ii in range(curr_num_samp):
                epsilon = uniform(0, 1)
                if epsilon <= epsilon_threshold:
                    self._queue.append(self._random_single_sample())
                    
                kk = ii % curr_num_resamp
                curr_elite = elites[kk]
                vk = elite_samps[kk,:]
                vj = np.delete(all_samps, curr_elite._index, 0)
                
                xx = vk.copy()
                
                # get initial distance to ith-axis (where i == 0)
                d2ki = 0.0
                d2ji = np.sum(np.square(vj[:,1:] - xx[1:]), axis=1)
                # random step within voronoi polygon in each dimension
                for ii in range(self._num_dim):
                    
                    # find limits of voronoi polygon
                    xji = 0.5*(vk[ii] + vj[:,ii] + (d2ki - d2ji)/(vk[ii] - vj[:,ii]))
                    try:
                        low = max(0.0, np.max(xji[xji <= xx[ii]]))
                    except ValueError: # no points <= current point
                        low = 0.0
                    try:
                        high = min(1.0, np.min(xji[xji >= xx[ii]]))
                    except ValueError: # no points >= current point
                        high = 1.0

                    # random move within voronoi polygon
                    xx[ii] = uniform(low, high)
                    
                    # update distance to next axis
                    if ii < (self._num_dim - 1):
                        d2ki += (np.square(vk[ii  ] - xx[ii  ]) - 
                                np.square(vk[ii+1] - xx[ii+1]))
                        d2ji += (np.square(vj[:,ii  ] - xx[ii  ]) - 
                                np.square(vj[:,ii+1] - xx[ii+1]))
                        
                # update queue
                xx = xx*self._param_rng + self._param_min # un-normalize
                self._queue.append(xx)
        except:
            raise Exception(f"{len(self.sample_manager._elites)}   {self.curr_num_samp}   {self.curr_num_resamp}")
    
    def get_best(self):
        return self.sample_manager._elites[0]      
                
    def __repr__(self):
        try:
            out = '{}(iteration={}, samples={}, best={:.20f})'.format(
            self.__class__.__name__,
            self._iter,
            len(self.sample_manager._samples),
            self.get_best()._res)
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out + f" {self.current_epsilon_threshold} {self.curr_num_samp}  {self.curr_num_resamp} {self.sample_manager.get_std()} {self.sample_manager._elites[0]._param}"
    
    def _validate_args(args):
        pass