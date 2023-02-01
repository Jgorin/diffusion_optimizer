# Code written by Josh Gorin to allow for an additional input into the neighborhood optimizer function
from re import S
import neighborhood as nbr
from diffusion_optimizer.neighborhood.objective import Objective
import bisect
import numpy as np
from random import uniform
from diffusion_optimizer.neighborhood.neighborhood_sampler import SampleManager, Sample

def adaptive_decay_greedy_epsilon(error, error_decay_rate, initial_epsilon):
    return initial_epsilon * np.exp(-error_decay_rate * error)

class Optimizer(nbr.Searcher):
    
    def __init__(self, objective:Objective, limits, num_samp:int, 
                 num_resamp:int, epsilon_threshold, names, maximize:bool=False, verbose:bool=True):
        
        if names == None:
            names = []
            
        super().__init__(
            objective.evaluate, 
            limits, 
            num_samp, 
            num_resamp, 
            names=names, 
            maximize=maximize, 
            verbose=verbose
        )
        self.epsilon_threshold = epsilon_threshold
        self.sample_manager = SampleManager(num_samples=num_samp, maximize=maximize)
        
    def update(self, num_iter=10):
        """
        tweaked from original codebase to pass in training data into objective function
        """
        for ii in range(num_iter):
            # udpate greedy epsilon
            
            # If the first iteration take a random sample
            if not self.sample_manager._samples:
                self.current_epsilon_threshold = 0
                self._random_sample(self._num_samp)
            else:
                self.current_epsilon_threshold = adaptive_decay_greedy_epsilon(self.sample_manager._elites[0]._res, 4, self.epsilon_threshold)
                self._neighborhood_sample(self.current_epsilon_threshold, self._num_samp, self._num_resamp)
                        
            # execute forward model for all samples in queue
            while self._queue:
                param = self._queue.pop()
                result = self._objective(param)
                self.sample_manager.add_sample(Sample(result, self._iter, param, len(self.sample_manager._samples)))
                
            self._iter += 1
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
                
    def __repr__(self):
        try:
            out = '{}(iteration={}, samples={}, best={:.20f})'.format(
            self.__class__.__name__,
            self._iter,
            len(self.sample_manager._samples),
            self.sample_manager._elites[0]._res)
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out + f" {self.current_epsilon_threshold}"