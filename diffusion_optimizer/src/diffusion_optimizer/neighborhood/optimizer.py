# Code written by Josh Gorin to allow for an additional input into the neighborhood optimizer function
import neighborhood as nbr
from diffusion_optimizer.neighborhood.objective import Objective
import bisect
import numpy as np
from random import uniform

def sig(x):
     return 1/(1 + np.exp(-x))
 

def lerp(a: float, b: float, t: float):
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    return (1 - t) * a + t * b


def inv_lerp(a: float, b: float, v: float):
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)


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
        self.objective = objective
        self.epsilon_threshold = epsilon_threshold
        
    def update(self, num_iter=10):
        """
        tweaked from original codebase to pass in training data into objective function
        """
        total_iterations = self._iter + num_iter
        initial_iterations = self._iter
        resamp_ratio = self._num_resamp / self._num_samp
        
        for ii in range(num_iter):
            # udpate greedy epsilon
            self.current_epsilon_threshold = self.epsilon_threshold - ((inv_lerp(0, total_iterations, ii + initial_iterations)) * self.epsilon_threshold)
            curr_num_samp = int(self._num_samp - lerp(0, self._num_samp * 0.5, (initial_iterations + ii) / total_iterations))
            curr_num_resamp = int(resamp_ratio * curr_num_samp)
            self.curr_num_samp = curr_num_samp
            self.curr_num_resamp = curr_num_resamp
            
            # If the first iteration take a random sample
            if not self._sample:
                self._random_sample(curr_num_samp)
            else:
                self._neighborhood_sample(self.current_epsilon_threshold, curr_num_samp, curr_num_resamp)
                        
            # execute forward model for all samples in queue
            while self._queue:
                param = self._queue.pop()
                result = self._objective(param)
                res = { 'param': param, 'result': result, 'iter': self._iter }
                bisect.insort_left(self._sample, res, key=lambda x: x['result'])
                
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
        """Generate random samples in best Voronoi polygons"""
        
        vv = np.array([x['param'] for x in self._sample])
        vv = (vv - self._param_min)/self._param_rng # normalize
        
        for ii in range(curr_num_samp):
            
            epsilon = uniform(0, 1)
            if epsilon <= epsilon_threshold:
                self._queue.append(self._random_single_sample())
            else:
                # get starting point and all other points as arrays
                kk = ii % curr_num_resamp  # index of start point            
                vk = vv[kk,:]
                vj = np.delete(vv, kk, 0)
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
            len(self._sample),
            self._sample[0]['result'])
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out + f" {self.current_epsilon_threshold}, {self.curr_num_samp}, {self.curr_num_resamp}"