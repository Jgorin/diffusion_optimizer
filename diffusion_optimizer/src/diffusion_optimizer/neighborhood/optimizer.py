# Code written by Josh Gorin to allow for an additional input into the neighborhood optimizer function
import neighborhood as nbr
import pandas as pd
from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.neighborhood.dataset import Dataset


class Optimizer(nbr.Searcher):
    
    def __init__(self, objective:Objective, limits, num_samp:int, 
                 num_resamp:int, names, maximize:bool=False, verbose:bool=True):
        
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
        
    def update(self, num_iter=10):
        """
        tweaked from original codebase to pass in training data into objective function
        """

        for ii in range(num_iter):
            
            # If the first iteration, or every 10th... take a random sample
            if not self._sample or ii % 10 == 0:
                self._random_sample()
            else:
                self._neighborhood_sample()
                        
            # execute forward model for all samples in queue
            while self._queue:
                param = self._queue.pop()
                result = self._objective(param)
                self._sample.append({
                    'param': param,
                    'result': result,
                    'iter': self._iter
                    })
             
            # prepare for next iteration
            self._sample.sort(key=lambda x: x['result'], reverse=self._maximize)
            self._iter += 1
            if self._verbose:
                print(self)