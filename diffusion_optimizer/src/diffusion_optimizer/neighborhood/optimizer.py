# Code written by Josh Gorin to allow for an additional input into the neighborhood optimizer function
import neighborhood as nbr
from diffusion_optimizer.neighborhood.objective import Objective
import bisect

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
        self.objective = objective
        
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
                res = { 'param': param, 'result': result, 'iter': self._iter }
                bisect.insort_left(self._sample, res, key=lambda x: x['result'])
                
            self._iter += 1
            if self._verbose:
                print(self)
                
    def __repr__(self):
        try:
            out = '{}(iteration={}, samples={}, best={:.20f})'.format(
            self.__class__.__name__,
            self._iter,
            len(self._sample),
            self._sample[0]['result'])
        except IndexError:
            out = '{}(iteration=0, samples=0, best=None)'.format(self.__class__.__name__)
        return out