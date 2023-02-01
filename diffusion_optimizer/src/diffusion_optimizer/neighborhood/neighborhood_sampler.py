import bisect

class Sample:
    def __init__(self, res:float, iter:int, param:list[float], index: int):
        self._res = res
        self._iter = iter
        self._param = param
        self._index = index
    
    def is_better(self, sample, maximize):
        comparator = max if maximize == True else min
        return comparator(self._res, sample._res) == self._res

class SampleManager:
    def __init__(self, samples:list[Sample]=None, elites:list[Sample]=None, num_samples:int=10, maximize=False):
        if samples is None:
            samples = []  
        if elites is None:
            elites = []
        
        self._samples = samples
        self._elites = elites
        self._num_samples = num_samples
        self._maximize = maximize
    
    def is_elite_set_full(self):
        return len(self._elites) >= self._num_samples

    def add_sample(self, sample:Sample):        
        # add to elites if better than current worst elite or if elite set isn't full
        if len(self._elites) == 0 or len(self._elites) < self._num_samples or sample.is_better(self._elites[-1], self._maximize):
            
            # if full, remove worst elite to be replaced
            if len(self._elites) >= self._num_samples:
                self._elites.pop(-1)
            
            # insert into elite set
            bisect.insort_left(self._elites, sample, key=lambda x: -1 * x._res if self._maximize else x._res)
        
        self._samples.append(sample)

def main():
    sm = SampleManager(num_samples=2, maximize=False)
    
    test_samples = []
    for i in range(0, 20):
        smp = Sample(i, 0, [i, i, i])
        test_samples.append(smp)
    for sample in test_samples:
        sm.add_sample(sample)
        
    print(sm._elites[0].samples_to_string())

if __name__ == "__main__":
    main()