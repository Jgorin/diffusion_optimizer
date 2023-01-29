from diffusion_optimizer.neighborhood.dataset import Dataset

class Objective:
    def __init__(self, data:Dataset):
        self.dataset = data
    
    def evaluate(self, X):
        raise NotImplementedError("Objective must implpement an 'evaluate' method.")