from diffusion_optimizer.neighborhood.dataset import Dataset
import torch 

class Objective:
    def __init__(self, data:Dataset,omitValueIndices:list):
        self.dataset = data
        self.omitValueIndices = torch.sub(torch.tensor(omitValueIndices),torch.tensor([1])) #Drew added this.. I think this is what you do?
    def evaluate(self, X):
        raise NotImplementedError("Objective must implpement an 'evaluate' method.")