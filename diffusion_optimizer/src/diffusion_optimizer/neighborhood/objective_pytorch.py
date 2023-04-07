from diffusion_optimizer.neighborhood.dataset import Dataset
import torch 
import pickle 
import os

file_dir = os.path.dirname(os.path.abspath(__file__))

class Objective:
    def __init__(self, data:Dataset,omitValueIndices:list, pickle_path="../lookup_table.pkl" ):
        self.dataset = data
        self.omitValueIndices = torch.sub(torch.tensor(omitValueIndices),torch.tensor([1])) #Drew added this.. I think this is what you do?
        self.lookup_table = torch.tensor(pickle.load(open(pickle_path,'rb')))
    def evaluate(self, X):
        raise NotImplementedError("Objective must implpement an 'evaluate' method.")