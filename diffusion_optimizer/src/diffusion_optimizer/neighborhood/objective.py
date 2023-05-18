from diffusion_optimizer.neighborhood.dataset import Dataset
import torch 
import pickle 
import os
import torch

file_dir = os.path.dirname(os.path.abspath(__file__))

class Objective:
    def __init__(self, data:Dataset, time_add: torch.Tensor, temp_add: torch.Tensor,pickle_path="../lookup_table.pkl",omitValueIndices = []):
        self.dataset = data
        self.lookup_table = pickle.load(open(pickle_path,'rb'))
        self.time_add = time_add
        self.temp_add = temp_add
        self.omitValueIndices = omitValueIndices
        if self.dataset.np_thr[0] >15:
            time = self.dataset.np_thr*60 #time in seconds
        else:
            time = self.dataset.np_thr*3600
            
        self.tsec = torch.cat([time_add,time])
        self.TC = torch.cat([temp_add,self.dataset.np_TC])
        

    def evaluate(self, X):
        raise NotImplementedError("Objective must implpement an 'evaluate' method.")