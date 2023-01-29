import pandas as pd
import torch

class Dataset(pd.DataFrame):
    def __init__(self, data:pd.DataFrame):
        super().__init__(data=data, index=data.index, columns=data.columns)
        assert self["TC"] is not None, "given dataset does not contain TC parameter."
        assert self["thr"] is not None, "given dataset does not contain thr parameter."
        assert self["ln(D/a^2)"] is not None, "given dataset does not contain ln(D/a^2) parameter."
        assert self["Fi"] is not None, "given dataset does not contain Fi parameter."

        self.np_TC = torch.tensor(self["TC"].values) 
        self.np_thr = torch.tensor(self["thr"].values) 
        self.np_lnDaa = torch.tensor(self["ln(D/a^2)"].values) 
        self.np_Fi_exp = torch.tensor(self["Fi"].values) 