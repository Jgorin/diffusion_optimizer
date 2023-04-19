import pandas as pd
import numpy as np
import warnings

class Dataset(pd.DataFrame):
    def __init__(self, data:pd.DataFrame):
        super().__init__(data=data, index=data.index, columns=data.columns)
        assert self["TC"] is not None, "given dataset does not contain TC parameter."
        assert self["thr"] is not None, "given dataset does not contain thr parameter."
        assert self["ln(D/a^2)"] is not None, "given dataset does not contain ln(D/a^2) parameter."
        assert self["Fi"] is not None, "given dataset does not contain Fi parameter."
        # temporarily ignores attribute setting warning from base dataframe class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            self.np_TC = np.array(self["TC"].values) 
            self.np_thr = np.array(self["thr"].values) 
            self.np_lnDaa = np.array(self["ln(D/a^2)"].values) 
            self.np_Fi_exp = np.array(self["Fi"].values)
            self.uncert = np.array(self["Fi uncertainty"].values)
            self.expected_y = None

    def set_expected_y(self,y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.expected_y = y