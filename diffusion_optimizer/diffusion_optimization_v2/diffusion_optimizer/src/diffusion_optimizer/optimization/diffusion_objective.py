import pandas as pd
import torch
import warnings
from diffusion_optimizer.optimization.calc import forwardModelKinetics

class DiffusionDataset(pd.DataFrame):
    """
    An extension of a pandas dataframe to allow for saving of tensors

    Args:
        pd (pd.DataFrame): the dataframe to wrap
    """
    def __init__(self, df:pd.DataFrame):
        assert df["TC"] is not None, "given dataset does not contain TC parameter."
        assert df["thr"] is not None, "given dataset does not contain thr parameter."
        assert df["ln(D/a^2)"] is not None, "given dataset does not contain ln(D/a^2) parameter."
        assert df["Fi"] is not None, "given dataset does not contain Fi parameter."
        super().__init__(data=df, index=df.index, columns=df.columns)
        
        # temporarily ignores atribute setting warning from base dataframe class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            self.np_TC = torch.tensor(self["TC"].values) 
            self.np_thr = torch.tensor(self["thr"].values) 
            self.np_lnDaa = torch.tensor(self["ln(D/a^2)"].values) 
            self.np_Fi_exp = torch.tensor(self["Fi"].values) 


class DiffusionObjective:
    def __init__(self, dataset:pd.DataFrame,):
        self._dataset = DiffusionDataset(dataset) 
    
    def __call__(self, X, lookup_table):
        """
        This function calculates the fitness of a given set of parameters.
        Args:
            • X (tensor): current parameters for evaluation

        Returns:
            • float: the calculated error of the inputted parameters
        """
        
        data = self._dataset
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        # Unpack the parameters and spit out a high misfit value if constraints are violated
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        
        temp = X[1:]
        lnD0aa = temp[0:ndom]
        fracstemp = temp[ndom:]
        if torch.sum(fracstemp) >1:
            return (torch.sum(fracstemp)-1)*10**17

                
        #This punishes the model for proposing input fractions that don't add up to 1.0
        # if torch.sum(fracstemp,axis=0,keepdim = True)>0.999:
        #     return torch.sum(fracstemp,axis=0,keepdim = True).item()*100

        # Now that we know the fracs add up to one, calculate the fraction for the last domain.
        # This is determined by the other fractions.
        sumTemp = (1-torch.sum(fracstemp,axis=0,keepdim = True))
        fracs = torch.concat((fracstemp,sumTemp),dim=-1)

        # Report high misfit values if conditions are not met

        # if any LnD0aa is greater than the previous, punish the model. This reduces model search space.
        
        # Forward model the results so that we can calculate the misfit.
        fwdModelResults = forwardModelKinetics(X,data,lookup_table)

        # Parameters that need to be read in
        TC = data.np_TC # Temperature (Celcius)
        thr = data.np_thr # Time in hours
        if thr[1]>10: # If time > 10, then units are likely in minutes, not hours-- convert to minutes.
            thr = thr/60
        lnDaa = data.np_lnDaa # LnDaa (1/s)
        Fi_exp = data.np_Fi_exp #Gas fraction released for each heating step in experiment
        Fi_MDD = fwdModelResults[2] # Gas fraction released for each heating step in model experiment
        not_released_flag = fwdModelResults[-1] # A flag for when the modeled results don't add to 1 before they get renormalized. This gets added to the misfit.

        # Calculate the Fraction released for each heating step in the real experiment
        TrueFracFi = (Fi_exp[1:]-Fi_exp[0:-1]) 
        TrueFracFi = torch.concat((torch.unsqueeze(Fi_exp[0],dim=-1),TrueFracFi),dim=-1)
        
        # Calculate the fraction released for each heating step in the modeled experiment
        trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
        trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-1),trueFracMDD),dim=-1)
       
        # Punish the model if the cumulative gas fraction reaches 1 before the last release step. 
        # II CURRENTLY HAVE THIS OFF, BUT IT SEEMS LIKE THIS MIGHT BE USEFUL AT SOME POINT. Though,
        # maybe it doesn't matter because of my not_released_flag

        #  TWO THINGS HERE. 
        # 1. I SHOULD EXPERIMENT WITH THE NUMBER OF DECIMALS TO ROUND TO.
        # 2. I SHOULD CONSIDER IMPLEMENTING A PUNISHMENT WITH A GRADIENT INSTEAD OF A FLAT PUNISHMENT.
        indicesForPunishment = torch.round(Fi_MDD[0:-2],decimals=4) == 1
        ran_out_too_early = torch.tensor(0)
        if sum(indicesForPunishment>0):
            ran_out_too_early = sum(indicesForPunishment)

            pass


        # Sometimes the forward model predicts kinetics such that ALL the gas would have leaked out during the irradiation and lab storage.
        # In this case, we end up with trueFracMDD == 0, so we should return a high misfit because we know this is not true, else we wouldn't
        # have measured any He in the lab. 
        if torch.sum(trueFracMDD) == 0:

            return 10**10
        
        output = (not_released_flag + ran_out_too_early).item()
        return output