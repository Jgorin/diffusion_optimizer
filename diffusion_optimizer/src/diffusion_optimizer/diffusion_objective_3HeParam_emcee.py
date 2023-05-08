from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.utils.utils_3HeParam_emcee import forwardModelKinetics
import torch
import math as math 
import numpy as np

class DiffusionObjective(Objective):
    
    # override evaluate function
    def __call__(self, X): #__call__ #evaluate
        
        # Un-normalize moles
             
        #X[0] = 50+X[0]*100
        
        data = self.dataset
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        # Below here will eventually get turned into a function
        # Code written by Marissa Tremblay and modified/transcribed into Python by Drew Gorin.
        #Last modified 1.2023.

        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as the
        # sum of absolute differences between the observed and modeled release
        # fractions over all steps.
        
        #Time both of these
        X = torch.as_tensor(X)
        total_moles = X[0]
        X = X[1:]
        

        
        # Forward model the results so that we can calculate the misfit.
        fwdModelResults = forwardModelKinetics(X,data,self.lookup_table)

 
        Fi_MDD = fwdModelResults[2] # Gas fraction released for each heating step in model experiment
        

        
        # Calculate the fraction released for each heating step in the modeled experiment
        trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
        trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-1),trueFracMDD),dim=-1)
       
        # Punish the model if the cumulative gas fraction reaches 1 before the last release step. 
        # II CURRENTLY HAVE THIS OFF, BUT IT SEEMS LIKE THIS MIGHT BE USEFUL AT SOME POINT. Though,
        # maybe it doesn't matter because of my not_released_flag

        exp_moles = torch.tensor(data.M)


        # Scale by chosen number of moles
        moles_MDD = trueFracMDD * total_moles

        misfit = torch.sum(((exp_moles-moles_MDD)**2)/(data.uncert**2))
        #misfit = (exp_moles-moles_MDD)**2
        if torch.sum(torch.isnan(misfit))>0:
            print(r"oops! {orch.sum(((exp_moles-torch.zeros([1,len(exp_moles)]))**2)/(data.uncert**2))}")
            return torch.sum(((exp_moles-torch.zeros([1,len(exp_moles)]))**2)/(data.uncert**2))

        return misfit 



    def check_flags(self,X):
        data = self.dataset

        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        # Below here will eventually get turned into a function
        # Code written by Marissa Tremblay and modified/transcribed into Python by Drew Gorin.
        #Last modified 1.2023.

        # This function calculates the fraction of gas released from each domain
        # in an MDD model during the heating schedule used in the diffusion
        # experiment. Then the fractions released from each domain are combined in
        # proportion to one another as specified by the MDD model, and the
        # diffusivity of each step is calculated. A residual is calculated as the
        # sum of absolute differences between the observed and modeled release
        # fractions over all steps.
        
        #Time both of these
        X = torch.as_tensor(X)
        total_moles = X[0]
        X = X[1:]
        

        # Grab the other parameters from the input
 

        # Report high misfit values if conditions are not met

        # if any LnD0aa is greater than the previous, punish the model. This reduces model search space.
        
        # First, calculate the difference between each set of LnD0aa
        # lnd0_off_counter = 0
        # for i in range(len(lnD0aa)-1):
        #     if lnD0aa[i+1]>lnD0aa[i]:
        #         lnd0_off_counter += (torch.abs((lnD0aa[i+1]-lnD0aa[i])/lnD0aa[i+1]))*10**17

        # # Now, return this misfit value if any of the LnD0aa values were greater than their previous.
        # # The idea is that the model is punished more greatly for having more of these set incorrectly. 
        # # Additionally, the magnitude is also implicitly considered.
        # for i in range(len(lnD0aa)-1):
        #     if lnD0aa[i+1]>lnD0aa[i]:
        #         return lnd0_off_counter.item()
        
        # Forward model the results so that we can calculate the misfit.
        fwdModelResults = forwardModelKinetics(X,data,self.lookup_table)

        Fi_MDD = fwdModelResults[2]
        # Calculate the fraction released for each heating step in the modeled experiment
        trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
        trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-1),trueFracMDD),dim=-1)
        
        #If all the gas released too early


        if torch.sum(trueFracMDD) == 0:
            return -np.inf
    
        elif fwdModelResults[-1] >0:
            return -np.inf
        
        elif torch.sum(torch.round(Fi_MDD[0:-2],decimals=6) ==1) > 1:
            return -np.inf
    
        else:
            return 0