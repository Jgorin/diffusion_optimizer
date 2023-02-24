from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.utils.utils import forwardModelKinetics
import torch

class DiffusionObjective(Objective):
    
    # override evaluate function
    def evaluate(self, X):
        data = self.dataset
        omitValueIndices = self.omitValueIndices
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
        #X = torch.from_numpy(X)
        X = torch.as_tensor(X)
        # Unpack the parameters and spit out a high misfit value if constraints are violated
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        
        temp = X[1:]
        lnD0aa = temp[0:ndom]
        fracstemp = temp[ndom:]

        #This punishes the model for creating sums that don't add up to one
        if torch.sum(fracstemp,axis=0,keepdim = True)>0.999:
            

            return torch.sum(fracstemp,axis=0,keepdim = True).item()*100


        sumTemp = (1-torch.sum(fracstemp,axis=0,keepdim = True))
        
        fracs = torch.concat((fracstemp,sumTemp),dim=-1)

        # Report high misfit values if conditions are not met

        lnd0_off_counter = 0
        for i in range(len(lnD0aa)-1):
            if lnD0aa[i+1]>lnD0aa[i]:
                lnd0_off_counter = lnd0_off_counter + (torch.abs((lnD0aa[i+1]-lnD0aa[i])/lnD0aa[i+1]))*10**17

        for i in range(len(lnD0aa)-1):
            if lnD0aa[i+1]>lnD0aa[i]:

                return lnd0_off_counter.item()
        
    
        fwdModelResults = forwardModelKinetics(X,data)

        # Parameters that need to be read in

        TC = data.np_TC #data["TC"].to_numpy()
        thr = data.np_thr#["thr"].to_numpy()
        lnDaa = data.np_lnDaa#["ln(D/a^2)"].to_numpy()
        Fi_exp = data.np_Fi_exp#["Fi"].to_numpy()
        Fi_MDD = fwdModelResults[2]#fwdModelResults["Fi_MDD"].to_numpy()
        
        #Calculate the Fraction released for each heating step
        TrueFracFi = (Fi_exp[1:]-Fi_exp[0:-1])
        TrueFracFi = torch.concat((torch.unsqueeze(Fi_exp[0],dim=-1),TrueFracFi),dim=-1)
        
        trueFracMDD = Fi_MDD[1:]-Fi_MDD[0:-1]
        trueFracMDD = torch.concat((torch.unsqueeze(Fi_MDD[0],dim=-1),trueFracMDD),dim=-1)
        # Calculate the L1 loss 

        
        #Remove user-specified values to be omitted from the misfit calculation
        indicesForPunishment = trueFracMDD == 0
        if len(omitValueIndices) != 0:
            TrueFracFi[omitValueIndices] = 0
            trueFracMDD[omitValueIndices] = 0
        
        #Calculate L1 loss
        misfit = torch.absolute(TrueFracFi-trueFracMDD)


        # Assign penalty for each step if the model runs out of gas before the experiment did

        # Add a misfit penalty of 1 for each heating step that ran out of gas early
        misfit[indicesForPunishment] = 1
        
        if torch.round(Fi_MDD[-1],decimals=2) != 1:
            return 10**3
        
        #     if torch.round(Fi_MDD[-2],decimals=2) >= 1:
        #          return 10**10
        # Return the sum of the residuals

        return torch.sum(misfit).item()