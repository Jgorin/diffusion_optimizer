from diffusion_optimizer.neighborhood.objective import Objective
from diffusion_optimizer.utils.utils import forwardModelKinetics
import torch
import math as math 

class DiffusionObjective(Objective):
    
    # override evaluate function
    def __call__(self, X): #__call__ #evaluate
        data = self.dataset
        #omitValueIndices = self.omitValueIndices
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
        #total_moles = X[0]
        #X = X[1:]
        

        # Unpack the parameters and spit out a high misfit value if constraints are violated
        if len(X) <= 3:
            ndom = 1
        else:
            ndom = (len(X))//2

        # Grab the other parameters from the input
        
        temp = X[1:]
        lnD0aa = temp[0:ndom]
        fracstemp = temp[ndom:]

                
        #This punishes the model for proposing input fractions that don't add up to 1.0
        # if torch.sum(fracstemp,axis=0,keepdim = True)>0.999:
        #     return torch.sum(fracstemp,axis=0,keepdim = True).item()*100

        # Now that we know the fracs add up to one, calculate the fraction for the last domain.
        # This is determined by the other fractions.
        sumTemp = (1-torch.sum(fracstemp,axis=0,keepdim = True))
        fracs = torch.concat((fracstemp,sumTemp),dim=-1)

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

            return 10**17
        
        #exp_moles = torch.tensor(data.M)
        #total_moles = torch.sum(exp_moles)


        # Calculate L1 loss
        # Production notes... 
        # 1. We should also try the L2 loss
        # 2. We should also consider trying percent loss, since our values span several orders of magnitude. Could maybe even try log fits..
        #misfit = torch.absolute(TrueFracFi-trueFracMDD) #TrueFracFi is real
        
        
        #moles_MDD = trueFracMDD * total_moles

        #misfit = torch.abs(exp_moles-moles_MDD)
       
        #misfit = torch.absolute(TrueFracFi-trueFracMDD) #TrueFracFi is real

        exp_moles = torch.tensor(data.M)
        MDD_moles = 4.000001338E+09*trueFracMDD
        misfit = (exp_moles-MDD_moles)**2/(torch.tensor(data.delM))
        #misfit = torch.abs(TrueFracFi-trueFracMDD)
        #misfit = ((TrueFracFi-trueFracMDD)**2)/(1/data.uncert)

        # Add a misfit penalty of 1 for each heating step that ran out of gas early.
        # THOUGHT: I'M CURENTLY OMITTING THE LAST TWO INDICES INSTEAD OF JUST THE LAST 1.
        # I NOTICED THAT THIS PROVIDED SOMEWHAT BETTER MODEL BEHAVIOR, THOUGH IT WOULD BE MORE SCIENTIFICALLY SOUND TO 
        # ONLY ASSERT THAT THE LAST STEP WAS ==1.
        #misfit[0:-2][indicesForPunishment] += 1

        misfit[0:-2][indicesForPunishment] += 10**17

        # Return the sum of the residuals
        #misfit = torch.sum(misfit)+not_released_flag
        if math.isnan((torch.sum(misfit)+not_released_flag).item()+ran_out_too_early.item()):
            breakpoint()
        #return (((torch.sum(misfit)+not_released_flag).item()+ran_out_too_early.item())/len(data.M))/(10**10)

        output = ((torch.sum(misfit)+not_released_flag)+ran_out_too_early).item()
        # output = torch.tensor(output,requires_grad=True)


        return output