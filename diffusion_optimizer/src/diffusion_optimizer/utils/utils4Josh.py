import math
import numpy as np
import pandas as pd
import torch
import random
import math as math


#The inputs are...
# X: This should be a vector of the input coefficients like we discused. For one domain, 
# there are three, for two domains there are 4, for 3 domains, there are 6, for 4 domains, 8, etc.

# data: This should be two columns, where data[0,:] is a column of times in hours (typically between 0.25 hours and 6 hours), and data[1:,0] 
# is a list of temperatures in C (usualy between ~50 and 1200).

def forwardModelX(X,data,lookup_table):
   

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    # Infer the number of domains from input
    if len(X) <= 3:
        ndom = 1
    else:
        ndom = (len(X))//2

    # Make a subset of X, removing the Ea so that we have an even number of elements
    temp = X[1:]


    TC = data[0]
    thr = data[1]
    if thr[0] > 15:
        thr = thr/60


    # reshape data to improve calculation speeds
    lnD0aa = torch.tile(temp[0:ndom],(len(thr)+2,1)) # Do this for LnD0aa
    fracstemp = temp[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)
    fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=-1),(len(thr)+2,1)) # Add the last frac as 1-sum(other fracs)
    Ea = torch.tile(X[0],(len(thr)+2,ndom)) # Do for Ea

    # unit conversions
    tsec = thr*3600 #Convert from hours to seconds (the units we use in diffusion calcs)
    

     
    # Put time and cumulative time in the correct shape
    tsec = torch.tile(torch.reshape(tsec,(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of getting tsec into a numdom x numstep matrix for multiplication
    cumtsec = torch.tile(torch.reshape(torch.cumsum(tsec[:,1],dim=0),(-1,1)),(1,Ea.shape[1])) #Same as above, but for cumtsec        

    # Convert TC to TK and put in correct shape for quick computation                                                 
    TK = torch.tile(torch.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1])) #This is a complicated-looking way of turning TC from a 1-d array to a 2d array and making two column copies of it

    # Calculate D/a^2 for each domain
    Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))
    

    # Pre-allocate fraction and Dtaa
    f = torch.zeros(Daa.shape)
    Dtaa = torch.zeros(Daa.shape)
    DtaaForSum = torch.zeros(Daa.shape)
    
    
    # Calculate Dtaa in incremental (not cumulative) form including the added heating steps
    DtaaForSum[0,:] = Daa[0,:]*tsec[0,:]
    DtaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])

    # Make the correction for P_D vs D_only
    for i in range(len(DtaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
        if DtaaForSum[0,i] <= 1.347419e-17:
            DtaaForSum[0,i] *= 0
        elif DtaaForSum[0,i] >= 4.698221e-06:
            pass
        else:
            DtaaForSum[0,i] *= lookup_table(DtaaForSum[0,i])

    # Calculate Dtaa in cumulative form.
    Dtaa = torch.cumsum(DtaaForSum, axis = 0)

    
    # Calculate f at each step
    Bt = Dtaa*torch.pi**2
    f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
    f[Bt>0.0091] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[Bt>0.0091])-(3/(torch.pi**2))* \
            ((torch.pi**2)*Dtaa[Bt>0.0091])
    f[Bt >1.8] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[Bt > 1.8])

    
    # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
    f_MDD = f*fracs

    # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
    # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
    sumf_MDD = torch.sum(f_MDD,axis=1)

    # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
    # Return that sumf_MDD == 0
    if (torch.round(sumf_MDD[2],decimals=6) == 1):
        return (TC[2:], torch.zeros(len(sumf_MDD)-2),torch.zeros(len(sumf_MDD)-2),0)
        

    # This is basically checking if the hypothetical experiment actually released all the gas. If it didn't, then we'll lose 
    # The ability to tell after we perform the next step, so I think this set of experiments is dangerous and unnecessary to explore.
    # As a result, I'll just return None if you happen to hit this combination of parameters.
    not_released = torch.tensor(0)
    if torch.round(torch.sum(f[-1,:]),decimals=3) != ndom: # if the gas released at the end of the experiment isn't 100% for each domain...
        return None # This will probably 

    # Remove the two steps we added, recalculate the total sum, and renormalize.
    newf = torch.zeros(sumf_MDD.shape)
    newf[0] = sumf_MDD[0]
    newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]
    newf = newf[2:]
    normalization_factor = torch.max(torch.cumsum(newf,0))
    diffFi= newf/normalization_factor 


    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released

    # Calculate duration for each individual step removing the added steps
    diffti = cumtsec[1:,1]-cumtsec[0:-1,1]
    diffti = torch.concat((torch.unsqueeze(cumtsec[0,0],dim=-1),diffti),dim=-1)
    diffti = diffti[2:]
    
   

    return (diffti) ##This returns the fraction released at each heating step, which is what we ultimately fit to