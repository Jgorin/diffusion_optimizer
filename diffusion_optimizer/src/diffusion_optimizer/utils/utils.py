import math
import numpy as np
import pandas as pd
import torch
import random
import math as math


def D0calc_MonteCarloErrors(data):
    # Function for calculating D0 and D/a^2 from experimental data. Input should be a
    # Pandas DataFrame with columns "TC", "thr",
    # M, and, and delM, which correspond to heating temperature (deg C), 
    # heating step duration (time in hours),
    # M (measured concentration in cps, atoms, or moles), delM (same units)
    
    # Calculate diffusivities from the previous experiment
    TC = data.loc[:,"TC"].array
    thr = data.loc[:,"thr"].array
    M = data.loc[:,"M"].array
    delM = data.loc[:,"delM"].array

    #Check if units are in minutes and convert from hours to minutes if necesssary
    if thr[1]>4:
        thr = thr/60

    #Convert units
    TK = 273.15+TC
    tsec = thr*60*60
    Tplot = 1*10**4/TK
    nstep = len(M)
    cumtsec = np.cumsum(tsec)
    Si = np.cumsum(M)
    S = np.amax(Si)
    Fi = Si/S


    # initialize diffusivity vectors fore each Fechtig and Kalbitzer equation
    DR2_a = np.zeros([nstep])
    DR2_b = np.zeros([nstep])
    DR2_c = np.zeros([nstep])

    # Create the a list of times for each heating step
    diffti = cumtsec[1:]-cumtsec[0:-1]

    # Create a list of the gas fraction differences between steps
    diffFi = Fi[1:]-Fi[0:-1]


    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released

    DR2_a[0] = ( (Fi[0]**2 - 0.**2 )*math.pi/(36*(cumtsec[0])))


    # Equation 5a for all other steps

    DR2_a[1:] = ((Fi[1:])**2 - (Fi[0:-1])**2 )*math.pi/(36*(diffti))

    # Fechtig and Kalbitzer Equation 5b, for cumulative gas fractions between 10 and 90%

    DR2_b[0] = (1/((math.pi**2)*tsec[0]))*((2*math.pi)-((math.pi*math.pi/3)*Fi[0])\
                                        - (2*math.pi)*(np.sqrt(1-(math.pi/3)*Fi[0])))
    DR2_b[1:] = (1/((math.pi**2)*diffti))*(-(math.pi*math.pi/3)*diffFi \
                                        - (2*math.pi)*( np.sqrt(1-(math.pi/3)*Fi[1:]) \
                                            - np.sqrt(1 - (math.pi/3)*Fi[0:-1]) ))

    # Fechtig and Kalbitzer Equation 5c, for cumulative gas fractions greater than 90%
    DR2_c[1:] = (1/(math.pi*math.pi*diffti))*(np.log((1-Fi[0:-1])/(1-Fi[1:])))

    # Decide which equation to use based on the cumulative gas fractions from each step
    use_a = (Fi<= 0.1) & (Fi> 0.00000001)
    use_b = (Fi > 0.1) & (Fi<= 0.9)
    use_c = (Fi > 0.9) & (Fi<= 1.0)

    # Compute the final values
    DR2 = use_a*DR2_a + np.nan_to_num(use_b*DR2_b) + use_c*DR2_c

    # Compute uncertainties in diffusivity using a Monte Carlo simulation
    # Generates simulated step degassing datasets, such that each step of the 
    # experiment has a Gaussian distribution centered at M and with 1s.d. of
    # delM across the simulated datasets.Then recomputes diffusivities for each 
    # simulated dataset and uses the range of diffusivities for each step across
    # all simulated datasets to estimate uncertainty. 
    # make vector with correct diffusivites for each step

    n_sim = 30000 #number of simulations in the monte carlo
    MCsim = np.zeros([nstep,n_sim])#initialize matrix for simulated measurements


    
    
    for i in range(nstep):
        #Generate the simulated measurements
        MCsim[i,:] = np.random.randn(1,n_sim)*delM[i] + M[i]

    #compute cumulative gas release fraction for each simulation
    MCSi = np.cumsum(MCsim,0)
    MCS = np.amax(MCSi,0)
    MCFi = np.zeros([nstep,n_sim])
    delMCFi = np.zeros([nstep,1])
    MCFimean = np.zeros([nstep,1])



    for i in range(n_sim):
        MCFi[:,i] = MCSi[:,i]/np.amax(MCSi[:,i])
    for i in range(nstep):
        #delMCFi[i] = (np.amax(MCFi[i,:],0) - np.amin(MCFi[i,:],0))/2
        MCFimean[i] = np.mean(MCFi[i,:],0)
    
    #Initialize vectors
    MCDR2_a = np.zeros([nstep,n_sim])
    MCDR2_b = np.zeros([nstep,n_sim])
    MCDR2_c = np.zeros([nstep,n_sim])
    MCdiffFi = np.zeros([nstep,n_sim])



    for m in range(1,nstep): #For step of each experiment...
        for n in range(n_sim):
            MCdiffFi[m,n] = MCFi[m,n] - MCFi[m-1,n] #calculate the fraction released at each step
            MCdiffFi[0,n] = MCFi[0,n]
    for m in range(0,nstep):
        delMCFi[m] = np.std(MCdiffFi[m,:])
    for n in range(n_sim): #For each first step of an experiment, insert 0 for previous amount released
        MCDR2_a[0,n] = ((MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]))
        
    for m in range(1,nstep): #Calculate fechtig and kalbitzer equations for each fraction
        for n in range(n_sim):
            MCDR2_a[m,n] = ( (MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]));
            MCDR2_b[m,n] = (1/((math.pi**2)*diffti[m-1]))*( -(math.pi*math.pi/3)* MCdiffFi[m,n] \
                            - (2*math.pi)*( np.sqrt(1-(math.pi/3)*MCFi[m,n]) \
                            -np.sqrt(1 - (math.pi/3)*MCFi[m-1,n]) ))
            MCDR2_c[m,n] = (1/(math.pi*math.pi*diffti[m-1]))*(np.log((1-MCFi[m-1,n])/(1-MCFi[m,n])));
    MCdiffFiFinal = np.zeros([nstep])
    for m in range(0,nstep):
        MCdiffFiFinal[m] = np.mean(MCdiffFi[m,:])

    use_a_MC = (MCFi<= 0.1) & (MCFi> 0.00000001)
    use_b_MC = (MCFi > 0.1) & (MCFi<= 0.9)
    use_c_MC = (MCFi > 0.9) & (MCFi<= 1.0) 


    MCDR2 = use_a_MC*MCDR2_a + np.nan_to_num(use_b_MC*MCDR2_b) + use_c_MC*MCDR2_c

    MCDR2_uncert = np.zeros([nstep,1])
    for i in range(nstep):
        MCDR2_uncert[i,0] = np.std(MCDR2[i,:]) 


    return pd.DataFrame({"Tplot": Tplot,"Fi": MCFimean.ravel(),"Fi uncertainty": \
                            delMCFi.ravel(), "Daa": DR2,"Daa uncertainty": MCDR2_uncert.ravel(), \
                            "ln(D/a^2)": np.log(DR2),"ln(D/a^2)-del": np.log(DR2-MCDR2_uncert.ravel()), \
                            "ln(D/a^2)+del": np.log(DR2+MCDR2_uncert.ravel()) })

def forwardModelKinetics(X,data,lookup_table): # (X,data,lookup_table)
    # X: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2

    # Infer the number of domains from input
    if len(X) <= 3:
        ndom = 1
    else:
        ndom = (len(X))//2

    # Make a subset of X, removing the Ea so that we have an even number of elements
    temp = X[1:]

    if type(data) is tuple: # Some functions use fwdModelX by inserting data as  tuple, while others input as a dataset
        TC = data[0]
        thr = data[1]
        if thr[0] >10: # If values are > 10, then units are likely in minutes, not hours.
            thr = thr/60
        lnDaa = data[2]
        Fi = data[3]

  
    else: # data is type Dataset.
        TC = data.np_TC
        thr = data.np_thr
        if thr[0] >10: # If values are > 10, then units are likely in minutes, not hours.
            thr = thr/60
        lnDaa = data.np_lnDaa
        Fi = data.np_Fi_exp


    # Copy the parameters into dimensions that mirror those of the experiment schedule to increase calculation speed.
    lnD0aa = torch.tile(temp[0:ndom],(len(thr)+2,1)) # Do this for LnD0aa
    fracstemp = temp[ndom:] # Grab fracs that were input (one will be missing because it is pre-determined by the others)
    fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=-1),(len(thr)+2,1)) # Add the last frac as 1-sum(other fracs)
    Ea = torch.tile(X[0],(len(thr)+2,ndom)) # Do for Ea


    # THIS IS TEMPORARY-- WE NEED TO ADD THIS AS AN INPUT.. THE INPUTS WILL NEED TO BE
    # 1. Duration of irradiation
    # 2. Temperature during irradiation
    # 3. Duration of lab storage
    # 4. Temperature during lab storage

    # We might also want to make this all optional at some point, since some minerals are so retentive 
    # that they wont lease any helium during irradiation and storage.
    
    # Currently, I'm Manually adding in steps from the irridiation and from the lab storage


    seconds_since_irrad = torch.tensor(110937600)  # seconds
    irrad_duration_sec = torch.tensor(5*3600) # in seconds
    irrad_T = torch.tensor(40) # in C
    storage_T = torch.tensor(21.1111111) # in C
    
    # Make a tensor with these two extra heating steps in order.
    time_add = torch.tensor([irrad_duration_sec,seconds_since_irrad])
    temp_add = torch.tensor([irrad_T, storage_T])
    
    # Add the two new steps to the schedule of the actual experiment
    tsec = torch.cat([time_add,thr*3600])
    TC = torch.cat([temp_add,TC])

     
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

    for i in range(1,len(f[1:,:])):
        temp =  f[i,:] - f[i-1,:]
        temp = torch.ceil(temp) 
        f[i,:] = f[i-1,:]*(1-temp) + f[i,:]*temp
        



  


             #if any negative values when you calculate the frac difference in a single domain



    # Multiply each gas realease by the percent gas located in each domain (prescribed by input)
    f_MDD = f*fracs


    # Renormalize everything by first calculating the fractional releases at each step, summing back up, 
    # and then dividing by the max released in each fraction. This simulates how we would have measured and calculated this in the lab.
    sumf_MDD = torch.sum(f_MDD,axis=1)

    # If the second heating step gets gas release all the way to 100%, then the rest of the calculation is not necessary. 
    # Return that sumf_MDD == 0
    if (torch.round(sumf_MDD[2],decimals=6) == 1):
        return (TC[2:], torch.zeros(len(sumf_MDD)-2),torch.zeros(len(sumf_MDD)-2),0)
        

    # Set a flag if the final value for sumf_MDD isn't really close to 1. If it isn't, then we'll lose our ability to tell
    # As soon as we renormalize. We'll pass this value to the misfit, and make it so that these models are not favorable.
    not_released = torch.tensor(0)
    if torch.round(torch.sum(f[-1,:]),decimals=3) != ndom: # if the gas released at the end of the experiment isn't 100% for each domain...

        not_released = (1-sumf_MDD[-1])*10**17 # Return a large misfit that's a function of how far off we were.

    # Remove the two steps we added, recalculate the total sum, and renormalize.
    newf = torch.zeros(sumf_MDD.shape)
    newf[0] = sumf_MDD[0]
    newf[1:] = sumf_MDD[1:]-sumf_MDD[0:-1]
    newf = newf[2:]
    normalization_factor = torch.max(torch.cumsum(newf,0))
    diffFi= newf/normalization_factor 

    # I THINK WE CAN ACTUALLY DITCH THIS CALCULATION FROM HERE DOWN TO INCREASE PERFORMANCE! LET'S DO LATER, THOUGH).
    # Calculate the apparent Daa from the MDD using equations of Fechtig and Kalbitzer 
    Daa_MDD_a = torch.zeros(diffFi.shape)
    Daa_MDD_b = torch.zeros(diffFi.shape)
    Daa_MDD_c = torch.zeros(diffFi.shape)

    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released

    # Calculate duration for each individual step removing the added steps
    diffti = cumtsec[1:,1]-cumtsec[0:-1,1]
    diffti = torch.concat((torch.unsqueeze(cumtsec[0,0],dim=-1),diffti),dim=-1)
    diffti = diffti[2:]
    
    # Resum the gas fractions into cumulative space that doesn't include the two added steps
    sumf_MDD = torch.cumsum(diffFi,axis=0)


    # Calculate Daa from the MDD model using fechtig and kalbitzer

    Daa_MDD_a[0] = ( (sumf_MDD[0]**2 - 0.**2 )*torch.pi/(36*(diffti[0])))


    # Equation 5a for all other steps

    Daa_MDD_a[1:] = ((sumf_MDD[1:])**2 - (sumf_MDD[0:-1])**2 )*math.pi/(36*(diffti[1:]))

    # Fechtig and Kalbitzer Equation 5b, for cumulative gas fractions between 10 and 90%
    Daa_MDD_b[0] = (1/((torch.pi**2)*tsec[0,0]))*((2*torch.pi)-((math.pi*math.pi/3)*sumf_MDD[0])\
                                        - (2*math.pi)*(torch.sqrt(1-(math.pi/3)*sumf_MDD[0])))
    Daa_MDD_b[1:] = (1/((math.pi**2)*diffti[1:]))*(-(math.pi*math.pi/3)*diffFi[1:] \
                                        - (2*math.pi)*( torch.sqrt(1-(math.pi/3)*sumf_MDD[1:]) \
                                            - torch.sqrt(1 - (math.pi/3)*sumf_MDD[0:-1]) ))

    # Fechtig and Kalbitzer Equation 5c, for cumulative gas fractions greater than 90%
    Daa_MDD_c[1:] = (1/(math.pi*math.pi*diffti[1:]))*(torch.log((1-sumf_MDD[0:-1])/(1-sumf_MDD[1:])))

    # Decide which equation to use based on the cumulative gas fractions from each step
    use_a = (sumf_MDD<= 0.1) & (sumf_MDD> 0.00000001)
    use_b = (sumf_MDD > 0.1) & (sumf_MDD<= 0.9)
    use_c = (sumf_MDD > 0.9) & (sumf_MDD<= 1.0)
    Daa_MDD = use_a*Daa_MDD_a + torch.nan_to_num(use_b*Daa_MDD_b) + use_c*Daa_MDD_c
    

    lnDaa_MDD = torch.log(Daa_MDD)


    # for i in range(len(sumf_MDD)):
    #     if sumf_MDD[i] <0:
    #         breakpoint()


    return (TC[2:],lnDaa,sumf_MDD,lnDaa_MDD,not_released) #Temperatures, 
