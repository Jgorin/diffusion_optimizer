import math
import numpy as np
import pandas as pd
import torch


def D0calc_MonteCarloErrors(expdata):
    # Function for calculating D0 and D/a^2 from experimental data. Input should be a
    # Pandas DataFrame with columns "TC", "thr",
    # M, and, and delM, which correspond to heating temperature (deg C), 
    # heating step duration (time in hours),
    # M (measured concentration in cps, atoms, or moles), delM (same units)
    
    # Calculate diffusivities from the previous experiment
    TC = expdata.loc[:,"TC"].array
    thr = expdata.loc[:,"thr"].array
    M = expdata.loc[:,"M"].array
    delM = expdata.loc[:,"delM"].array

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
    use_b = (Fi > 0.1) & (Fi<= 0.85)
    use_c = (Fi > 0.85) & (Fi<= 1.0)

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
        delMCFi[i] = (np.amax(MCFi[i,:],0) - np.amin(MCFi[i,:],0))/2
        MCFimean[i] = np.mean(MCFi[i,:],0)
    
    #Initialize vectors
    MCDR2_a = np.zeros([nstep,n_sim])
    MCDR2_b = np.zeros([nstep,n_sim])
    MCDR2_c = np.zeros([nstep,n_sim])
    MCdiffFi = np.zeros([nstep,n_sim])



    for m in range(1,nstep): #For step of each experiment...
        for n in range(n_sim):
            MCdiffFi[m,n] = MCFi[m,n] - MCFi[m-1,n] #calculate the fraction released at each step
    for n in range(n_sim): #For each first step of an experiment, insert 0 for previous amount released
        MCDR2_a[0,n] = ((MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]))
    for m in range(1,nstep): #Calculate fechtig and kalbitzer equations for each fraction
        for n in range(n_sim):
            MCDR2_a[m,n] = ( (MCFi[m,n])**2 - (MCFi[m-1,n])**2 )*math.pi/(36*(diffti[m-1]));
            MCDR2_b[m,n] = (1/((math.pi**2)*diffti[m-1]))*( -(math.pi*math.pi/3)* MCdiffFi[m,n] \
                            - (2*math.pi)*( np.sqrt(1-(math.pi/3)*MCFi[m,n]) \
                            -np.sqrt(1 - (math.pi/3)*MCFi[m-1,n]) ))
            MCDR2_c[m,n] = (1/(math.pi*math.pi*diffti[m-1]))*(np.log((1-MCFi[m-1,n])/(1-MCFi[m,n])));

    use_a_MC = (MCFi<= 0.1) & (MCFi> 0.00000001)
    use_b_MC = (MCFi > 0.1) & (MCFi<= 0.85)
    use_c_MC = (MCFi > 0.85) & (MCFi<= 1.0) 


    MCDR2 = use_a_MC*MCDR2_a + np.nan_to_num(use_b_MC*MCDR2_b) + use_c_MC*MCDR2_c

    MCDR2_uncert = np.zeros([nstep,1])
    for i in range(nstep):
        MCDR2_uncert[i,0] = np.std(MCDR2[i,:]) 


    return pd.DataFrame({"Tplot": Tplot,"Fi": MCFimean.ravel(),"Fi uncertainty": \
                            delMCFi.ravel(), "Daa": DR2,"Daa uncertainty": MCDR2_uncert.ravel(), \
                            "ln(D/a^2)": np.log(DR2),"ln(D/a^2)-del": np.log(DR2-MCDR2_uncert.ravel()), \
                            "ln(D/a^2)+del": np.log(DR2+MCDR2_uncert.ravel()) })

def forwardModelKinetics(kinetics,expData): 
    # kinetics: (Ea, lnd0aa_x, fracs_x). To make this compatible with other functions, if there are x fracs, input x-1 fractions, and the code will determine the
    # final fraction.

    R = 0.008314 #gas constant
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    # Parameters that need to be read in (These I'll likely read from a file eventually
    # But we'll read the data from above just to test the function for now...)
    if len(kinetics) <= 3:
        ndom = 1
    else:
        ndom = (len(kinetics))//2

    # Make a subset of X, removing the Ea so that we have an even number of elements
    temp = kinetics[1:]

    if type(expData) is tuple:
        
        TC = expData[0]
        thr = expData[1]
        lnDaa = expData[2]
        Fi = expData[3]

  
    else:
      
        TC = expData.np_TC
        thr = expData.np_thr/60
        lnDaa = expData.np_lnDaa
        Fi = expData.np_Fi_exp


    # Grab the parameters from the input
    lnD0aa = torch.tile(temp[0:ndom],(len(thr)+2,1)) #lnD0aa = np.tile(lnD0aa,(len(thr),1))
    fracstemp = temp[ndom:]
    fracs = torch.tile(torch.concat((fracstemp,1-torch.sum(fracstemp,axis=0,keepdim=True)),axis=-1),(len(thr)+2,1))
    Ea = torch.tile(kinetics[0],(len(thr)+2,ndom)) # This is an Ea for each domain



    # Temporary!!! 
    # Manually add in steps from the irridiation and from the lab storage

    seconds_since_irrad = torch.tensor(1391*24*60*60)
    irrad_duration_sec = torch.tensor(5*3600) # in seconds
    irrad_T = torch.tensor(40) # in C
    storage_T = torch.tensor(21.11111) # in C
    time_add = torch.tensor([irrad_duration_sec,seconds_since_irrad])
    temp_add = torch.tensor([irrad_T, storage_T])
    tsec = torch.cat([time_add,thr*3600])
    TC = torch.cat([temp_add,TC])

     
    # Put variables in correct units
    tsec = torch.tile(torch.reshape(tsec,(-1,1)),(1,Ea.shape[1])) #This is a complicated way of getting tsec into a numdom x numstep matrix for multiplication
    cumtsec = torch.tile(torch.reshape(torch.cumsum(tsec[:,1],dim=0),(-1,1)),(1,Ea.shape[1])) #Same as above, but for cumtsec                                                         
    TK = torch.tile(torch.reshape((TC + 273.15),(-1,1)),(1,Ea.shape[1])) #This is a complicated way of turning TC from a 1-d array to a 2d array and making two column copies of it


    Daa = torch.exp(lnD0aa)*torch.exp(-Ea/(R*TK))
    
    # Pre-allocate fraction and Dtaa
    f = torch.zeros(Daa.shape)
    Dtaa = torch.zeros(Daa.shape)
    DaaForSum = torch.zeros(Daa.shape)
    
    
    #Calculate D
    DaaForSum[0,:] = Daa[0,:]*tsec[0,:]
    DaaForSum[1:,:] = Daa[1:,:]*(cumtsec[1:,:]-cumtsec[0:-1,:])
   

    lookup_table = pd.read_parquet("DiffusionLookup.parquet")


    # It might be smart for us to make this stage optional at some point
    # because not every mineral is actually diffusive enough that room temperature will do anything
    for i in range(len(DaaForSum[0,:])): #This is a really short loop... range of i is # domains. Maybe we could vectorize to improve performance?
        min_index = torch.argmin(torch.abs(torch.tensor(lookup_table["Model_Dtaa"])-DaaForSum[0,i]))
        DaaForSum[0,i] = DaaForSum[0,i]*lookup_table["Ratio"][min_index.item()]


    Dtaa = torch.cumsum(DaaForSum, axis = 0)

    


    f = (6/(math.pi**(3/2)))*torch.sqrt((math.pi**2)*Dtaa)
    f[f>=0.1] = (6/(torch.pi**(3/2)))*torch.sqrt((torch.pi**2)*Dtaa[f>=0.1])-(3/(torch.pi**2))* \
            ((torch.pi**2)*Dtaa[f>=0.1])
    f[f>=0.9] = 1 - (6/(torch.pi**2))*torch.exp(-(torch.pi**2)*Dtaa[f>=0.9])

    


    # # If any member of f is preceeded by a value greater than it, set equal to 1 (we've reached total gas released)
    # f[1:,:][f[1:,:]<f[0:-1,:]] = 1
    # f[1:,:][f[0:-1,:]==1] = 1

     # f for one to end where f from 0 to end-1 ==1, set to 1


    for i in range(len(f)-1): #rows
        for j in range(len(f[0])): #cols
            if f[i,j] > f[i+1,j]:
                f[i+1,j] = f[i,j]
    
    # loop backwards through the columns and find the point at which we've converged to 1 and
    # set all values preceeding it to 1 as well
    exit_flag = 0
    for i in range(len(f[0])): #columns
        exit_flag = 0
        j = len(f)-1
        while exit_flag == 0: #rows
            if (f[j,i] > f[j-2,i]) and (f[j,i] == f[j-1,i]):
                f[j:,i] = 1
                exit_flag = 1
            j -= 1
            if j == -1:
                exit_flag = 1
    for i in range(len(f[0])):
        if f[0,i] < 0:
            f[0:,:] = 1

    
    # Multiply each gas realease by the percent gas located in each domain
    
    f_MDD = f*fracs
    # Now I need to renormalize by calculating first the individual steps and then the new sum

    # Calculate the total gas released at each step from each gas fraction
    sumf_MDD = torch.sum(f_MDD,axis=1)
    
    TrueFracMDD = (sumf_MDD[1:]-sumf_MDD[0:-1])
    TrueFracMDD = torch.concat((torch.unsqueeze(sumf_MDD[0],dim=-1),TrueFracMDD),dim=-1)
    TrueFracMDD = TrueFracMDD[2:]


    #Calculate the apparent Daa from the MDD using equations of Fechtig and Kalbitzer
    Daa_MDD_a = torch.zeros(TrueFracMDD.shape)
    Daa_MDD_b = torch.zeros(TrueFracMDD.shape)
    Daa_MDD_c = torch.zeros(TrueFracMDD.shape)

    # use equations 5a through c from Fechtig and Kalbitzer for spherical geometry
    # Fechtig and Kalbitzer Equation 5a, for cumulative gas fractions up to 10%
    # special case when i = 1; need to insert 0 for previous amount released

    #torch.concat((torch.unsqueeze(Fi_exp[0],dim=-1),TrueFracFi),dim=-1)
    diffti = cumtsec[1:,1]-cumtsec[0:-1,1]
    diffti = torch.concat((torch.unsqueeze(cumtsec[0,0],dim=-1),diffti),dim=-1)
    diffti = diffti[2:]
    diffFi = TrueFracMDD
    sumf_MDD = torch.cumsum(diffFi,axis=0)



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


    return (TC[2:],lnDaa,sumf_MDD,lnDaa_MDD)
