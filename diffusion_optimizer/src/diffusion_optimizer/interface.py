import argparse
# from diffusion_optimizer.parameter_tuning import tune_parameters
from diffusion_optimizer.generate_inputs import generate_inputs
from diffusion_optimizer.parameter_tuning.parameter_tuning_v2 import tune_parameters
import yaml
import numpy as np
import os 
import uuid
import sys
import pandas as pd
import matplotlib.pyplot as plt
from diffusion_optimizer.diffusion_objective import DiffusionObjective
from diffusion_optimizer.neighborhood.objective import Objective
import torch
from diffusion_optimizer.neighborhood.optimizer import Optimizer, OptimizerOptions
from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.utils.utils import forwardModelKinetics
import seaborn as sns

file_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    root_output_dir = f"{file_path}/../../../main/output"
    
    # generate output folder
    run_id = uuid.uuid1()
    full_output_dir = f"{root_output_dir}/{run_id}"
    config = yaml.safe_load(open(sys.argv[sys.argv.index("--config_path") + 1]))
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", required=True)    
    argparser.add_argument("--input_csv_path", default=f"{file_path}/../../../main/input/{config['nameOfInputCSVFile']}")
    argparser.add_argument("--output_path", default=full_output_dir)
    argparser.add_argument("-generate_inputs", action='store_true')
    argparser.add_argument("-parameter_tuning", action='store_true')
    return (argparser.parse_args(), config)

def list2range(list):
    return range(list[0], list[1])

def list2tuple(list):
    return (list[0], list[1])

def plot_results(optimizer,dataset):
    
    params = torch.tensor(optimizer.sample_manager._elites[0]._param)
    data = forwardModelKinetics(params,(torch.tensor(dataset.TC), torch.tensor(dataset.thr/60),dataset.np_lnDaa,torch.tensor(dataset.Fi)))
    T_plot = 10000/(dataset["TC"]+273.15)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(T_plot,dataset["ln(D/a^2)"],'bo',markersize=10)
    plt.plot(T_plot,data[-1],'ko',markersize=7)
    plt.ylabel("ln(D/a^2)")
    plt.xlabel("10000/T (K)")
    plt.subplot(2,1,2)

    Fi_MDD = np.array(data[-2])
    temp = Fi_MDD[1:]-Fi_MDD[0:-1]
    Fi_MDD =np.insert(temp,0,Fi_MDD[0])


    Fi = np.array(dataset.Fi) 
    temp = Fi[1:]-Fi[0:-1]
    Fi = np.insert(temp,0,Fi[0])
    plt.plot(range(0,len(T_plot)),Fi,'b-o',markersize=5)
    plt.plot(range(0,len(T_plot)),Fi_MDD,'k-o',markersize=3)
   
    plt.xlabel("step number")
    plt.ylabel("Fractional Release (%)")
    plt.show()
    
def main():
    (args, config) = parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    
    # convert arrays to tuples and limits in config
    limitDict = config["limits"]
    names = list(limitDict.keys())
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # generate inputs
    if args.generate_inputs:
        generate_inputs(args.input_csv_path, f"{args.output_path}/{config['nameOfExperimentalResultsFile']}")
    # run parameter tuning
    if args.parameter_tuning:
        tune_parameters(args.config_path, f"{args.output_path}/{config['nameOfExperimentalResultsFile']}", args.output_path)
    
    else:
        dataset = Dataset(pd.read_csv(f"{args.output_path}/{config['nameOfExperimentalResultsFile']}"))
        #Drew added this to pass in the omitted values in misfit calc
        objective = DiffusionObjective(dataset,config["omitValueIndices"])
        finalList = []
        
        #for i in range(0,30):
         #   print(i)
        options = OptimizerOptions(
            #num_samp_range=[8,800],
            num_samp_decay=0.540790596825097,
            resamp_to_samp_ratio = 0.548907670029791,
            #epsilon_range=[0.000001,0.00001],
            epsilon_growth=8.910183135980043*1**(-9),
            num_samp_max= 1089, #873,
            num_samp_ratio = 0.04926705645229101 ,
            epsilon_max=5.824566276586376*1**(-8),
            epsilon_range_ratio= 0.35374549914032105,
            
                
            )
            
        optimizer = Optimizer(objective, limits, names, options,maximize=False)
        optimizer.update(500)
        plot_results(optimizer,dataset)
   
        #     finalList.append(optimizer.param_list)

        
        
        # for i in range(len(finalList)):
        #     Ea = []
        #     lnD01 = []
        #     lnd02 = []
        #     lnd03 = []
        #     Frac1 = []
        #     Frac2 = []
        #     for j in range(len(finalList[i])):
        #         Ea.append(finalList[i][j][0][0])
        #         lnD01.append(finalList[i][j][0][1])
        #         lnd02.append(finalList[i][j][0][2])
        #         lnd03.append(finalList[i][j][0][3])
        #         Frac1.append(finalList[i][j][0][4])
        #         Frac2.append(finalList[i][j][0][5])
        #     if i==0:
        #         EaData = pd.DataFrame(Ea)
        #         lnD01Data = pd.DataFrame(lnD01)
        #         lnD02Data = pd.DataFrame(lnd02)
        #         lnD03Data = pd.DataFrame(lnd03)
        #         Frac1Data = pd.DataFrame(Frac1)
        #         Frac2Data = pd.DataFrame(Frac2)
                

        #     else:
        #         EaData[i] = Ea
        #         lnD01Data[i] = lnD01
        #         lnD02Data[i] = lnd02
        #         lnD03Data[i] = lnd03
        #         Frac1Data[i] = Frac1
        #         Frac2Data[i] = Frac2 
        # #data = pd.DataFrame({"Ea": Ea,"LnD0aa1":lnD01, "LnD0aa2": lnd02, "Lnd0aa3": lnd03, "Frac1": Frac1, "Frac2":Frac2})
        
        # EaData = EaData.median(axis=1)
        # lnD01Data = lnD01Data.median(axis=1)
        # lnD02Data = lnD02Data.median(axis=1)
        # lnD03Data = lnD03Data.median(axis=1)
        # Frac1Data = Frac1Data.median(axis=1)
        # Frac2Data = Frac2Data.median(axis=1)

        # plt.figure()
        # plt.subplot(7,1,1)
        # plt.plot(range(len(EaData)),EaData,'r')
        # plt.ylabel("Ea")
        # plt.subplot(7,1,2)
        # plt.plot(range(len(lnD01Data)),lnD01Data,'g')
        # plt.ylabel("Lnd0aa1")
        # plt.subplot(7,1,3)
        # plt.plot(range(len(lnD02Data)),lnD02Data,'b')
        # plt.ylabel("Lnd0aa2")
        # plt.subplot(7,1,4)
        # plt.plot(range(len(lnD03Data)),lnD03Data,'c')
        # plt.ylabel("Lnd0aa3")
        # plt.subplot(7,1,5)
        # plt.plot(range(len(Frac1Data)),Frac1Data,'m')
        # plt.ylabel("Frac1")
        # plt.subplot(7,1,6)
        # plt.plot(range(len(Frac2Data)),Frac2Data,'k')
        # plt.ylabel("Frac2")
        # plt.subplot(7,1,7)
        # plt.plot(range(len(Frac2Data[1:])),(optimizer.std[1:]),'r')
        # plt.xlabel("numIters")
        # plt.ylabel("STD of best values")
        # plt.show()

        
        
        
        
        # breakpoint()

        pass

if __name__ == "__main__":
    main()