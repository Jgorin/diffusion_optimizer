import argparse
from diffusion_optimizer.parameter_tuning import tune_parameters
from diffusion_optimizer.generate_inputs import generate_inputs
import yaml
import numpy as np
import os 

file_path = os.path.dirname(os.path.realpath(__file__))

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", required=True)
    argparser.add_argument("-generate_inputs", action='store_true')
    argparser.add_argument("-parameter_tuning", action='store_true')
    return argparser.parse_args()

def list2range(list):
    return range(list[0], list[1])

def list2tuple(list):
    return (list[0], list[1])
    
def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config_path))
    output_path = f"{file_path}/../../../main/output"
    input_path = f"{file_path}/../../../main/input"
    
    # convert arrays to tuples and limits in config
    limitDict = config["limits"]
    names = list(limitDict.keys())
    limits = [list2tuple(limit) for limit in limitDict.values()]
    
    # generate inputs
    if args.generate_inputs:
        generate_inputs(f"{input_path}/{config['nameOfInputCSVFile']}", f"{output_path}/{config['nameOfExperimentalResultsFile']}")
    
    # run parameter tuning
    if args.parameter_tuning:
        tune_parameters(
            f"{input_path}/{config['nameOfInputCSVFile']}",
            f"{output_path}/{config['nameOfExperimentalResultsFile']}",
            config["threshold"],
            list2range(config["num_sampToTry"]),
            list2range(config["num_ResampToTry"]),
            misfitVals=np.ones([101,101])*10*11,
            numIterations4Save=np.ones([100,101])*10*11,
            durations=np.ones([101,101])*10*11,
            names=names,
            limits=limits
        )
    
    else:
        print("TODO: implement single pass process")
        pass

if __name__ == "__main__":
    main()