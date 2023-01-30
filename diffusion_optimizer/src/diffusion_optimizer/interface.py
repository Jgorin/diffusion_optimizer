import argparse
from diffusion_optimizer.parameter_tuning import tune_parameters
from diffusion_optimizer.generate_inputs import generate_inputs
import yaml
import numpy as np
import os 
import uuid
import sys

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
        tune_parameters(
            f"{args.input_csv_path}/{config['nameOfInputCSVFile']}",
            f"{args.output_path}/{config['nameOfExperimentalResultsFile']}",
            config["threshold"],
            list2range(config["num_sampToTry"]),
            list2range(config["num_ResampToTry"]),
            config["num_epsilonsToTry"],
            misfitVals=np.ones([101,101])*10*11,
            numIterations4Save=np.ones([100,101])*10*11,
            durations=np.ones([101,101])*10*11,
            names=names,
            limits=limits,
            output_dir=args.output_path
        )
    
    else:
        print("TODO: implement single pass process")
        pass

if __name__ == "__main__":
    main()