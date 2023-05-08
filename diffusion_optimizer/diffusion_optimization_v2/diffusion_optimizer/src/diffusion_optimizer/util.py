# load dataset
import pandas as pd
import numpy as np
from diffusion_optimizer.optimization.diffusion_objective import DiffusionObjective
import pickle
import torch

dataset = pd.read_parquet("/Users/josh/repos/drew_nn_opt/params.parquet")

lookup_table = pickle.load(open("/Users/josh/repos/diffusion_optimization_v2/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl", "rb"))

print(len(dataset))
removed_row_indices = []
# remove all rows with -1 in the x column
for i in range(len(dataset)):
    # check if the row has any negatives
    negs = np.where(dataset.iloc[i, 0] < 0)[0]
    # if negs is not empty, remove the row
    if len(negs) > 0:
        removed_row_indices.append(i)

# remove the rows
dataset = dataset.drop(removed_row_indices)

# try running each row through the model
# create diffusion objective
do = DiffusionObjective(pd.read_csv("/Users/josh/repos/diffusion_optimization_v2/demos/single_run/data4Optimizer.csv"))

# loop through each row in the dataset and try running it through the model. if it fails, remove it
removed_row_indices = []
for i in range(len(dataset)):
    res = do(torch.tensor(dataset.iloc[i, 0]), lookup_table)
    if res != 0:
        removed_row_indices.append(i)
        # print that a row is being removed and for what reason
        print("removing row: ", i, " because: ", res)

dataset = dataset.drop(removed_row_indices)
print("removed rows: ", len(removed_row_indices))
        
# save dataset as params_edited.parquet
dataset.to_parquet("/Users/josh/repos/drew_nn_opt/params_edited.parquet")