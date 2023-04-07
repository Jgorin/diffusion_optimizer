from diffusion_optimizer.neighborhood.dataset import Dataset
from diffusion_optimizer.diffusion_objective_pytorch import DiffusionObjective_pytorch
import pandas as pd
import torch as torch

dataset = Dataset(pd.read_csv("/Users/andrewgorin/diffusion_optimizer/main/output/default_output/data4Optimizer.csv"))
objective = DiffusionObjective_pytorch(dataset, [],pickle_path = "/Users/andrewgorin/diffusion_optimizer/diffusion_optimizer/src/diffusion_optimizer/lookup_table.pkl")





params = [
    torch.tensor([87.],requires_grad=True),
    torch.tensor([20.],requires_grad=True),
    torch.tensor([18.],requires_grad=True),
    torch.tensor([12.5],requires_grad=True),
    torch.tensor([0.8],requires_grad=True),
    torch.tensor([0.1],requires_grad=True)]


optimizer = torch.optim.SGD(params, lr = 0.1)


num_iters = 100000
for i in range(num_iters):
    loss = objective(params)
    print(loss)
    loss.backward()
    optimizer.step()
