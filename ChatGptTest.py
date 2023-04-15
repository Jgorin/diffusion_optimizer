import numpy as np
from scipy.optimize import minimize
from pyDOE import lhs

from diffusion_optimizer import diffusion_objective


# Define the constraint function
def constraint_function(x):
    # Define the constraints on X
    # X[0] must be between 50 and 150
    # X[1:7] must be between -10 and 35 and must be in descending order
    # X[7:] must not add up to more than 1

    c1 = x[0] - 50
    c2 = 150 - x[0]
    c3 = np.all(x[1:7] >= -10) - 1
    c4 = np.all(x[1:6] >= x[2:7]) - 1
    c5 = np.sum(x[7:]) - 1

    return [c1, c2, c3, c4, c5]

# Set the number of input parameters and the number of samples
n_params = 12
n_samples = 10

# Define the bounds for each input parameter
bounds = [(50, 150)]
bounds += [(-10, 35)] * 6
bounds += [(0, 1)] * 5

# Generate the initial Latin hypercube samples
samples = lhs(n_params, samples=n_samples)

# Apply the constraints to each sample using the minimize function
for i in range(n_samples):
    x0 = samples[i]
    cons = ({'type': 'ineq', 'fun': constraint_function})
    #result = minimize(diffusion_objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
    #samples[i] = result.x

breakpoint()