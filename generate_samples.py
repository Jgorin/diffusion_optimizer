from torch import tensor, rand

def create_samples(num_samples, bounds):
    """
    create a set of samples from the input space which follow the non-linear constraints.
    inputs:
        num_samples: (int) - the number of samples to create
        bounds: (tuple[tensor, tensor]) - the lower and upper bounds of the input space, 
                                          with bounds[0] == lower bounds and bounds[1] == upper bounds
    """    
    min, max = bounds
    # generate values in normalized space
    normalized_inputs = rand(num_samples, len(min))
    temp = normalized_inputs
    
    # get the number of domains and fractions
    num_domains = int((len(min) - 1) / 2) + 1
    num_frac = num_domains - 1
    
    # Assert that the first num_domains after are in descending order
    for i in range(1, 1 + num_domains):
        val = temp[:, i] * temp[:, i - 1]
        val = val + 10**-12
        normalized_inputs[:, i] = val
    
    # assert that the last num_frac inputs add up to between 0.5 and 1
    frac = rand(num_samples) * 0.5 + 0.5
    for i in range(1 + num_domains, 1 + num_domains + num_frac):
        val = normalized_inputs[:, i] * frac
        val = val + 10**-12
        normalized_inputs[:, i] = val
        frac -= val
    
    return normalized_inputs * (max - min) + min
    
if __name__ == "__main__":
    # define input space
    bounds = [
        tensor([ 50.0,    -10,   -10,   -10,   -10,   -10,   -10,   0.001, 0.001, 0.001, 0.001, 0.001 ]),
        tensor([ 150.0,  35.0,  35.0,  35.0,  35.0,  35.0,  35.0,  1.0,   1.0,   1.0,   1.0,   1.0   ])
    ]
    samples = create_samples(1000000, bounds)
    print(samples.shape)
    print(samples[0])