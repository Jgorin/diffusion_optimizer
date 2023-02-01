# Diffusion Optimization

## Installation
#### Install via Docker
1. Download docker-desktop at `https://www.docker.com/` or via homebrew with `brew install docker`
2. Clone this repo with `git clone https://github.com/Jgorin/diffusion_optimizer`
3. In the root of this directory run `make build`
4. Run `make run` to create a container with the built docker image

#### Install via pip
1. Clone this repo with `git clone https://github.com/Jgorin/diffusion_optimizer`
2. In the root of this directory run `./setup.sh`
3. Run `make run` to create a container with the built docker image

## Usage
- Place any input files in `diffusion_optimization/main` to have access inside the docker container
- Set optimization settings via `diffusion_optimization/main/config/default_config.yaml`
- Run the full pipeline using the command `./main/pipeline.sh`

## YAML config structure

```yaml
nameOfInputCSVFile: "3Ddomains.csv" 
nameOfExperimentalResultsFile: "data4Optimizer.csv"

threshold: 0.01
num_dim: 6

# sample space
num_sampToTry: [3, 21]
num_ResampToTry: [3, 20]

limits:
  Ea: [70, 110]
  LnDaa1: [0, 25]
  LnD0aa2: [0, 25]
  LnD0aa3: [0, 25]
  Frac1: [0.0000000001,1]
  Frac2: [0.0000000001,1]
```

- `nameOfInputCSVFile` designates the name of the input csv file
- `nameOfExperimentalResultsFile` designates the name of the csv file for generated synthetic data.
- `threshold` convergence threshold?
- TODO finish this lul