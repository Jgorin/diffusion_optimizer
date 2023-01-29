#!/bin/bash
FILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $FILE_DIR/../diffusion_optimizer/src/diffusion_optimizer

python3 interface.py \
    --config_path "$FILE_DIR/config/default_config.yaml" \
    -parameter_tuning 
    # -generate_inputs
