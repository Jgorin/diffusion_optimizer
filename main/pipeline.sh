#!/bin/bash
PROFILE="false"

FILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $FILE_DIR/../diffusion_optimizer/src/diffusion_optimizer

if $PROFILE != "false"
then
python3.10 interface.py \
    --output_path "$FILE_DIR/output/default_output" \
    --config_path "$FILE_DIR/config/default_config.yaml" \
    -parameter_tuning \
    -generate_inputs
else
echo "running in cproile mode"
python3.10 -m cProfile -o performance.pstats interface.py \
    --output_path "$FILE_DIR/output/default_output" \
    --config_path "$FILE_DIR/config/default_config.yaml" \
    -parameter_tuning \
    -generate_inputs
fi