#!/bin/bash

FILE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd $FILE_DIR/diffusion_optimizer
pip install -e .
pip install -r requirements.txt