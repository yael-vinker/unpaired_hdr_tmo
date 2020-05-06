#!/bin/bash

number_of_images=1000
echo "number_of_images $number_of_images"
sbatch --mem=4000m -c2 --gres=gpu:2 --time=2-0 small_dset_run_fid.sh $noise $number_of_images