#!/bin/bash

func_to_run="run_summary"
results_path=""
summary_path=""

python3.6 image_quality_assessment_util.py --func_to_run $func_to_run --results_path $results_path \
  --summary_path $summary_path