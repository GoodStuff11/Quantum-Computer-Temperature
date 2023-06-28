#!/bin/bash
# Batch submit jobs to Graham

tot=$1
for (( i=0; i<$tot; i++ ))
  do
    JOB_NAME="run_$(expr $i + 1)-${tot}"
    echo $JOB_NAME
    sbatch --output="slurm_${JOB_NAME}.out" --job-name $JOB_NAME --export=i=$i,tot=${tot} $2
done
sleep 0.5s