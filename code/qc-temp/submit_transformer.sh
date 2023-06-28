#!/bin/bash
#SBATCH --mem=12G
#SBATCH --account=def-rgmelko
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=jkambulo@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_%J.out
#SBATCH --cpus-per-task=1

module purge
module load python/3.10

cd ~/projects/def-rgmelko/jkambulo/code/qc-temp/transformer
python ./main.py