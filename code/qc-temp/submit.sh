#!/bin/bash
#SBATCH --mem=30G
#SBATCH --account=def-rgmelko
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --mail-user=jkambulo@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/slurm_%J.out
#SBATCH --cpus-per-task=1

module purge
module load julia/1.8.5

cd ~/projects/def-rgmelko/jkambulo/code/qc-temp
julia simulation_main.jl --split ${i} ${tot}
