#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gpt_test
#SBATCH --mem=32G
#SBATCH --gpus-per-node=v100:1
#SBATCH --output=gpt_test-%J.out
#SBATCH --account=def-rgmelko
#SBATCH --cpus-per-task=4

module purge

module load StdEnv/2020 apptainer/1.1.8
module load python/3.10

# Declare the Python script name as a variable
python_script_name="train.py"
dir_path="/home/jkambulo/projects/def-rgmelko/jkambulo/code/qc-temp"
data_path="/home/jkambulo/projects/def-rgmelko/jkambulo/data/rydbergGPT"
# python_script_name="examples/3_train_encoder_decoder.py"

# cd to RydbergGPT dir
# sbatch scripts/myTrain.sh
# source scripts/myTrain.sh
apptainer exec -B ${dir_path}:/code -B ${data_path}:/data --nv -H /code/RydbergGPT/src \
          ${dir_path}/RydbergGPT/container/pytorch2.sif \
          python ${python_script_name} --config_name=config_small

echo 'python program completed'