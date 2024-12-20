#!/bin/bash
#SBATCH --job-name="permeability-cnn"
#SBATCH --output=output-a100-informed-900.out
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8000MB
#SBATCH --account=Education-EEMCS-Courses-CSEMinor

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load 2023r1
module load openmpi
module load py-numpy
module load py-scikit-learn
module load py-matplotlib
module load py-torch

python Permeability_CNN_db_informed_900.py
nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
