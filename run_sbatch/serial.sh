#!/bin/bash
#SBATCH --job-name=foo
##SBATCH --mem-per-cpu=1gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu6248R
##SBATCH --gres=gpu:1

#time ./a.out
# module load cuda/11.4.4-gcc-10.3.0
#cat /proc/cpuinfo | grep 'model name'
# modinfo -F version nvidia
#free -h
# nvidia-smi
./create_slurm.sh