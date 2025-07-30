#!/bin/bash

#SBATCH --job-name=gevpjack
#SBATCH --partition=gpu-debug
#SBATCH --mail-type=end
#SBATCH --output=gevp_jack.out
#SBATCH --error=gevp_jack.out
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --cpus-per-task=1
##SBATCH --time=2:00:00
##SBATCH --nodelist=gpu012
#SBATCH --gres=gpu:1
##SBATCH --exclude=gpu013


source /public/home/jiayu/research/opt/miniconda3/etc/profile.d/conda.sh
conda activate LQCD_cupy
module load cuda/11.4.4-gcc-10.3.0
output_dir=./output
exe=/public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P002_G1-/data_analysis_nobase/gevp_jack.py
echo "gevp_jack job starts at" `date` > $output_dir/bash_jack_output.log
ipython $exe >> $output_dir/bash_jack_output.log 2>&1 
echo "gevp_jack job ends at" `date` >> $output_dir/bash_jack_output.log
