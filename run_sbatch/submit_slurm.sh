#!/bin/bash

#SBATCH --job-name=jyh=CONF=
#SBATCH --partition=gpu-debug
#SBATCH --mail-type=end
#SBATCH --output=lap.=CONF=.out
#SBATCH --error=lap.=CONF=.out
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --cpus-per-task=1
##SBATCH --time=2:00:00
##SBATCH --nodelist=gpu012
#SBATCH --gres=gpu:1
##SBATCH --exclude=gpu013


source /public/home/jiayu/research/opt/miniconda3/etc/profile.d/conda.sh
conda activate LQCD
module load cuda/11.4.4-gcc-10.3.0

# 检测是否为A100 GPU并设置显存限制
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA A100-SXM4-40GB" ]; then
    #export CUPY_GPU_MEMORY_LIMIT=34359738368  # 32GB in bytes
    export CUPY_GPU_MEMORY_LIMIT=33355443200  # 31GB in bytes
    echo "A100 detected: Limiting VRAM to 32GB"
fi

run_dir=.
input_dir=${run_dir}
exe=/public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/source_code/contrac_baryon_bycupy_P2_nobase.py
echo "=CONF= job starts at" `date` > $run_dir/output_=CONF=.log
ipython $exe $input_dir/input_=CONF= >> $run_dir/output_=CONF=.log 2>&1
echo "=CONF= job ends at" `date` >> $run_dir/output_=CONF=.log
