#!/bin/bash
# ====================================================
# 持久化运行核心配置
# ====================================================
# 确保脚本具有执行权限
chmod +x "$0"
# ====================================================
# 信号处理配置
# ====================================================
trap 'shutdown_hook' EXIT TERM INT  # 优雅终止信号
trap '' HUP  # 忽略终端挂断

shutdown_hook() {
    echo "🛑 收到终止信号，等待子进程退出..."
    wait  # 等待所有后台任务
    echo "💤 进程正常终止于: $(date)"
}


run_dir=.
input_dir=./run_created/input
exe=/home/jiayuhang/research_wu/projects/K+N/VVV/run/operate_VVV.py
# exe=/beegfs/home/xinghy/LapH/contraction_run/test.py
echo "2700_0_1_1 job starts at" `date` > $run_dir/output_2700_0_1_1.log
CUDA_VISIBLE_DEVICES=3 ipython $exe $input_dir/input_2700_0_1_1 >> $run_dir/output_2700_0_1_1.log 2>&1
echo "2700_0_1_1 job ends at" `date` >> $run_dir/output_2700_0_1_1.log

