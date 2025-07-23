#!/bin/bash

# 重定向所有输出到master_control.log
exec &> "master_control.log"

# 定义可配置变量（便于后续修改）
MAX_TASKS=4                  # 最大并发任务数
MAX_TASKS_PER_GPU=2          # 每个GPU最大任务数
AVAILABLE_GPUS="1 2 3"       # 可用GPU编号（根据实际情况修改）

# 后台运行整个脚本
{
    # 设置关键路径和文件名模式
exe="./operate_VVV.py"              # 替换为实际的Python脚本路径
input_dir="./run_created/input/"              # 输入文件目录
output_dir="./run_created/output/"                  # 日志输出目录
error_dir="./run_created/error/"                    # 错误日志输出目录
FILE_PATTERN="${input_dir}/input_*"        # 匹配任务脚本的模式（示例：task_CONFIG1_PX10_PY20_PZ30.sh）



# 获取所有任务脚本并按自然顺序排序
files=($(ls $FILE_PATTERN 2>/dev/null | sort -V))

# 检查是否存在任务文件
if [ ${#files[@]} -eq 0 ]; then
    echo "❌ 未找到匹配的任务文件：$FILE_PATTERN"
    exit 1
fi

# 输出任务计划
echo "📋 共发现 ${#files[@]} 个任务："
echo "----------------------------------------"

# 函数：获取当前显存剩余最多的GPU编号
get_free_gpu() {
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -ne 0 ] || [ -z "$gpu_info" ]; then
        echo "0"  # 默认使用GPU 0
        return
    fi
    echo "$gpu_info" | awk -F ', ' '{print $1, $2}' | sort -k2 -nr | head -1 | cut -d' ' -f1
}

# 遍历满足 FILE_PATTERN 的所有文件
for input_file in "${files[@]}"; do
    # 提取任务标识（例如从文件名 input_CONF1_PX10_PY20_PZ30 中提取 CONF1_PX10_PY20_PZ30）
    task_id=$(basename "$input_file" | sed 's/^input_//')

    # 提取任务参数（例如从任务标识 CONF1_PX10_PY20_PZ30 中提取 CONF1、PX10、PY20、PZ30）
    CONF=$(echo "$task_id" | cut -d'_' -f1)
    PX=$(echo "$task_id" | cut -d'_' -f2)
    PY=$(echo "$task_id" | cut -d'_' -f3)
    PZ=$(echo "$task_id" | cut -d'_' -f4)

    # 生成日志路径
    log_file="${output_dir}/output_Px${PX}Py${PY}Pz${PZ}.conf${CONF}.log"
    error_log_file="${error_dir}/error_Px${PX}Py${PY}Pz${PZ}.conf${CONF}.log"


    # 初始化变量
    if [ -z "${gpu_tasks_initialized}" ]; then
        declare -A gpu_tasks
        total_tasks=0
        last_gpu=""
        available_gpus=$AVAILABLE_GPUS  # 指定可用GPU
        gpu_tasks_initialized=1
    fi

    # 等待直到满足条件
    while true; do
        # 检查全局任务数
        if [ $total_tasks -ge $MAX_TASKS ]; then
            sleep 10
            continue
        fi

        # 第一阶段：优先选择非上次使用的GPU
        best_gpu=""
        best_mem=0
        for gpu in $available_gpus; do
            # 跳过满载的GPU
            if [ ${gpu_tasks[$gpu]:-0} -ge $MAX_TASKS_PER_GPU ]; then
                continue
            fi
            # 跳过上次使用的GPU
            if [ "$gpu" = "$last_gpu" ]; then
                continue
            fi
            # 获取GPU剩余显存
            mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
            # 选择剩余显存最大的GPU
            if [ -n "$mem_free" ] && [ $mem_free -gt $best_mem ]; then
                best_gpu=$gpu
                best_mem=$mem_free
            fi
        done
        # 第二阶段：如果没找到，则考虑所有可用GPU（包括上次使用的）
        if [ -z "$best_gpu" ]; then
            for gpu in $available_gpus; do
                if [ ${gpu_tasks[$gpu]:-0} -ge $MAX_TASKS_PER_GPU ]; then
                    continue
                fi
                mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu)
                if [ -n "$mem_free" ] && [ $mem_free -gt $best_mem ]; then
                    best_gpu=$gpu
                    best_mem=$mem_free
                fi
            done
        fi

        # 找到可用GPU则跳出循环
        if [ -n "$best_gpu" ]; then
            free_gpu=$best_gpu
            break
        else
            sleep 10
        fi
    done

    # 更新任务计数器
    gpu_tasks[$free_gpu]=$(( ${gpu_tasks[$free_gpu]:-0} + 1 ))
    total_tasks=$(( total_tasks + 1 ))
    last_gpu=$free_gpu
    echo "▶️ 启动任务：$task_id (GPU $free_gpu, 当前GPU任务: ${gpu_tasks[$free_gpu]}, 总任务: $total_tasks)"
    # 执行任务并在完成后更新计数器
    (
        nohup bash -c "CUDA_VISIBLE_DEVICES=$free_gpu ipython \"$exe\" \"$input_file\"" > "$log_file" 2> "$error_log_file"
        # 任务完成后减少计数器
        gpu_tasks[$free_gpu]=$(( ${gpu_tasks[$free_gpu]} - 1 ))
        total_tasks=$(( total_tasks - 1 ))
    ) &
done

# 等待所有后台任务完成
wait

# 检查所有任务的错误日志
for input_file in "${files[@]}"; do
    task_id=$(basename "$input_file" | sed 's/^input_//')
    CONF=$(echo "$task_id" | cut -d'_' -f1)
    PX=$(echo "$task_id" | cut -d'_' -f2)
    PY=$(echo "$task_id" | cut -d'_' -f3)
    PZ=$(echo "$task_id" | cut -d'_' -f4)
    error_log_file="${output_dir}/error_Px${PX}Py${PY}Pz${PZ}.conf${CONF}.log"

    if [ -s "$error_log_file" ]; then
        echo "❌ 任务失败：$task_id"
        cat "$error_log_file"
        exit 1
    fi
done


echo "🎉 所有任务执行完毕！"
} &