source_dir="/public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P001/corr_nobase"
goal_dir="/public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P002_G1-/corr_nobase"

# 组态范围设置&动量组合设置
conf_start=4050
conf_end=4100
conf_step=50
menson=("00" "01")
moments=("01" "10" "11")



# 创建目标目录（如果不存在）
mkdir -p "$goal_dir"

# 遍历组态范围
for (( conf=conf_start; conf<=conf_end; conf+=conf_step )); do
    # 遍历动量组合
    for mom in "${moments[@]}"; do
        for men in "${menson[@]}"; do
            # 源文件名格式
            src_file="${source_dir}/${men}_P1_conf${conf}_test_${mom}.dat"
            # 目标文件名格式
            dest_file="${goal_dir}/${men}_P2_conf${conf}_${mom}.dat"
            # 检查源文件是否存在
            if [ -f "$src_file" ]; then
                # 复制文件并重命名
                cp "$src_file" "$dest_file"
                echo "已复制: $src_file -> $dest_file"
            else
                echo "警告: 源文件不存在 - $src_file"
            fi
        done
    done
done

echo "文件复制完成"
