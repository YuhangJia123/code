#!/bin/bash
# 清理base_output目录中以"base_"开头的所有文件

# 检查执行权限
if [ ! -x "$0" ]; then
    echo "错误：缺少执行权限"
    echo "请先运行以下命令添加权限："
    echo "chmod +x base_output/clear.sh"
    echo "然后再次执行：./base_output/clear.sh"
    exit 1
fi


rm -f base_* ../nobase_output/*.npz
echo "文件已删除"

