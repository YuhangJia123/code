import numpy as np
import os
from generte_Ogrpup import *

O_group = generate_O_group_dynamically()
irrep_num = 5
# 使用整数类型创建start_vector
start_vector = np.zeros((irrep_num, 3), dtype=int)
start_vector[0] = [0,0,0]
start_vector[1] = [0,0,1]
start_vector[2] = [0,0,2]
start_vector[3] = [1,1,1]
start_vector[4] = [0,1,1]



# 创建存储所有算子的列表
all_operators = []

# 为每个起始向量生成算子
for i in range(irrep_num):
    operators, _ = generate_operator(O_group, start_vector[i])
    all_operators.append(operators)
    print(f"start_vector[{i}] 生成了 {len(operators)} 个算子")

# 示例：访问第一个起始向量的所有算子
print("第一个起始向量的算子示例:", all_operators[0][0])

# 创建输出目录
os.makedirs("base_output", exist_ok=True)

# 处理所有[i,j]对
for i in range(irrep_num,):
    for j in range(i+1):
        #j=i
        print(f"\n处理关联函数矩阵元素 [{i},{j}]")
        # 获取source和sink算子集合
        source_ops = all_operators[i]
        sink_ops = all_operators[j]
        # 使用新的基底生成方法
        base_dict = {}
        # 遍历所有算子对
        for source in source_ops:
            for sink in sink_ops:
                # 创建2x3算子矩阵
                operator_matrix = np.array([source, sink])
                # 使用标准化的generate_operator_base函数
                base_op = generate_operator_base(O_group, start_vector[j], operator_matrix)
                if base_op is not None:
                    # 转换为可哈希的元组
                    base_tuple = (tuple(base_op[0]), tuple(base_op[1]))
                    if base_tuple in base_dict:
                        base_dict[base_tuple] += 1
                    else:
                        base_dict[base_tuple] = 1
        # 保存结果
        filename = f"./base_output/base_{i}{j}"
        save_base(base_dict, filename)
        print(f"已保存到 {filename}，包含 {len(base_dict)} 个基底")

print("\n所有关联函数矩阵元素处理完成！")

# 读取所有的基底文件并合并，得出最终所需要的所有vector


