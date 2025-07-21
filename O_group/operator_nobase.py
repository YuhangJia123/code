import numpy as np
import os
import multiprocessing
from generte_Ogrpup import *
import itertools

def process_single_operator(operator, params):
    """
    处理单个算子的函数
    参数:
        operator: 单个算子
        params: 计算参数
    返回:
        处理结果
    """
    # 这里添加实际的处理逻辑
    # 示例：返回算子特征值的和
    return np.sum(operator)

def save_operators(file_path, operators_list):
    """
    保存不等长算子数组到.npz文件
    参数:
        file_path: 输出文件路径
        operators_list: 不等长算子数组列表 (list of lists of operators)
    """
    save_dict = {}
    for i, operators in enumerate(operators_list):
        # 将每个算子列表转换为numpy对象数组
        save_dict[f"operators_{i}"] = np.array(operators, dtype=object)
    np.savez(file_path, **save_dict)
    print(f"算子已保存到: {file_path}")



# 主程序
if __name__ == "__main__":
    O_group = generate_O_group_dynamically()
    irrep_num = 5
    # 使用整数类型创建start_vector
    start_vector = np.zeros((irrep_num, 3), dtype=int)
    start_vector[0] = [0,0,0]
    start_vector[1] = [0,0,1]
    start_vector[2] = [0,1,1]
    start_vector[3] = [1,1,1]
    start_vector[4] = [0,0,2]
    
    # 创建存储所有算子的列表
    all_operators = []
    operator_num=0
    # 为每个起始向量生成算子
    for i in range(irrep_num):
        operators, _ = generate_operator(O_group, start_vector[i])
        all_operators.append(operators)
        operator_num += len(operators)
        print(f"start_vector[{i}] 生成了 {len(operators)} 个算子")
    # 示例：访问第一个起始向量的所有算子
    print("第一个起始向量的算子示例:", all_operators[0][0])
    print(f"总共生成了 {operator_num} 个算子")
    # 创建输出目录
    os.makedirs("nobase_output", exist_ok=True)
    # 保存算子到文件
    save_operators("nobase_output/operators.npz", all_operators)
    
    # 示例：从文件加载算子
    loaded_operators = load_operators("nobase_output/operators.npz")
    print(f"从文件加载了 {len(loaded_operators)} 组算子")
    for i, ops in enumerate(loaded_operators):
        print(f"第{i}组包含 {len(ops)} 个算子")
    
    # # 示例：直接对不等长数组进行并行处理
    # print("\n开始并行处理所有算子...")
    # params = {"param1": 1.0, "param2": 2.0}  # 示例参数
    # nproc = 4  # 并行进程数
    
    # # 创建算子索引映射表 (group_index, operator_index)
    # operator_tasks = []
    # for group_idx, operators in enumerate(loaded_operators):
    #     for op_idx, operator in enumerate(operators):
    #         operator_tasks.append((group_idx, op_idx, operator))
    
    # # 使用进程池并行处理
    # with multiprocessing.Pool(processes=nproc) as pool:
    #     # 创建处理函数 (固定params参数)
    #     def process_task(task):
    #         group_idx, op_idx, operator = task
    #         return (group_idx, op_idx, process_single_operator(operator, params))
        
    #     # 并行处理所有算子
    #     results = pool.map(process_task, operator_tasks)
    
    # # 按原始分组结构重组结果
    # grouped_results = [[] for _ in range(len(loaded_operators))]
    # for group_idx, op_idx, result in results:
    #     grouped_results[group_idx].append(result)
    
    # # 打印处理结果
    # print("\n处理结果:")
    # for i, res in enumerate(grouped_results):
    #     print(f"第{i}组结果: {len(res)}个值, 示例: {res[0] if res else '空'}")
