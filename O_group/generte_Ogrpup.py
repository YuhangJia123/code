import numpy as np
from collections import deque

def generate_O_group_dynamically():
    # 定义生成元：绕 z 轴和 x 轴旋转90°
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    # 初始化队列和集合
    queue = deque()
    seen = set()
    I = np.eye(3)
    queue.append(I)
    seen.add(tuple(I.flatten()))
    # BFS生成所有群元素
    generators = [Rz, Rx]
    while queue:
        mat = queue.popleft()
        for gen in generators:
            new_mat = gen @ mat  # 矩阵乘法
            new_flat = tuple(new_mat.flatten())
            if new_flat not in seen:
                seen.add(new_flat)
                queue.append(new_mat)
                if len(seen) >= 24:
                    break
        if len(seen) >= 24:
            break
    # 将集合转换回矩阵列表
    return [np.array(mat).reshape(3, 3) for mat in seen]



def generate_operator(O_group_3d_rep, start_vector):
    # 生成操作符
    operators = []
    littlegroup = []
    seen_vectors = set()
    for mat in O_group_3d_rep:
        transformed_vector = mat @ start_vector
        # 四舍五入并转换为整数（旋转矩阵操作可能产生浮点误差）
        transformed_vector = np.rint(transformed_vector).astype(int)
        transformed_tuple = tuple(transformed_vector)
        if transformed_tuple not in seen_vectors:
            seen_vectors.add(transformed_tuple)
            operators.append(transformed_vector)
            littlegroup.append(mat)  # 保存对应的群元素
    print(f"Number of unique operators: {len(operators)}")
    return np.array(operators), littlegroup

def my_inv(mat):
    mat_inv = mat
    identity = np.eye(3)  # 单位矩阵
    while not np.allclose(mat_inv @ mat, identity):  # 检查是否接近单位矩阵
        mat_inv = mat_inv @ mat  # 不断相乘
    return mat_inv



def generate_operator_base(O_group, start_vector, opertaor):
    if opertaor.shape != (2,3): #(2*3)代表是一个向量组（两个向量），这两个向量共同转动，将第二个向量转为start_vector,并且用小群作用，根据xyz由小到大排序，得到最终的算子。
        raise ValueError("Operator must be a 2x3 matrix representing two vectors.")
    final_operator=[]
    for mat in O_group:
        # 分别变换两个向量
        transformed_vector1 = mat @ opertaor[0]
        transformed_vector1 = np.rint(transformed_vector1).astype(int)
        transformed_vector2 = mat @ opertaor[1]
        transformed_vector2 = np.rint(transformed_vector2).astype(int)
        
        if np.array_equal(transformed_vector2, start_vector):
            # 组合变换后的向量形成新算子
            new_operator = np.array([transformed_vector1, transformed_vector2])
            final_operator.append(new_operator)
    print(f"Number of final operators: {len(final_operator)}")
    #排序选出最后的最佳算子，排序方式为从z开始比较，选出最大的，如果z相同则比较y，如果y相同则比较x
    final_operator = sorted(final_operator, key=lambda x: (x[0][2], x[0][1], x[0][0]), reverse=True)
    f_operator=final_operator[0] if final_operator else None
    return f_operator


# 定义保存基底函数
def save_base(base_dict, filename):
    # """
    # 将基底字典保存到文件
    # 格式：
    #     第一行：基底个数
    #     后续行：系数*元组（元组用三行二列矩阵表示）
    # """
    with open(filename, 'w') as f:
        # 第一行：基底个数
        f.write(f"{len(base_dict)}\n")
        # 写入每个基底
        for base_tuple, count in base_dict.items():
            # base_tuple 是标准化后的算子 (2x3矩阵)
            source = base_tuple[0]
            sink = base_tuple[1]
            # 写入系数和元组开始标记
            f.write(f"{count}*[\n")
            # 写入三行二列矩阵
            # 第一行：source_x, sink_x
            f.write(f"{source[0]} {sink[0]}\n")
            # 第二行：source_y, sink_y
            f.write(f"{source[1]} {sink[1]}\n")
            # 第三行：source_z, sink_z
            f.write(f"{source[2]} {sink[2]}\n")
            # 写入元组结束标记
            f.write("]\n")

def load_operators(file_path):
    """
    从.npz文件加载不等长算子数组
    参数:
        file_path: 输入文件路径
    返回:
        list of lists of operators
    """
    data = np.load(file_path, allow_pickle=True)
    operators_list = []
    i = 0
    while f"operators_{i}" in data:
        operators_list.append(data[f"operators_{i}"].tolist())
        i += 1
    return operators_list