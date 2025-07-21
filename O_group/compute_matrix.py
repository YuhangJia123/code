import os
import numpy as np

def VVV(px, py, pz):
    """动量向量函数"""
    return np.array([px, py, pz])

# 1. 从vector_need文件加载动量向量
def load_vectors_from_file(file_path):
    """从vector_need文件加载动量向量和start_vectors向量"""
    with open(file_path, 'r') as f:
        # 跳过前两行（基底总数和向量总数）
        f.readline()
        f.readline()
        # 读取三个分量的行（vector）
        x_components = list(map(int, f.readline().split()))
        y_components = list(map(int, f.readline().split()))
        z_components = list(map(int, f.readline().split()))
        
        # 跳过一行（包含startvector个数）
        num_startvectors = int(f.readline().strip())
        
        # 读取startvector的三个分量
        start_x = list(map(int, f.readline().split()))
        start_y = list(map(int, f.readline().split()))
        start_z = list(map(int, f.readline().split()))
    
    # 组合成vector列表
    vectors = []
    for i in range(len(x_components)):
        vectors.append((x_components[i], y_components[i], z_components[i]))
    
    # 组合成startvector列表
    start_vectors = []
    for i in range(len(start_x)):
        start_vectors.append((start_x[i], start_y[i], start_z[i]))
    
    return vectors, start_vectors

# 2. 预计算所有向量的VVV值
def precompute_vvv_values(vectors):
    """预计算所有向量的VVV值"""
    vvv_values = {}
    for vec in vectors:
        vvv = VVV(*vec)
        vvv_values[vec] = vvv
    return vvv_values

# 3. 解析base文件并计算矩阵元素
def compute_matrix_element(base_dir, i, j, vvv_values):
    """计算单个矩阵元素[i,j]的值"""
    filename = f"base_{i}{j}"
    file_path = os.path.join(base_dir, filename)
    if not os.path.exists(file_path):
        return 0.0
    total_value = 0.0
    with open(file_path, 'r') as f:
        # 跳过第一行（基底个数）
        f.readline()
        while True:
            # 读取系数行
            coeff_line = f.readline().strip()
            if not coeff_line:
                break
            coefficient = int(coeff_line)
            # 读取向量分量行
            comp1 = list(map(int, f.readline().split()))
            comp2 = list(map(int, f.readline().split()))
            comp3 = list(map(int, f.readline().split()))
            # 读取结束标记行
            end_line = f.readline().strip()
            # 提取source和sink向量
            source_vec = (comp1[0], comp2[0], comp3[0])
            sink_vec = (comp1[1], comp2[1], comp3[1])
            # 获取预计算的VVV值
            vvv_source = vvv_values.get(source_vec)
            vvv_sink = vvv_values.get(sink_vec)
            
            if vvv_source is not None and vvv_sink is not None:
                # 计算点积并累加
                dot_product = np.dot(vvv_source, vvv_sink)
                total_value += coefficient * dot_product
    
    return total_value

# 主函数
def main():
    # 文件路径
    vector_need_path = "output/vector_need"
    base_dir = "base_output"
    matrix_size = 5
    
    # 加载向量并预计算VVV值
    vectors, startvectors = load_vectors_from_file(vector_need_path)
    vvv_values = precompute_vvv_values(vectors)
    
    # 初始化5x5矩阵
    matrix = np.zeros((matrix_size, matrix_size))
    
    # 计算矩阵的每个元素
    for i in range(matrix_size):
        for j in range(matrix_size):
            # 文件命名规则是base_ij，其中i是行，j是列
            element_value = compute_matrix_element(base_dir, i, j, vvv_values)
            matrix[i, j] = element_value
    
    # 打印结果矩阵
    print("5x5 矩阵结果:")
    print(matrix)
    
    # 可选：保存结果到文件
    np.savetxt("result_matrix.txt", matrix, fmt="%.4f")

if __name__ == "__main__":
    main()