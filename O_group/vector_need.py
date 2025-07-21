import numpy as np
import os

def collect_vectors(input_dir, output_dir):
    """
    从指定输入目录收集所有唯一基底向量
    并保存为三行n列的矩阵文件
    """
    # 存储所有唯一向量
    unique_vectors = set()
    unique_start = set()  # 新增：存储所有start_vectors
    total_files = 0
    total_bases = 0
    
    print(f"开始处理目录: {input_dir}")
    base_num=0
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.startswith("base_") and os.path.isfile(os.path.join(input_dir, filename)):
            total_files += 1
            filepath = os.path.join(input_dir, filename)
            print(f"\n处理文件: {filename}")
            
            with open(filepath, 'r') as f:
                # 读取基底个数
                base_count = f.readline().strip()
                print(f"文件包含 {base_count} 个基底")
                base_num += int(base_count)
                base_index = 0
                # 处理每个基底
                for line in f:
                    # 跳过空行
                    if not line.strip():
                        continue
                    
                    # 解析基底行
                    if '*' in line and '*[' in line:
                        base_index += 1
                        total_bases += 1
                        print(f"处理基底 #{base_index}")
                        
                        try:
                            # 读取三行向量数据
                            line1 = f.readline().strip()
                            line2 = f.readline().strip()
                            line3 = f.readline().strip()
                            
                            # 跳过结束标记行
                            end_line = f.readline().strip()
                            
                            # 打印原始数据
                            print(f"原始数据: ")
                            print(f"Line1: {line1}")
                            print(f"Line2: {line2}")
                            print(f"Line3: {line3}")
                            print(f"End: {end_line}")
                            
                            # 解析分量
                            comp1 = line1.split()
                            comp2 = line2.split()
                            comp3 = line3.split()
                            
                            # 提取所有向量分量
                            vectors = [
                                (int(comp1[0]), int(comp2[0]), int(comp3[0])),  # source向量
                                (int(comp1[1]), int(comp2[1]), int(comp3[1]))   # sink向量
                            ]
                            start_vectors = [(int(comp1[1]), int(comp2[1]), int(comp3[1]))]  # 新增：记录start_vector
                            print(f"解析的向量: {vectors}")
                            
                            # 添加到唯一集合
                            unique_vectors.update(vectors)
                            unique_start.update(start_vectors)  # 新增：添加start_vector到集合
                            print(f"当前唯一向量数: {len(unique_vectors)}")
                            
                        except Exception as e:
                            print(f"解析基底错误: {e}")
    
    print(f"\n处理摘要:")
    print(f"扫描文件数: {total_files}")
    print(f"处理基底数: {total_bases}")
    print(f"找到唯一向量数: {len(unique_vectors)}")
    
    if not unique_vectors:
        print("错误: 未找到任何向量，请检查文件格式")
        return
    
    # 转换为列表并按绝对值升序排序
    vectors_list = sorted(unique_vectors, key=lambda v: (abs(v[2]), abs(v[1]), abs(v[0])), reverse=False)
    start_vectors_list = sorted(unique_start, key=lambda v: (abs(v[2]), abs(v[1]), abs(v[0])), reverse=False)
    # 创建三行n列矩阵
    matrix = np.array(vectors_list).T  # 转置为3行n列
    matrix_start = np.array(start_vectors_list).T  # 新增：start_vector矩阵
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_file = os.path.join(output_dir, "vector_need")
    
    # 保存结果
    with open(output_file, 'w') as f:
        # 第一行：base总数
        f.write(f"{base_num}\n")
        # 第二行：向量总数
        f.write(f"{len(vectors_list)}\n")
        
        # 写入矩阵数据（三行）
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")
        f.write(f"{len(start_vectors_list)}\n")  # 新增：写入start_vector数量
        for row in matrix_start:
            f.write(" ".join(map(str, row)) + "\n")


    print(f"已保存所有唯一向量到 {output_file}，共 {len(vectors_list)} 个向量")

# 收集并保存所有唯一基底向量
collect_vectors("./base_output", "./output")
