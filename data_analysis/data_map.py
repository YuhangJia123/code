import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


moden ="jack"
data_dir=("./output/fit_%s_output" %(moden))
mode =2
# ======== 改进点1: 使用数据结构组织参数 ========
# 定义动量配置列表: (P_vector, label, color)
P_CONFIGS = [
    ([0, 0, 0], 'P=[0,0,0]', 'green'),
    ([0, 0, 1], 'P=[0,0,1]', 'yellow'),
    ([0, 1, 1], 'P=[0,1,1]', 'blue'),
    ([1, 1, 1], 'P=[1,1,1]', 'red'),
    ([0, 0, 2], 'P=[0,0,2]', 'purple')
]

# 定义误差棒配置: (x, y, yerr, color, label)
def parse_errorbars(isospin):
    """从数据文件解析误差棒数据"""
    import re
    errorbars = []
    color = 'red' if isospin == 0 else 'blue'
    
    # 匹配模式：同位旋、能级索引、E0值、误差值
    pattern = re.compile(
        r'=====\s*同位旋(\d+).*?第(\d+)个能级.*?E0=([\d.]+).*?error=([\d.]+)'
    )
    
    try:
        with open(data_dir, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 直接查找所有匹配项
        matches = pattern.findall(content)
        for match in matches:
            spin = int(match[0])
            level_idx = int(match[1])
            e0_val = float(match[2])
            err_val = float(match[3])
            
            # 只收集指定同位旋的数据
            if spin == isospin:
                errorbars.append((
                    24,
                    e0_val,
                    err_val,
                    color,
                    f'I={isospin},a{level_idx} with error'
                ))
                
    except FileNotFoundError:
        print(f"错误: 文件不存在 {data_dir}")
    except Exception as e:
        print(f"解析误差棒数据错误: {str(e)}")
    
    return errorbars

ERRORBARS_0 = parse_errorbars(0)
ERRORBARS_1 = parse_errorbars(1)
if mode == 0:
    ERRORBARS = ERRORBARS_0
elif mode == 1:
    ERRORBARS = ERRORBARS_1
elif mode == 2:
    ERRORBARS = ERRORBARS_0 + ERRORBARS_1

# ======== 改进点2: 常量定义 ========
M1 = 0.27154#格点的K质量（即之前计算的K质量）
M2 = 0.5461#格点的N质量
# M1 = 493.677*0.1053/197#格点的K质量（物理K质量）
# M2 = 938.27208816*0.1053/197#格点的N质量
# 调试输出 - 添加用户要求的调试信息
print(f"ERRORBARS_0 包含 {len(ERRORBARS_0)} 个数据点")
print(f"ERRORBARS_1 包含 {len(ERRORBARS_1)} 个数据点")
print(f"总共的误差棒数据点: {len(ERRORBARS)}")

# ======== 创建图表 ========
fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 增加图表尺寸
x = np.linspace(16, 26, 100)

# ======== 改进点3: 向量化计算 ========
# 预计算公共项
x_sq = (2 * np.pi / x) ** 2

# 循环处理所有动量配置
for P, label, color in P_CONFIGS:
    P_norm = np.dot(P, P)
    term = np.sqrt(M1**2 + x_sq * P_norm) + np.sqrt(M2**2 + x_sq * P_norm)
    ax.plot(x, term, label=label, color=color)

# 设置坐标轴范围
ax.set_xlim(16, 26)
ax.set_ylim(0.7, 1.8)

# 设置坐标轴标签
ax.set_xlabel('$L/a_s$', fontsize=12)
ax.set_ylabel('$aE_{cm}$', fontsize=12)
if mode == 0:
    ax.set_title(r"$P_max=2, I=0$", fontsize=14)
elif mode == 1:
    ax.set_title(r"$P_max=2, I=1$", fontsize=14)
elif mode == 2:
    ax.set_title(r"$P_max=0, all_I$", fontsize=14)

# 设置x轴刻度
ax.xaxis.set_major_locator(MultipleLocator(2))

# ======== 改进点4: 添加误差棒 ========
for x_pos, y_val, y_err, color, label in ERRORBARS:
    ax.errorbar(
        x=x_pos, y=y_val, yerr=y_err,
        fmt='o', color=color, label=label,
        markerfacecolor='none', capsize=3, markersize=6
    )

# ======== 关键改进: 将图例移到图表外部 ========
ax.legend(
    loc='upper left',          # 原始位置在左上角
    bbox_to_anchor=(1.05, 1), # 定位在图表右侧外部
    fontsize=10,
    frameon=True,
    fancybox=True,
    framealpha=0.8,
    borderpad=0.8
)

# ======== 改进点5: 调整布局 ========
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 右侧保留15%空白给图例

# 保存图像
if mode == 0:
    fig.savefig("./output/00_%s_map.png" %(moden), bbox_inches='tight')  # bbox_inches确保外部图例被保存
elif mode == 1:
    fig.savefig("./output/01_%s_map.png" %(moden), bbox_inches='tight')
elif mode == 2:
    fig.savefig("./output/all_I_%s_map.png" %(moden), bbox_inches='tight')  # bbox_inches确保外部图例被保存