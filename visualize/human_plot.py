import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['OriginalityAI', 'Quil', 'Sapling', 'HFOpenAI', 'GPTZero', 'Crossplag', 'ZeroGPT', 'Ours']
phases = ['TNR', 'FPR']
data = [
    [0.24, 0.76],
    [0.25, 0.75],
    [0.32, 0.68],
    [0.42, 0.58],
    [0.48, 0.52],
    [0.48, 0.52],
    [0.52, 0.48],
    [1.00, 0.00],
]

# 配色
# colors = ['#98fb98', '#d8fdd8']
colors = ['#afc7e8', '#f9beb9']
# colors = ['#89c9c8', '#f9bebb']

# colors = ['#99d9d8', '#f1cacb']
# colors = ['#aecedf', '#f6dcce']
# colors = ['#afd3df', '#bae0cd']

# 绘图
fig, ax = plt.subplots(figsize=(8, 4))
width = 0.5  # 条形宽度
x = np.arange(len(categories))

# 堆叠绘制条形图
bottom_values = np.zeros(len(categories))
for i, phase in enumerate(phases):
    values = [row[i] for row in data]
    ax.barh(x, values, left=bottom_values, color=colors[i], label=phase)
    bottom_values += values

# 添加比例标识（顶部）
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.2))  # 每隔0.2设置一个刻度
ax.xaxis.set_label_position('top')  # 移动比例标签到顶部
ax.xaxis.tick_top()  # 将刻度移动到顶部
ax.set_xticklabels([f'{int(tick * 100)}%' for tick in np.arange(0, 1.1, 0.2)])

# 设置分类标签
ax.set_yticks(x)
ax.set_yticklabels(categories)

# 移除边框
# ax.spines['top'].set_visible(False)  # 隐藏顶部边框
ax.spines['right'].set_visible(False)  # 隐藏右侧边框
ax.spines['left'].set_visible(False)  # 隐藏左侧边框
ax.spines['bottom'].set_visible(False)  # 隐藏底部边框

# 隐藏底部的x轴刻度
ax.tick_params(axis='x', bottom=False)

# 添加标题
# ax.set_title('Phases of the clinical trials', pad=20)

# 将图例放在下方
ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4, frameon=False)

# 调整布局
plt.tight_layout()
plt.show()