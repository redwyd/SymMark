import json

import matplotlib.pyplot as plt
import pandas as pd

from math import pi

A_list = []
B_list = []
C_list = []
TPR_list = []
TNR_list = []
F1_list = []
GPT4_list = []

TE_list = [0.5, 1, 1.5, 2, 2.5]
SE_list = [0.5, 1, 1.5, 2, 2.5]


for TE, SE in zip(TE_list, SE_list):
    with open(f'../radar/c4_50/seed_42/opt-1.3b/UniEXP-H/top_k_64_n_10/TE_{TE}-SE_{SE}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            A_list.append(line['A'])
            B_list.append(line['B'])
            C_list.append(line['C'])
            TPR_list.append(line['TPR'])
            TNR_list.append(line['TNR'])
            F1_list.append(line['F1'])
            GPT4_list.append(line['GPT4'])


for TE, SE in zip(TE_list, SE_list):
    with open(f'../radar/c4_50/seed_42/opt-1.3b/UniEXP-H/top_k_256_n_15/TE_{TE}-SE_{SE}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            A_list.append(line['A'])
            B_list.append(line['B'])
            C_list.append(line['C'])
            TPR_list.append(line['TPR'])
            TNR_list.append(line['TNR'])
            F1_list.append(line['F1'])
            GPT4_list.append(line['GPT4'])

# Set data
df = pd.DataFrame({
    'group': ['α=0.5 β=0.5', 'α=1.0 β=1.0', 'α=1.0 β=0.5', 'α=2.0 β=2.0', 'α=2.5 β=2.5',
              'α=0.5 β=0.5', 'α=1.0 β=1.0', 'α=1.0 β=0.5', 'α=2.0 β=2.0', 'α=2.5 β=2.5'],
    'Symbiotic': A_list,
    'F1': F1_list,
    # 'TPR': TPR_list,
    # 'TNR': TNR_list,
    'Sampling': B_list,
    'Logits': C_list,
    'GPT4': GPT4_list,
})


# ---------- 步骤1 创建背景
def make_spider(row, title, color):
    # number of variable
    # 变量类别
    categories = list(df)[1:]
    # 变量类别个数
    N = len(categories)

    # 设置每个点的角度值
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # 分图
    ax = plt.subplot(2, 5, row + 1, polar=True, )

    # If you want the first axis to be on top:
    # 设置角度偏移
    ax.set_theta_offset(pi / 2)
    # 设置顺时针还是逆时针，1或者-1
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    # 设置x轴的标签
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    # 画标签
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)

    # Ind
    # 填充数据
    values = df.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.3)

    # Add a title
    # 设置标题
    plt.title(title, size=11, color=color, y=1.15)

# ---------- 步骤2 绘制图形
plt.figure(figsize=(20, 10))

# Create a color palette:
# 设定颜色
# my_palette1 = plt.cm.get_cmap("Pastel1", len(df.index))
my_palette = plt.cm.get_cmap("tab10", len(df.index))

# Loop to plot
for row in range(0, len(df.index)):
    if row < 5:
        make_spider(row=row, title='k=64 n=10 ' + df['group'][row], color=my_palette(row))
    else:
        make_spider(row=row, title='k=256 n=15 ' + df['group'][row], color=my_palette(row))

plt.savefig('parameter_analysis.pdf', bbox_inches='tight')
plt.show()
