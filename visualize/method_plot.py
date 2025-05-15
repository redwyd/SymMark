import json

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.lines import Line2D
from math import pi

A_list = []
B_list = []
C_list = []
D_list = []
TPR_list = []
TNR_list = []
F1_list = []
GPT4_list = []

TE_list = [0.1, 1, 1.2, 1.5, 1.5, 2, 2.5, 3]
SE_list = [0.1, 0.5, 0.8, 0.5, 1, 1, 1, 1.5]

for TE, SE in zip(TE_list, SE_list):
    with open(f'../radar/c4_50/seed_0/opt-1.3b/UniEXP-H/v0/TE_{TE}-SE_{SE}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            A_list.append(line['A'])
            B_list.append(line['B'])
            C_list.append(line['C'])
            D_list.append(line['D'])
            TPR_list.append(line['TPR'])
            TNR_list.append(line['TNR'])
            F1_list.append(line['F1'])
            GPT4_list.append(line['GPT4'])

# Set data
df1 = pd.DataFrame({
    'group': ['α=0.1 β=0.1', 'α=1.0 β=0.5', 'α=1.2 β=0.8', 'α=1.5 β=0.5', 'α=2 β=1', 'α=2 β=1.5',
              'α=2.5 β=1.0', 'α=3.0 β=1.5'],
    'Symbiotic': A_list,
    # 'Logits': C_list,
    'TPR': TPR_list,
    'TNR': TNR_list,
    'None': D_list,
    # 'Sampling': B_list,
    'GPT4': GPT4_list,
    'F1': F1_list,
})

A_list = []
B_list = []
C_list = []
D_list = []
TPR_list = []
TNR_list = []
F1_list = []
GPT4_list = []

for TE, SE in zip(TE_list, SE_list):
    with open(f'../radar/c4_50/seed_0/opt-1.3b/UniEXP-H/v1/TE_{TE}-SE_{SE}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            A_list.append(line['A'])
            B_list.append(line['B'])
            C_list.append(line['C'])
            D_list.append(line['D'])
            TPR_list.append(line['TPR'])
            TNR_list.append(line['TNR'])
            F1_list.append(line['F1'])
            GPT4_list.append(line['GPT4'])

# Set data
df2 = pd.DataFrame({
    'group': ['TE=0.1 SE=0.1', 'TE=1.0 SE=0.5', 'TE=1.2 SE=0.8', 'TE=1.5 SE=0.5', 'TE=2 SE=1', 'TE=2.5 SE=1.0',
              'TE=3.0 SE=1.5', 'TE=3.0 SE=3.0'],
    'Symbiotic': A_list,
    # 'Logits': C_list,
    'TPR': TPR_list,
    'TNR': TNR_list,
    'None': D_list,
    # 'Sampling': B_list,
    'GPT4': GPT4_list,
    'F1': F1_list,
})


# ---------- 步骤1 创建背景
def make_spider(row, title, color1, color2):
    # number of variable
    # 变量类别
    categories = list(df1)[1:]
    # 变量类别个数
    N = len(categories)

    # 设置每个点的角度值
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # 分图
    ax = plt.subplot(2, 4, row + 1, polar=True, )

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
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)

    if row == 0:
        # Ind
        # 填充数据
        values = df1.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color1, linewidth=3, linestyle='solid', label='High TE & High SE')
        ax.fill(angles, values, color=color1, alpha=0.3)

        values = df2.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color2, linewidth=3, linestyle='solid', label='High TE & Low SE')
        ax.fill(angles, values, color=color2, alpha=0.3)
    else:
        # Ind
        # 填充数据
        values = df1.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color1, linewidth=3, linestyle='solid')
        ax.fill(angles, values, color=color1, alpha=0.3)

        values = df2.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color2, linewidth=3, linestyle='solid')
        ax.fill(angles, values, color=color2, alpha=0.3)

    # Add a title
    # 设置标题
    plt.title(title, size=11, color='black', y=1.1)


# ---------- 步骤2 绘制图形
fig = plt.figure(figsize=(18, 10))

# Create a color palette:
# 设定颜色
my_palette = plt.cm.get_cmap("Set2", len(df1.index))

color1 = '#afcddc'
color2 = '#ebbba5'

# color1 = '#f9bebb'
# color2 = '#afcddc'

# Loop to plot
for row in range(0, len(df1.index)):
    make_spider(row=row, title=df1['group'][row], color1=color1, color2=color2)


# 创建色块图例
legend_elements = [Line2D([0], [0], color=color1, lw=12, label='High TE & High SE'),
                   Line2D([0], [0], color=color2, lw=12, label='High TE & Low SE')]

# 添加共用图例
fig.legend(handles=legend_elements, loc='upper center', prop={'size': 12}, frameon=False, ncol=2)

# fig.legend(loc='upper left', prop={'size': 12})
plt.savefig('method_analysis.pdf', bbox_inches='tight')
plt.show()
