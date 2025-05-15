import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import make_interp_spline, splprep, splrep, splev, UnivariateSpline, PchipInterpolator
from scipy.signal import savgol_filter

# 设置Seaborn的样式
sns.set(style="ticks")

flag = 'other'

def uniform_sampling(lst, n):
    return [lst[i] for i in np.linspace(0, len(lst) - 1, num=n, dtype=int)]

def get_auroc_list(attack_method, num):
    kgw_fpr_list = []
    kgw_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/KGW-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            kgw_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            kgw_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            kgw_auc = line['AUROC']

    unigram_fpr_list = []
    unigram_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/Unigram-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            unigram_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            unigram_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            unigram_auc = line['AUROC']

    unbiased_fpr_list = []
    unbiased_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/Unbiased-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            unbiased_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            unbiased_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            unbiased_auc = line['AUROC']

    dip_fpr_list = []
    dip_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/DIP-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dip_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            dip_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            dip_auc = line['AUROC']

    ewd_fpr_list = []
    ewd_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/EWD-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            ewd_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            ewd_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            ewd_auc = line['AUROC']

    exp_fpr_list = []
    exp_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/EXP-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            exp_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            exp_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            exp_auc = line['AUROC']

    synthid_fpr_list = []
    synthid_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/SynthID-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            synthid_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            synthid_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            synthid_auc = line['AUROC']

    GumbelSoft_fpr_list = []
    GumbelSoft_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/EXPGumbel-I/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            GumbelSoft_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            GumbelSoft_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            GumbelSoft_auc = line['AUROC']

    ours_S_fpr_list = []
    ours_S_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/UniEXP-S/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            ours_S_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            ours_S_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            ours_S_auc = line['AUROC']

    ours_P_fpr_list = []
    ours_P_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/UniEXP-P/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            ours_P_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            ours_P_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            ours_P_auc = line['AUROC']

    ours_H_fpr_list = []
    ours_H_tpr_list = []
    with open(f'../auroc/c4_200/seed_42/opt-6.7b/UniEXP-H/{attack_method}.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            ours_H_fpr_list = np.array(uniform_sampling(line['FPRs'], num))
            ours_H_tpr_list = np.array(uniform_sampling(line['TPRs'], num))
            ours_H_auc = line['AUROC']

    if flag == 'main':
        fpr_list = [
            unigram_fpr_list,
            exp_fpr_list,
            synthid_fpr_list,
            ours_S_fpr_list,
            ours_P_fpr_list,
            ours_H_fpr_list
        ]

        tpr_list = [
            unigram_tpr_list,
            exp_tpr_list,
            synthid_tpr_list,
            ours_S_tpr_list,
            ours_P_tpr_list,
            ours_H_tpr_list,
        ]
    else:
        fpr_list = [
            kgw_fpr_list,
            ewd_fpr_list,
            dip_fpr_list,
            GumbelSoft_fpr_list,
            synthid_fpr_list,
            unbiased_fpr_list,
            ours_H_fpr_list,
        ]

        tpr_list = [
            kgw_tpr_list,
            ewd_tpr_list,
            dip_tpr_list,
            GumbelSoft_tpr_list,
            synthid_tpr_list,
            unbiased_tpr_list,
            ours_H_tpr_list,
        ]

    auc_dict = {
        'kgw': "{:.3f}".format(kgw_auc),
        'unigram': "{:.3f}".format(unigram_auc),
        'exp': "{:.3f}".format(exp_auc),
        'GumbelSoft': "{:.3f}".format(GumbelSoft_auc),
        'dip': "{:.3f}".format(dip_auc),
        'unbiased': "{:.3f}".format(unbiased_auc),
        'synthid': "{:.3f}".format(synthid_auc),
        'ewd': "{:.3f}".format(ewd_auc),
        'S': "{:.3f}".format(ours_S_auc),
        'P': "{:.3f}".format(ours_P_auc),
        'H': "{:.3f}".format(ours_H_auc)
    }

    return fpr_list, tpr_list, auc_dict

# 平滑曲线函数
def smooth_curve(x, y):
    param = np.linspace(x.min(), x.max(), x.size)
    f = make_interp_spline(param, np.c_[x, y], k=3)
    x_smooth, y_smooth = f(np.linspace(x.min(), x.max(), x.size * 100)).T

    # tck, _ = splprep([x, y], k=2, s=50)
    # x_smooth = np.linspace(0, 1, num=100, endpoint=True)
    # x_smooth, y_smooth = splev(x_smooth, tck)

    return x_smooth, y_smooth


# 绘制每条AUROC曲线
plt.figure(figsize=(20, 10))

# 颜色列表
# colors = sns.color_palette("seismic", n_colors=6)
# colors = ['#ffafb0', '#9dc3e7', '#f3b080', '#a1d99c', '#f18280']
# colors = ['#56519b', '#5baeff', '#4c9390', '#d65d48', '#f6aa54']
# colors = ['#13C487', '#F87A72', '#00B0F6', '#C5C75F', '#2666B8', '#FF7F07']
colors = ['#8dcec8', '#c2bdde', '#3480b8', '#ffbe7a', '#9bbf8a', '#6A80B9', '#ff1515']
# colors = ['#ee4035', '#f37736', '#fdf498', '#7bc043', '#0392cf']
# colors = ['#5673e0', '#00008c', '#1515ff', '#ff7575', '#ff1515', '#ff7575'] # **
# colors = ['#a1c6db', '#3375b6', '#f7ad8c', '#ff878d', '#c92721', '#ff7575']
# colors = ['#618264', '#0A3981', '#1F509A', '#D4EBF8', '#cad2c5', '#6989c2']

# colors = ['#5673e0', '#00008c', '#659287', '#ff7575', '#A5BFCC', '#ff1515']
# labels = ['KGW', 'Unigram', 'EXP', 'SynthID', 'Ours']
# labels = ['Unbiased', 'DIP', 'EWD', 'UniEXP-S', 'UniEXP-P']

# labels = ['KGW', 'EWD', 'DIP', 'SynthID', 'Unbiased', 'UniEXP-H']

markers = ['s', 'o', 'X', '<', 'D']

plt.subplot(2, 4, 1)

fpr_list, tpr_list, auc_dict = get_auroc_list('Word-D', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]

for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Word-D (ratio=0.3)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

plt.subplot(2, 4, 2)

fpr_list, tpr_list, auc_dict = get_auroc_list('Word-S-DICT', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Word-S-DICT (ratio=0.5)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

plt.subplot(2, 4, 3)

fpr_list, tpr_list, auc_dict = get_auroc_list('Word-S-BERT', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Word-S-BERT (ratio=0.5)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')


plt.subplot(2, 4, 4)

fpr_list, tpr_list, auc_dict = get_auroc_list('Copy-Paste', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Copy-Paste (CP-3-20%)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

plt.subplot(2, 4, 5)

fpr_list, tpr_list, auc_dict = get_auroc_list('Translation', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Translation (en-zh)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

plt.subplot(2, 4, 6)

fpr_list, tpr_list, auc_dict = get_auroc_list('Doc-P-GPT', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Rephrase (gpt-3.5-turbo)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

plt.subplot(2, 4, 7)

fpr_list, tpr_list, auc_dict = get_auroc_list('Doc-P-Dipper', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Rephrase (Dipper-1)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')


plt.subplot(2, 4, 8)

fpr_list, tpr_list, auc_dict = get_auroc_list('Doc-P-Dipper-1', 10)

if flag == 'main':
    labels = [
        f'Unigram (AUC: {auc_dict["unigram"]})',
        f'AAR (AUC: {auc_dict["exp"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Ours-S (AUC: {auc_dict["S"]})',
        f'Ours-P (AUC: {auc_dict["P"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]
else:
    labels = [
        f'KGW (AUC: {auc_dict["kgw"]})',
        f'EWD (AUC: {auc_dict["ewd"]})',
        f'DIP (AUC: {auc_dict["dip"]})',
        f'GumbelSoft (AUC: {auc_dict["GumbelSoft"]})',
        f'SynthID (AUC: {auc_dict["synthid"]})',
        f'Unbiased (AUC: {auc_dict["unbiased"]})',
        f'Ours-H (AUC: {auc_dict["H"]})'
    ]


for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
    # smooth_fpr, smooth_tpr = smooth_curve(fpr, tpr)
    plt.scatter(fpr, tpr, s=50, color=colors[i], edgecolors=colors[i], alpha=0.5, marker='o')
    # plt.plot(smooth_fpr, smooth_tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)
    # fit_and_plot(fpr, tpr, degree=2, color=colors[i], label=f'{label_list[i]}')
    plt.plot(fpr, tpr, '--', label=f'{labels[i]}', color=colors[i], lw=3)

# 绘制对角线（随机分类器的参考线）
plt.plot([0, 1], [0, 1], 'k--', label='Random', lw=3)


# 添加标题和标签
plt.title('Rephrase (Dipper-2)', fontsize=16)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)

# 添加图例
plt.legend(loc='lower right')

# 显示图形
# plt.grid(True)
plt.tight_layout()

if flag == 'main':
    plt.savefig('robustness.pdf')
else:
    plt.savefig('attack_appendix.pdf')

plt.show()