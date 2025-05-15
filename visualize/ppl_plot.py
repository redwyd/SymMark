import json
import matplotlib.pyplot as plt
import numpy as np

with open('../ppl/c4_100/seed_42/natural.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    natural = line['natural']

with open('../ppl/c4_100/seed_42/opt-1.3b/Unigram-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_1_unwatermarked = line['unwatermarked']
    opt_1_KGW = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-1.3b/EXP-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_1_unwatermarked = line['unwatermarked']
    opt_1_EXP = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-1.3b/UniEXP-S.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_1_unwatermarked = line['unwatermarked']
    opt_1_KgwExp_S = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-1.3b/UniEXP-P.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_1_unwatermarked = line['unwatermarked']
    opt_1_KgwExp_P = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-1.3b/UniEXP-H.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_1_unwatermarked = line['unwatermarked']
    opt_1_KgwExp_H = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-2.7b/Unigram-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_2_unwatermarked = line['unwatermarked']
    opt_2_KGW = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-2.7b/EXP-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_2_unwatermarked = line['unwatermarked']
    opt_2_EXP = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-2.7b/UniEXP-S.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_2_unwatermarked = line['unwatermarked']
    opt_2_KgwExp_S = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-2.7b/UniEXP-P.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_2_unwatermarked = line['unwatermarked']
    opt_2_KgwExp_P = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-2.7b/UniEXP-H.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_2_unwatermarked = line['unwatermarked']
    opt_2_KgwExp_H = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-6.7b/Unigram-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_3_unwatermarked = line['unwatermarked']
    opt_3_KGW = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-6.7b/EXP-I.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    opt_3_unwatermarked = line['unwatermarked']
    opt_3_EXP = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-6.7b/UniEXP-S.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_3_unwatermarked = line['unwatermarked']
    opt_3_KgwExp_S = line['watermarked']

with open('../ppl/c4_100/seed_42/opt-6.7b/UniEXP-P.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_3_unwatermarked = line['unwatermarked']
    opt_3_KgwExp_P = line['watermarked']


with open('../ppl/c4_100/seed_42/opt-6.7b/UniEXP-H.jsonl', 'r') as f:
    lines = f.readlines()
    line = json.loads(lines[0])
    # opt_3_unwatermarked = line['unwatermarked']
    opt_3_KgwExp_H = line['watermarked']


# Data for plotting
categories = ['Human', 'OPT-1.3B', 'OPT-2.7B', 'OPT-6.7B']
data = {
    'Human': [natural],
    'OPT-1.3B': [
        opt_1_unwatermarked,
        opt_1_KGW,
        opt_1_EXP,
        opt_1_KgwExp_S,
        opt_1_KgwExp_P,
        opt_1_KgwExp_H,
    ],
    'OPT-2.7B': [
        opt_2_unwatermarked,
        opt_2_KGW,
        opt_2_EXP,
        opt_2_KgwExp_S,
        opt_2_KgwExp_P,
        opt_2_KgwExp_H
    ],
    'OPT-6.7B': [
        opt_3_unwatermarked,
        opt_3_KGW,
        opt_3_EXP,
        opt_3_KgwExp_S,
        opt_3_KgwExp_P,
        opt_3_KgwExp_H
    ]
}
# colors = ['#5679BA', '#66BC98', '#AAD09D', '#E3EA96', '#FCDC89', '#E26844', '#EE553D']
# colors = ['#e54c35', '#92d2c3', '#059a89', '#475378', '#f29c7f', '#8491b2', '#53b9d8']
# colors = ['#6e88b0', '#e1a259', '#d37474', '#92bdbb', '#7cab6f', '#e8d074', '#b38dab']
colors = ['#66BC98', '#6a87c2', '#c3e4f5', '#fce594', '#f4cfd6', '#f2a7a2', '#eb6468']
# labels = ['Human', 'Un-watermarked', 'KGW Watermark', 'EXP Watermark', 'KgwExp-S Watermark', 'KgwExp-P Watermark',
#           'KgwExp-H Watermark']
labels = ['Human', 'Un-watermarked', 'Unigram Watermark', 'EXP Watermark', 'Ours-S Watermark', 'Ours-P Watermark',
          'Ours-H Watermark']


# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
positions = [0.5]  # Start position for Human
width = 0.15  # Width of each boxplot group
gap = 0.3  # Gap between groups
positions.extend(np.arange(1, len(categories)) * (len(labels) * width + gap))

print(positions)

for i, category in enumerate(categories):
    for j, box_data in enumerate(data[category]):
        pos = positions[i] + j * width
        if i == 0:
            bplot = ax.boxplot(
                box_data,
                positions=[pos],
                widths=width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor=colors[j], color='black'),
                medianprops=dict(color='black')
            )
            break
        bplot = ax.boxplot(
            box_data,
            positions=[pos],
            widths=width * 0.8,
            patch_artist=True,
            boxprops=dict(facecolor=colors[j + 1], color='black'),
            medianprops=dict(color='black')
        )

# Adjusting plot
ax.set_xticks([0.5, 1.72, 3.07, 4.42])
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim([0, 70])
ax.set_ylabel('Text Perplexity', fontsize=13)
# ax.set_xlabel('Model Type', fontsize=13)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors], labels, loc='upper left', ncol=1, frameon=False)
plt.title("")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('ppl_uniexp.pdf')
plt.show()

