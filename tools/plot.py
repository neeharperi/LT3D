import csv
from os.path import join, basename

import numpy as np
import matplotlib.pyplot as plt


palettes = ['#293E5C', '#b8e366', '#7b5be3', '#13C4E8', '#FAD893',  '#FF550D', '#FA72B4', '#70D4BC']

    #x-axis, y-axis
data = np.array([
    [44.8,	29.9],
    [194.0,	30.0],
    [43.5,	13.9],
    [48.2,	24.4],
    [42.4,	24.1],
    [285.7,	32.5],
])

tags = [
    '100/4 → 50',
    '100/4 → 100',
    '100/4 → 150',
    '50/8 → 50',
    '150/2 → 150',
    'Range Ensemble',
]


fig = plt.gcf()
# fig.set_size_inches(9, 5)
# plt.rcParams.update({'font.size': 13.9})

fig.set_size_inches(6.5, 6)
plt.rcParams.update({'font.size': 15})

plt.plot(
    data[:3, 0], 
    data[:3, 1], 
    '-o',
    color=palettes[7],
    linewidth=3,
    markersize=8,
    zorder=1,
)

plt.scatter(
    data[3:6, 0], 
    data[3:6, 1],
    s=50,
    c=palettes[4],
    marker='^',
    zorder=1,
)

plt.scatter(
    data[7:, 0], 
    data[7:, 1],
    s=50,
    c=palettes[5],
    marker='P',
    zorder=1,
)

colors = 4*[palettes[7]] + 3*[palettes[4]] + [palettes[5]]
offsets = np.array([
    [-30, 0.6],
    [-43, -1.5],
    [5, 0.3],
    [2, 0.5],
    [-8, -1.4],
    [-110, -0.8],
])

for i in range(len(tags)):
    plt.annotate(tags[i], data[i] + offsets[i], color=colors[i], zorder=2)

plt.xlabel('Runtime (ms)')
plt.ylabel('Accuracy (CDS)')

plt.locator_params('both', nbins=6)
plt.grid('on', linestyle='--', zorder=0)

plt.savefig("pointpillars_lotl.pdf", bbox_inches='tight')
plt.show()
