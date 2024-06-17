import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
sns.set_style("darkgrid", {"axes.facecolor": ".93"})

def temper_plot_1(SOCS_RENDER, ax):
    ax.axhspan(19, 25, facecolor='#F3E5AB', alpha=0.25)
    ax.set_ylabel("Temperature (°C)", fontdict={'size': 20})
    positions = np.arange(1, 25)
    ax.set_yticks(np.arange(17, 28, step=2))
    ax.set_yticklabels(np.arange(17, 28, step=2), fontsize=14)
    ax.boxplot(SOCS_RENDER, positions=positions, patch_artist=True,
               boxprops={'facecolor': "#5875A4"}, showfliers=False)
    ax.set_xticks(positions[::2])
    ax.set_xticklabels(positions[::2],fontsize=14)
    ax.tick_params(axis='y', labelsize=15)
def temper_plot_2(SOCS_RENDER, ax):
    ax.axhspan(19, 25, facecolor='#F3E5AB', alpha=0.25)
    # ax.set_ylabel("indoor temperature (°C)", fontdict={'size': 17})
    positions = np.arange(1, 25)
    ax.set_yticks(np.arange(17, 28, step=2))
    ax.set_yticklabels(np.arange(17, 28, step=2), fontsize=14)
    ax.boxplot(SOCS_RENDER, positions=positions, patch_artist=True,
               boxprops={'facecolor': "#5875A4"}, showfliers=False)
    ax.set_xticks(positions[::2])
    ax.set_xticklabels(positions[::2],fontsize=14)
    ax.tick_params(axis='y', labelsize=15)

def temper_plot_3(SOCS_RENDER, ax):
    ax.axhspan(19, 25, facecolor='#F3E5AB', alpha=0.25)
    # ax.set_ylabel("indoor temperature (°C)", fontdict={'size': 17})
    ax.set_xlabel("Time (h)", fontdict={'size': 20})
    positions = np.arange(1, 25)
    ax.set_ylabel("Temperature (°C)", fontdict={'size': 20})
    ax.set_yticks(np.arange(17, 28, step=2))
    ax.set_yticklabels(np.arange(17, 28, step=2), fontsize=14)
    ax.boxplot(SOCS_RENDER, positions=positions, patch_artist=True,
               boxprops={'facecolor': "#5875A4"}, showfliers=False)
    ax.set_xticks(positions[::2])
    ax.set_xticklabels(positions[::2],fontsize=14)
    ax.tick_params(axis='y', labelsize=15)

def temper_plot_4(SOCS_RENDER, ax):
    ax.axhspan(19, 25, facecolor='#F3E5AB', alpha=0.25)
    # ax.set_ylabel("indoor temperature (°C)", fontdict={'size': 17})
    ax.set_xlabel("Time (h)", fontdict={'size': 20})
    positions = np.arange(1, 25)
    ax.set_yticks(np.arange(17, 28, step=2))
    ax.set_yticklabels(np.arange(17, 28, step=2), fontsize=14)
    ax.boxplot(SOCS_RENDER, positions=positions, patch_artist=True,
               boxprops={'facecolor': "#5875A4"}, showfliers=False)
    ax.set_xticks(positions[::2])
    ax.set_xticklabels(positions[::2],fontsize=14)
    ax.tick_params(axis='y', labelsize=15)

# 读入四个不同的 SOCS_RENDER 数据
SOCS_RENDER1 = np.load("QEMS_SOCS_RENDER0.npy").tolist()
SOCS_RENDER2 = np.load("QEMS_SOCS_RENDER1.npy").tolist()
SOCS_RENDER3 = np.load("baseline2_SOCS_RENDER0.npy").tolist()
SOCS_RENDER4 = np.load("baseline2_SOCS_RENDER1.npy").tolist()

# 创建 2x2 的 subplot
fig, axs = plt.subplots(2, 2, figsize=(12,6))

# 绘制每个子图
temper_plot_1(SOCS_RENDER1, axs[0, 0])
temper_plot_2(SOCS_RENDER2, axs[0, 1])
temper_plot_3(SOCS_RENDER3, axs[1, 0])
temper_plot_4(SOCS_RENDER4, axs[1, 1])

# 调整子图布局
plt.tight_layout()
# 再次调整子图布局，增加顶部空白
plt.subplots_adjust(top=0.9)
plt.savefig("QEMS_temper_combined.pdf", dpi=1500, format="pdf")

# 保存为 .pdf 文件
# plt.savefig('QEMS_temper_combined.pdf', format='pdf')
plt.show()
