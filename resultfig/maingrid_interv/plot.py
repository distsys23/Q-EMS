import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
sns.set()
DEFAULT_UP_REG = np.genfromtxt("up_regulation.csv", delimiter=',', skip_header=1, usecols=[-1])
DEFAULT_DOWN_REG = np.genfromtxt("down_regulation.csv", delimiter=',', skip_header=1, usecols=[-1])
def sell_plot_1(ENERGY_SOLD_RENDER, ENERGY_BOUGHT_RENDER, ax):
    # Step 1: Prepare Data
    rewards = np.vstack((ENERGY_SOLD_RENDER ,ENERGY_BOUGHT_RENDER))
    Day = list(range(len(ENERGY_SOLD_RENDER)))
    methods = ['Energy Sold', 'Energy Purchased']
    data = pd.DataFrame({
        'Hour': Day * len(methods),
        'Energy':[reward for reward_list in rewards for reward in reward_list],
        'Transaction': [m for m in methods for _ in Day]
    })
    # Step 2: Create Seaborn Barplot
    sns.barplot(x='Hour', y='Energy', hue='Transaction', data=data, ax=ax,
                palette=['#5875A4', '#EFA842'],width=1)

    # Step 3: Adjust Axes and Labels
    ax.set_xlabel(" ")
    ax.set_ylabel("Energy Tran. (kWh)", fontsize=22)
    ax.tick_params(labelsize=20)
    #ax.legend(bbox_to_anchor=(0.2, 1.05), ncol=6, fontsize=18)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    # Step 4: Secondary Axis Plot
    ax2 = ax.twinx()
    ax2.plot(np.array(DEFAULT_UP_REG), linestyle='solid', color='#B7410E', linewidth=2.0)
    ax2.plot(np.array(DEFAULT_DOWN_REG), linestyle='dashed', color='#5f9e6e', linewidth=2.0)
    ax2.set_yticks([])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    #ax2.legend(['Buying prices', 'Selling prices'], bbox_to_anchor=(1, 1.05), ncol=6, fontsize=18)
    ax.legend().remove()

def sell_plot_2(ENERGY_SOLD_RENDER, ENERGY_BOUGHT_RENDER, ax):
    # Step 1: Prepare Data
    rewards = np.vstack((ENERGY_SOLD_RENDER ,ENERGY_BOUGHT_RENDER))
    Day = list(range(len(ENERGY_SOLD_RENDER)))
    methods = ['Energy Sold', 'Energy Purchased']
    data = pd.DataFrame({
        'Hour': Day * len(methods),
        'Energy':[reward for reward_list in rewards for reward in reward_list],
        'Transaction': [m for m in methods for _ in Day]
    })
    # Step 2: Create Seaborn Barplot
    sns.barplot(x='Hour', y='Energy', hue='Transaction', data=data, ax=ax,
                palette=['#5875A4', '#EFA842'],width=1)

    # Step 3: Adjust Axes and Labels
    ax.set_xlabel([])
    ax.set_ylabel(" ")
    ax.tick_params(labelsize=20)
    #ax.legend(bbox_to_anchor=(0.2, 1.05), ncol=6, fontsize=18)
    ax.yaxis.grid(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    # Step 4: Secondary Axis Plot
    ax2 = ax.twinx()
    ax2.plot(np.array(DEFAULT_UP_REG), linestyle='solid', color='#B7410E', linewidth=2.0)
    ax2.plot(np.array(DEFAULT_DOWN_REG), linestyle='dashed', color='#5f9e6e', linewidth=2.0)
    ax2.tick_params(labelsize=20)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用
    ax2.set_ylabel("Tariff (€ cents)", fontdict={'size': 25})
    #ax2.legend(['Buying prices', 'Selling prices'], bbox_to_anchor=(1, 1.05), ncol=6, fontsize=18)
    ax.legend().remove()

def sell_plot_3(ENERGY_SOLD_RENDER, ENERGY_BOUGHT_RENDER, ax):
    # Step 1: Prepare Data
    rewards = np.vstack((ENERGY_SOLD_RENDER ,ENERGY_BOUGHT_RENDER))
    Day = list(range(len(ENERGY_SOLD_RENDER)))
    methods = ['Energy Sold', 'Energy Purchased']
    data = pd.DataFrame({
        'Hour': Day * len(methods),
        'Energy':[reward for reward_list in rewards for reward in reward_list],
        'Transaction': [m for m in methods for _ in Day]
    })
    # Step 2: Create Seaborn Barplot
    sns.barplot(x='Hour', y='Energy', hue='Transaction', data=data, ax=ax,
                palette=['#5875A4', '#EFA842'],width=1)

    # Step 3: Adjust Axes and Labels
    ax.set_xlabel("Time (h)", fontsize=25)
    ax.set_ylabel("Energy Tran. (kWh)", fontsize=22)
    ax.tick_params(labelsize=20)
    #ax.legend(bbox_to_anchor=(0.2, 1.05), ncol=6, fontsize=18)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    # Step 4: Secondary Axis Plot
    ax2 = ax.twinx()
    ax2.plot(np.array(DEFAULT_UP_REG), linestyle='solid', color='#B7410E', linewidth=2.0)
    ax2.plot(np.array(DEFAULT_DOWN_REG), linestyle='dashed', color='#5f9e6e', linewidth=2.0)
    ax2.set_yticks([])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    #ax2.legend(['Buying prices', 'Selling prices'], bbox_to_anchor=(1, 1.05), ncol=6, fontsize=18)
    ax.legend().remove()

def sell_plot_4(ENERGY_SOLD_RENDER, ENERGY_BOUGHT_RENDER, ax):
    # Step 1: Prepare Data
    rewards = np.vstack((ENERGY_SOLD_RENDER, ENERGY_BOUGHT_RENDER))
    Day = list(range(len(ENERGY_SOLD_RENDER)))
    methods = ['Energy Sold', 'Energy Purchased']
    data = pd.DataFrame({
        'Hour': Day * len(methods),
        'Energy': [reward for reward_list in rewards for reward in reward_list],
        'Transaction': [m for m in methods for _ in Day]
    })
    plt.rcParams['axes.facecolor'] = '#ededed'
    plt.rcParams['figure.facecolor'] = '#ededed'
    sns.barplot(x='Hour', y='Energy', hue='Transaction', data=data, ax=ax,
                palette=['#5875A4', '#EFA842'],width=1)

    ax.set_xlabel("Time (h)", fontsize=25)
    ax.set_ylabel("Energy Tran. (kWh)", fontsize=25)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    ax.set_ylabel('')

    # Step 4: Secondary Axis Plot
    ax2 = ax.twinx()
    ax2.plot(np.array(DEFAULT_UP_REG), linestyle='solid', color='#B7410E', linewidth=2.0, label="Buying prices")
    ax2.plot(np.array(DEFAULT_DOWN_REG), linestyle='dashed', color='#5f9e6e', linewidth=2.0, label="Selling prices")
    ax2.tick_params(labelsize=20)
    ax2.grid(False, axis='y')    #ax2.set_yticks([])
    ax2.set_ylabel("Tariff (€ cents)", fontdict={'size': 25})
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))  # 对 ax 子图应用

    # Step 1: Collect Legend Handles and Labels
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Step 2: Create Combined Legend
    combined_handles = handles1 + handles2
    combined_labels = labels1 + labels2
    ax.legend().remove()

    return combined_handles, combined_labels
    # ax.legend(combined_handles, combined_labels, loc='upper right', fontsize='medium')
    # ax2.legend(['Buying prices', 'Selling prices'], bbox_to_anchor=(0.6, 1.3), ncol=6, fontsize=18)

    # Step 5: Final Customizations
    # Any other customizations...

# 读入四个不同的 SOCS_RENDER 数据
ENERGY_SOLD_RENDER0_QEMS = np.load("QEMS_ENERGY_SOLD_RENDER0.npy").tolist()
ENERGY_SOLD_RENDER7_QEMS = np.load("QEMS_ENERGY_SOLD_RENDER7.npy").tolist()
ENERGY_SOLD_RENDER0_Baseline3 = np.load("baseline1_ENERGY_SOLD_RENDER0.npy").tolist()
ENERGY_SOLD_RENDER7_Baseline3 = np.load("baseline1_ENERGY_SOLD_RENDER7.npy").tolist()
ENERGY_BU_RENDER0_QEMS = np.load("QEMS_ENERGY_BOUGHT_RENDER0.npy").tolist()
ENERGY_BU_RENDER7_QEMS = np.load("QEMS_ENERGY_BOUGHT_RENDER7.npy").tolist()
ENERGY_BU_RENDER0_Baseline3 = np.load("baseline1_ENERGY_BOUGHT_RENDER0.npy").tolist()
ENERGY_BU_RENDER7_Baseline3 = np.load("baseline1_ENERGY_BOUGHT_RENDER7.npy").tolist()
sns.set_style("darkgrid", {"axes.facecolor": ".93"})

# 创建 2x2 的 subplot
fig, axs = plt.subplots(2, 2, figsize=(18,9))

# 绘制每个子图
sell_plot_1(ENERGY_SOLD_RENDER0_QEMS, ENERGY_BU_RENDER0_QEMS, axs[0, 0])
sell_plot_2(ENERGY_SOLD_RENDER7_QEMS, ENERGY_BU_RENDER7_QEMS, axs[0, 1])
sell_plot_3(ENERGY_SOLD_RENDER0_Baseline3, ENERGY_BU_RENDER0_Baseline3, axs[1, 0])
cb,cp =sell_plot_4(ENERGY_SOLD_RENDER7_Baseline3, ENERGY_BU_RENDER7_Baseline3, axs[1, 1])
handles, labels = [], []
plt.subplots_adjust(top=0.83)
# Create Unified Legend
fig.legend(cb, cp, bbox_to_anchor=(0.91, 0.95), ncol=4, fontsize=23)
# 调整子图布局
# plt.tight_layout()

# 保存为 .pdf 文件
# plt.savefig("QEMS_sell_combined.svg", dpi=1500, bbox_inches = 'tight',format="svg")

plt.savefig('QEMS_sell_combined.pdf', format='pdf')
plt.show()
