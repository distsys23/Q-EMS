from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

days = ["day "+str(i+1) for i in range(10)]
names=["Q-EMS","Baseline1","Baseline2"]
colors = ['#D62828','#00C957','#0099CC','purple', 'mediumorchid','yellow','navy','darkorange','red','green','magenta']
marks = ["^","s","p","o"]
sns.set_style("darkgrid", {"axes.facecolor": ".93"})
def day_profit():
    x = range(1, 241, 1)
    qems = np.load("QEMS_avg_soc.npy")
    baseline1 = np.load("baseline1_avg_soc.npy")
    baseline2 = np.load("baseline2_avg_soc.npy")
    meth = [qems,baseline1,baseline2]
    ax1 = plt.gca()
    ax1.set_axisbelow(True)
    #ax1.patch.set_facecolor("#EBEBEB")  # 设置 ax1 区域背景颜色
    # ax1.patch.set_alpha(0.5)  # 设置 ax1 区域背景颜色透明度
    plt.rcParams['figure.figsize'] = (8.0, 4.0)
    for i, method in enumerate(meth[:]):
        print(method)
        for j in range(240):
            method[j] = method[j]*6+19
        if i == 0:
            plt.plot(x, method, marker =marks[i], linewidth =2.0,color=colors[i], markevery=20, zorder=3)
        else:
            plt.plot(x, method, marker=marks[i], linewidth=2.0, color=colors[i], markevery=20, zorder=i)
        #line_chart.add(names[i], method)
    plt.rcParams.update({'font.size': 15})
    plt.legend(names, loc='upper right',prop={'size':13})

    plt.ylabel("Temperature (°C)",fontdict={'size':15})
    plt.xlabel("Time (h)",fontdict={'size':15})
    plt.savefig('avg_soc.pdf', format='pdf')
    # plt.title("Comparison")
    plt.show()

day_profit()
