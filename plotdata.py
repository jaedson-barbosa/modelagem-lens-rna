from cProfile import label
import matplotlib.pyplot as plt


def plot_2_lines(a, label_a, b, label_b, name):
    figure = plt.figure()
    plt.plot(a, lw=2, color='Blue')
    plt.plot(b, lw=2, color='Red')
    plt.grid(True)
    plt.xlim()
    plt.legend([label_a, label_b])
    figure.savefig(f'./results/{name}.png')


def plot_n_lines(data_l, data_r, name):
    colors = ['b', 'g', 'r', 'c', 'm']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i, l in enumerate(data_l):
        ax1.plot(l[0], lw=2, color=colors[i], label=l[1])
    for i, l in enumerate(data_r):
        ax2.plot(l[0], lw=2, color=colors[i + len(data_l)], label=l[1])
    plt.grid(True)
    fig.legend()
    fig.savefig(f'./results/{name}.png')
