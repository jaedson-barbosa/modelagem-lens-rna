import matplotlib.pyplot as plt

def plot_2_lines(a, label_a, b, label_b, name):
  figure = plt.figure()
  plt.plot(a, lw=2, color='Blue')
  plt.plot(b, lw=2, color='Red')
  plt.grid(True)
  plt.xlim()
  plt.legend([label_a, label_b])
  figure.savefig(f'./results/{name}.png')
