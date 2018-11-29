import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nA = 5

# y = np.transpose(df.values)
# plt.figure()
# df.plot()
# for a in range(1):
a = 0
state = 26
MazeX = 4
MazeY = 4
val_nums = 10000000 #500000
len_plot_state_num = 3
legend_label = []
for num in range(len_plot_state_num):
    # file_path = 'value_for_'+str(num)+'.csv'
    file_path = 'value_for_' + str(num) + 'sarsa' + '.csv'
    df = pd.read_csv(file_path)

    y = df.values[0:val_nums]
# position = np.unravel_index(state, (MazeX, MazeY, MazeX, MazeY))
#     m = y.shape[0]
#     ar = np.arange(m)
#     s = 100
#     n = int(m/s)
#     reduced = []
#     for i in range(n):
#         reduced.append(np.mean(y[ar[i*s:(i+1)*s]]))
#     plt.plot(reduced)
    # plt.savefig('plot_value'+str(a)+'.png')

    plt.plot(y)
    legend_label.append('state'+str(num))
plt.legend(legend_label, loc='best')
plt.savefig('plot_value_sarsa'+'.png')
plt.show()
print('done')