import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

nA = 5


# y = np.transpose(df.values)
# plt.figure()
# df.plot()
for a in range(nA):
    file_path = 'Value' + str(a) + '.csv'
    df = pd.read_csv(file_path)
    y = df.values
    m = y.shape[0]
    ar = np.arange(m)
    s = 100
    n = int(m/s)
    reduced = []
    for i in range(n):
        reduced.append(np.mean(y[ar[i*s:(i+1)*s]]))
    plt.plot(reduced)
    plt.savefig('plot_value'+str(a)+'.png')


plt.legend(['0','1','2','3','4'],loc='best')
plt.savefig('plot_value.png')
print('done')