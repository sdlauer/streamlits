import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

fig, ax = plt.subplots()
r_values = [2,3,4,5,6,7,8,9,10,11,12]
dist = [1/36, 1/18, 1/12, 1/9, 5/36,1/6,5/36,1/9,1/12,1/18,1/36]
plt.xticks(np.arange(2.5, 12.6, step=1))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
ax.tick_params(axis=u'x', which=u'minor',length=0)
# Hide tick labels because LaTeX in Python isn't the same font
ax.set_xticklabels('')
ax.set_yticklabels('')
ax.grid(axis='y')
ax.set(axisbelow=True)
plt.rcParams['text.usetex'] = True
plt.ylim([0.0, 0.18])
plt.xlim([1.5, 12.5])
# plotting the graph 
plt.bar(r_values, dist, width=0.5)
plt.show()
plt.savefig("output.png")