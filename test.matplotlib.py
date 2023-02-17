import matplotlib.pyplot as plt
print(plt.get_backend())
import matplotlib
matplotlib.use('module://backend_interagg')  # ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
print(plt.get_backend())


x = [1, 2, 3]
y1 = [1, 2, 3]
y2 = [2, 4, 1]
plt.plot(x, y1, label='a')
plt.plot(x, y2, label='b')
plt.legend()  # 图例
plt.xlabel('time')
plt.ylabel('distance')
plt.show()



























