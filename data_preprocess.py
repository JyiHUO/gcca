# sythetic data

import scipy.io
import matplotlib.pyplot as plt

# dict_keys(['__header__', '__version__', '__globals__', 'view1', 'view2', 'view3'])
# all_views["view1"] shape is [400, 2] x axis and y axis
all_views = scipy.io.loadmat('data/synthdata.mat') 

# for plot
for v in ["view1", 'view2', 'view3']:
	x = all_views[v][:, 0]
	y = all_views[v][:, 1]
	plt.scatter(x, y)
	plt.show()
