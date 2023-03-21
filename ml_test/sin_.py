if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from cores import Variable
import cores.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1) # sin func

# Plot
plt.scatter(x, y, s=1)
plt.xlabel('x')
plt.ylabel('y')
# y_pred = predict(x)
# plt.plot(x.data, y_pred.data, color='r')
plt.show()