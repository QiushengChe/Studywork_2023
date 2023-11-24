import torch
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from sklearn.metrics import average_precision_score

test = average_precision_score
b = test(0, 0)
a = np.random.rand()
a = 1
scio.savemat("./test.mat", {'a': a})
a = torch.cuda.is_available()
debug_point = 0
