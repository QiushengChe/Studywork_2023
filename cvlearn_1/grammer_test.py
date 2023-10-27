import torch
import matplotlib.pyplot as plt
import scipy.io as scio

a = 1
scio.savemat("./test.mat", {'a': a})
a = torch.cuda.is_available()
debug_point = 0
