import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

data_mat = scio.loadmat("./data_for_task1.mat")
# data = 2 * np.random.rand(int(1e6), 1)
data = data_mat['data'][:, 0]
x = np.atleast_2d(data[:int(20000)])
p = 10  # LD arithmetic's order

N = max(x.shape)
Rx = np.zeros(N)
for k1 in range(N):
    Rx[k1] = np.dot(x[0, :N - k1], np.transpose(x[0, k1:N])) / N

a = np.zeros((p, 1 + p))
sigma = np.zeros(p)
delta = np.zeros(p)
k = np.zeros(p)

# initial
a[0, 0] = 1
a[0, 1] = -Rx[1] / Rx[0]
sigma[0] = Rx[0] + a[0, 1] * Rx[1]

for pp in range(1, p):
    a[pp, 0] = 1
    sum_tmp = 0
    for p1 in range(1, pp + 1):
        sum_tmp += a[pp - 1, p1] * Rx[pp - p1 + 1]
    delta[pp] = sum_tmp + Rx[1 + pp]
    k[pp] = -delta[pp] / sigma[pp - 1]
    a[pp, pp + 1] = k[pp]
    sigma[pp] = sigma[pp - 1] * (1 - k[pp] ** 2)
    for p2 in range(1, pp + 1):
        a[pp, p2] = a[pp - 1, p2] + k[pp] * a[pp - 1, pp - p2 + 2]

# figure
plt.figure()
plt.plot(np.arange(int(p)) + 1, sigma, marker="x", markersize=10)
plt.xlabel("Iteration")
plt.ylabel("Error Power")
plt.grid()
plt.show()
