import matplotlib.pyplot as plt
import scipy.io as scio

data_mat = scio.loadmat("./itex.mat")
loss_list = data_mat['loss_list'][0, :]
acc_t1_list = data_mat['acc_t1_list'][0, :]
acc_t5_list = data_mat['acc_t5_list'][0, :]
time_list = data_mat['time_list'][0, :]
itex_num = len(time_list)

data_mat_augm = scio.loadmat("./itex_augm.mat")
loss_list_augm = data_mat_augm['loss_list'][0, :]
acc_t1_list_augm = data_mat_augm['acc_t1_list'][0, :]
acc_t5_list_augm = data_mat_augm['acc_t5_list'][0, :]
time_list_augm = data_mat_augm['time_list'][0, :]
itex_num_augm = len(time_list_augm)

plt.figure(1, figsize=(10, 5))
plt.plot(range(1, itex_num + 1), loss_list, marker='*', label='Without augmentation')
plt.plot(range(1, itex_num_augm + 1), loss_list_augm, marker='*', label='With augmentation')
plt.xlim((0, max(itex_num, itex_num_augm)))
plt.title('Epoch Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('./Loss.png')

plt.figure(2, figsize=(10, 5))
plt.plot(range(1, itex_num + 1), acc_t1_list, marker='*', label='Top-1 accuracy + wo augm')
plt.plot(range(1, itex_num + 1), acc_t5_list, marker='*', color='Top-5 accuracy + wo augm')
plt.plot(range(1, itex_num_augm + 1), acc_t1_list_augm, marker='*', label='Top-1 accuracy + w augm')
plt.plot(range(1, itex_num_augm + 1), acc_t5_list_augm, marker='*', color='Top-5 accuracy + w augm')
plt.xlim((0, max(itex_num, itex_num_augm)))
plt.title('Epoch Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('./Accuracy.png')
plt.show()
