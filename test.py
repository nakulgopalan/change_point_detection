import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage.filters import minimum_filter1d
# from scipy.signal import decimate
# a = np.array([1,2,23,2,1,2134,2,13,23,2,199]).reshape((1,11))
#
# # b = minimum_filter1d(a,size=3,axis=0,mode='wrap')
#
# b = decimate(a, q=3, n=10, ftype='fir', axis=-1, zero_phase=True)
# print(b)
#
# print(a.shape)
# print(b.shape)


l_tf = np.load('/home/ng/Downloads/new_l4_odom.npy')
r_tf = np.load('/home/ng/Downloads/new_r5_odom.npy')
stop_odom = np.load('/home/ng/Downloads/stop2_odom.npy')
stop_tf = np.load('/home/ng/Downloads/stop2.npy')


print('non-sense')

arr_of_choice = r_tf

plt.plot(np.cumsum(arr_of_choice[:,0]))
# plt.show()
plt.plot(np.cumsum(arr_of_choice[:,1]))
plt.show()
