import bnpy
import numpy as np
import os
import glob
import bnpy.data.XData as XData
from scipy.signal import savgol_filter


from matplotlib import pylab
import seaborn as sns

from bnpy.data import GroupXData
import matplotlib
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances


def read_data_movo(path, doc_range=0,z_value=0):
    os.chdir(path)
    doc_range = [doc_range]
    extension = 'npy'
    eoc_files = glob.glob('*.{}'.format(extension))
    print(path)

    x = None
    x_prev = None
    z = None
    list_of_empty_arrays = []
    list_of_action_indices = []
    list_of_complete_data = []
    file_names_list = []
    for i in eoc_files:
        print(i)
        # i = i+3
        # string_path = path+str(i)+'.npy'
        file_names_list.append(path + i)

        data = np.load(path + i)  # .transpose()

        # raw_data = np.load(path + i[:-3]+'full.npy').transpose()
        print(data.shape)
        # print(raw_data.shape)
        list_of_complete_data.append(data)
        # shape is samples x dim ( = 13) now

        # data cleaning:  removed all nan's, for actions the nans are zeros as there is no velocity
        # and for lidar data it is the previously noted lidar measurement!

        # check places where nan in the first elememnts and zeros in the last 6

        deleted_elem_array = np.array(range(data.shape[0]))

        elements_without_action_data = np.where(~data[:, 0:2].any(axis=1))[0]
        # last_six_zeros = np.where(~data[:,-6:].any(axis=1))[0]
        #
        # common_elements = np.intersect1d(last_six_zeros, nan_array)

        data = np.delete(data, elements_without_action_data, 0)
        deleted_elem_array = np.delete(deleted_elem_array, elements_without_action_data, 0)

        # nan_array = np.argwhere(np.isnan(data[:, 0]))
        #
        # if 0 in nan_array:
        #     nan_array = np.delete(nan_array, 0, 0)
        #     data = np.delete(data, 0, 0)
        #     nan_array = nan_array - 1

        # check if entire row is zeros delete them
        # empty_array = np.where(~data[:,0:2].any(axis=1))[0]
        # data = np.delete(data,empty_array,0)
        # deleted_elem_array = np.delete(deleted_elem_array,empty_array,0)
        #
        # list_of_empty_arrays.append(empty_array)
        list_of_action_indices.append(deleted_elem_array)

        elements_greater_than_0 = np.argwhere(data[:, 1] > 0.)
        elements_lesser_than_0 = np.argwhere(data[:, 1] < 0.)
        action_data = data[:, 0:3]
        action_data[:, 2] = action_data[:, 1] * 1.
        action_data[elements_greater_than_0, 1] = 0.
        action_data[elements_lesser_than_0, 2] = 0.

        # data = np.nan_to_num(data)
        # data[nan_array, 0:-6] = data[nan_array - 1, 0:-6]

        # while True:
        #     lidar_state_zero = np.where(~data[:, :-6].any(axis=1))[0]
        #     if(lidar_state_zero.size == 0):
        #         break
        #     data[lidar_state_zero, 0:-6] = data[lidar_state_zero - 1, 0:-6]

        # matplotlib.pyplot.plot(data[:,-4]) # 3 is for 0 deg -4 was interesting and -3
        #
        # matplotlib.pyplot.show()
        # data = data * 100

        action_data = np.cumsum(action_data, 0)

        # action_data = moving_average(action_data, n=10)

        data_prev = np.vstack([action_data[0, :], action_data[0:-1, :]])
        doc_range.append(doc_range[-1] + action_data.shape[0])
        if x is not None:
            x = np.vstack((x, action_data))
            x_prev = np.vstack((x_prev, data_prev))
            z = np.hstack((z, np.ones(action_data.shape[0]) * z_value))
            # if i < 2:
            #     z = np.hstack((z, np.zeros(data.shape[0])))
            # else:
            #     z = np.hstack((z, np.ones(data.shape[0])))

        else:
            x = action_data
            x_prev = data_prev
            z = np.ones(action_data.shape[0]) * z_value
    # matplotlib.pyplot.show()
    print(doc_range)

    return (x, x_prev, z, doc_range, list_of_action_indices\
                , list_of_empty_arrays, file_names_list, list_of_complete_data)


def read_data_simulated(path, doc_range=0, z_value=0):
    os.chdir(path)
    doc_range = [doc_range]
    extension = 'csv.npy'
    eoc_files = glob.glob('*.{}'.format(extension))
    print(path)

    x = None
    x_prev = None
    z = None
    list_of_empty_arrays = []
    list_of_action_indices = []
    list_of_complete_data = []
    file_names_list = []
    for i in eoc_files:
        print(i)
        # i = i+3
        # string_path = path+str(i)+'.npy'
        file_names_list.append(path+i)

        data = np.load(path + i).transpose()

        raw_data = np.load(path + i[:-3]+'full.npy').transpose()
        print(data.shape)
        print(raw_data.shape)
        list_of_complete_data.append(raw_data)
        # shape is samples x dim ( = 13) now

        # data cleaning:  removed all nan's, for actions the nans are zeros as there is no velocity
        # and for lidar data it is the previously noted lidar measurement!

        #check places where nan in the first elememnts and zeros in the last 6

        deleted_elem_array = np.array(range(data.shape[0]))


        nan_array = np.argwhere(np.isnan(data[:, -1]))
        # last_six_zeros = np.where(~data[:,-6:].any(axis=1))[0]
        #
        # common_elements = np.intersect1d(last_six_zeros, nan_array)

        data = np.delete(data, nan_array, 0)
        deleted_elem_array = np.delete(deleted_elem_array,nan_array,0)

        # nan_array = np.argwhere(np.isnan(data[:, 0]))
        #
        # if 0 in nan_array:
        #     nan_array = np.delete(nan_array, 0, 0)
        #     data = np.delete(data, 0, 0)
        #     nan_array = nan_array - 1



        # check if entire row is zeros delete them
        empty_array = np.where(~data[:,-6:].any(axis=1))[0]
        data = np.delete(data,empty_array,0)
        deleted_elem_array = np.delete(deleted_elem_array,empty_array,0)

        list_of_empty_arrays.append(empty_array)
        list_of_action_indices.append(deleted_elem_array)

        elements_greater_than_0 = np.argwhere(data[:,-4]>0.)
        elements_lesser_than_0 = np.argwhere(data[:,-4]<0.)
        data[:,-5] = data[:,-4] *1.;
        data[elements_greater_than_0,-5] = 0.
        data[elements_lesser_than_0,-4] = 0.






        # data = np.nan_to_num(data)
        # data[nan_array, 0:-6] = data[nan_array - 1, 0:-6]

        # while True:
        #     lidar_state_zero = np.where(~data[:, :-6].any(axis=1))[0]
        #     if(lidar_state_zero.size == 0):
        #         break
        #     data[lidar_state_zero, 0:-6] = data[lidar_state_zero - 1, 0:-6]



        # matplotlib.pyplot.plot(data[:,-4]) # 3 is for 0 deg -4 was interesting and -3
        #
        # matplotlib.pyplot.show()
        # data = data * 100
        data = np.cumsum(data,0)

        data_prev = np.vstack([data[0, :], data[0:-1, :]])
        doc_range.append(doc_range[-1] + data.shape[0])
        if x is not None:
            x = np.vstack((x, data))
            x_prev = np.vstack((x_prev, data_prev))
            z = np.hstack((z, np.ones(data.shape[0]) * z_value))
            # if i < 2:
            #     z = np.hstack((z, np.zeros(data.shape[0])))
            # else:
            #     z = np.hstack((z, np.ones(data.shape[0])))

        else:
            x = data
            x_prev = data_prev
            z = np.ones(data.shape[0]) * z_value
    # matplotlib.pyplot.show()
    print(doc_range)

    return (x,x_prev,z, doc_range, list_of_action_indices,list_of_empty_arrays, file_names_list, list_of_complete_data)



def moving_average(a, n=3) :
    ret = np.cumsum(a,axis=0, dtype=float)
    ret[n:,:] = ret[n:,:] - ret[:-n,:]
    return ret[n - 1:,:] / n




def euclidean_and_cosine_similarity(X, Y):
    return paired_euclidean_distances(X[:270].reshape(1, -1), Y[:270].reshape(1, -1))+paired_cosine_distances(X[270:].reshape(1, -1), Y[270:].reshape(1, -1))
