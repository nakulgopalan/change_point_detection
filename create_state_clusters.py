import scipy

import bnpy
import numpy as np
import os
import glob
import bnpy.data.XData as XData
from scipy.signal import savgol_filter


from matplotlib import pylab
import seaborn as sns

from bnpy.data import GroupXData
from matplotlib import pyplot as plt


from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from helper_functions import read_data
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from helper_functions import euclidean_and_cosine_similarity


# open the npz file with all the segments!
npz_file_handler = np.load('segment_run.npz')

print(npz_file_handler.files)
z_hat_list = npz_file_handler['arr_1']
# file_name_list = npz_file_handler['arr_2']
# action_indices_list = npz_file_handler['arr_3']
# empty_elements_list = npz_file_handler['arr_4']

# complete_state_data =

path_right = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_right/'
path_left = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_left/'
path_straight = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_right/'
path_4 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_left/'
path_5 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/end_of_corridor/'

list_of_paths = [path_right,path_left,path_straight, path_4, path_5]
doc_range_const = 0
z_const = 0

list_of_empty_arrays = []
list_of_action_indices = []
file_names_list = []
list_of_full_data = []

for path in list_of_paths:
    x_temp,x_prev_temp, z_temp, doc_range_temp, list_of_action_indices_temp ,list_of_empty_arrays_temp , file_names_list_temp, list_of_full_data_temp  = read_data(path,doc_range=doc_range_const,z_value=z_const)
    list_of_empty_arrays.extend(list_of_empty_arrays_temp)
    list_of_action_indices.extend(list_of_action_indices_temp)
    list_of_full_data.extend(list_of_full_data_temp)
    file_names_list.extend(file_names_list_temp)
    if z_const==0:
        x, x_prev, z, doc_range = x_temp, x_prev_temp, z_temp, doc_range_temp
    else:
        x = np.vstack((x, x_temp))
        x_prev = np.vstack((x_prev, x_prev_temp))
        z = np.hstack((z, z_temp))
        doc_range = np.hstack((doc_range[:-1], doc_range_temp))
    doc_range_const = doc_range_temp[-1]
    z_const=z_const+1

# print("right")
# x_eoc, x_prev_eoc, z_eoc, doc_range_eoc = read_data(path_right, z_value=0)
# print("straight")
# x_straight, x_prev_straight , z_straight , doc_range_straight = read_data(path_straight,doc_range=doc_range_eoc[-1],z_value=10)
# print("left")
# x_left, x_prev_left, z_left , doc_range_left = read_data(path_left,doc_range=doc_range_straight[-1],z_value=55)


# x = np.vstack((x_eoc,x_straight,x_left))
# x_prev = np.vstack((x_prev_eoc,x_prev_straight, x_prev_left))
# z = np.hstack((z_eoc, z_straight, z_left))
# doc_range = np.hstack((doc_range_eoc[:-1], doc_range_straight[:-1], doc_range_left))

print("total trajectories: ", doc_range.shape)



dataset = GroupXData(X=x[:,-5:-2],doc_range=doc_range, Xprev=x_prev[:,-5:-2]) #, TrueZ=z

trajectory_of_interest = 14
print(z_hat_list[trajectory_of_interest].shape)

# smooth z data

# read old data
# files 10 to 19 intersection and right or left and 20-25 are all at intersection!
# 0 is straight, 1 is left, 2 is right

z_hat_temp = z_hat_list[trajectory_of_interest]
z_hat_skill_change = np.ediff1d(z_hat_temp)
z_hat_skill_change_indices = np.nonzero(z_hat_skill_change)
print(z_hat_temp[z_hat_skill_change_indices])
# print(z_hat_temp[23])

print(trajectory_of_interest)
print(z_hat_skill_change_indices)


# get state data after the skill to go straight! Nothing smart just extract direct points!

trajectory_for_intersection = [10,12,14,16,18]
indices_for_intersection = [262,225,227,120,259]
trajectory_for_corridor_end = [21,22,23,24,25]
indices_for_corridor_end = [60,45,38,66,27]
print("----------")

window= 5

sentence_id_list = []

for i in range(len(indices_for_intersection)):
    trajectory_num = trajectory_for_intersection[i]
    # print(len(list_of_action_indices[trajectory_num]))
    # print(len(z_hat_list[trajectory_num]))
    index_in_trajectory = list_of_action_indices[trajectory_num][indices_for_intersection[i]]
    state_array = list_of_full_data[trajectory_num]
    # print(state_array.shape)
    # print(index_in_trajectory)
    slice_of_states_needed = state_array[index_in_trajectory-window:index_in_trajectory,:]


    nan_array = np.argwhere(np.isnan(slice_of_states_needed[:, -1]))
    slice_of_states_needed = np.delete(slice_of_states_needed, nan_array, 0)
    for _ in range(slice_of_states_needed.shape[0]):
        sentence_id_list.append(i)
    # plt.plot(slice_of_states_needed[0, :])
    # plt.show()
    # plt.plot(slice_of_states_needed[0, :])
    # plt.plot(slice_of_states_needed[-1, :])
    # plt.show()
    # print(i)
    # print(file_names_list[trajectory_for_intersection[i]])
    # plt.close()
    if i==0:
        intersection_array = slice_of_states_needed
    else:
        intersection_array = np.vstack((intersection_array,slice_of_states_needed))


for i in range(len(indices_for_corridor_end)):
    print("end number" + str(i))
    trajectory_num = trajectory_for_corridor_end[i]
    # print(len(list_of_action_indices[trajectory_num]))
    # print(len(z_hat_list[trajectory_num]))
    index_in_trajectory = list_of_action_indices[trajectory_num][indices_for_corridor_end[i]-1]
    state_array = list_of_full_data[trajectory_num]
    # print(state_array.shape)
    # print(index_in_trajectory)
    slice_of_states_needed = state_array[index_in_trajectory-window:index_in_trajectory,:]

    # plt.plot(slice_of_states_needed[0, :])
    # plt.plot(slice_of_states_needed[-1, :])
    # plt.show()


    # for j in range(slice_of_states_needed.shape[0]):
    #     plt.plot(slice_of_states_needed[j,:])
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close()


    nan_array = np.argwhere(np.isnan(slice_of_states_needed[:, -1]))
    slice_of_states_needed = np.delete(slice_of_states_needed, nan_array, 0)
    for _ in range(slice_of_states_needed.shape[0]):
        sentence_id_list.append(i+len(indices_for_intersection))
    if i==0:
        corridor_end_array = slice_of_states_needed
    else:
        corridor_end_array = np.vstack((corridor_end_array,slice_of_states_needed))


# run plain clustering to see what looks best...

print("clustering next")
full_array = np.vstack((intersection_array,corridor_end_array))


sentence_embeddings = np.load('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/lggltl/lggltl/models/torch/sentence_embeddings.npy')
print(sentence_embeddings.shape)
print(full_array.shape)
repeated_sentence_embeddings = np.zeros((full_array.shape[0],sentence_embeddings.shape[1]))
for i in range(len(sentence_id_list)):
    repeated_sentence_embeddings[i] = sentence_embeddings[sentence_id_list[i],:]

# full_array = np.hstack((full_array,repeated_sentence_embeddings))
# full_array = repeated_sentence_embeddings
list_of_labels = np.hstack((np.ones(intersection_array.shape[0]),np.zeros(corridor_end_array.shape[0])))

print(full_array.shape)
# for i in range(full_array.shape[0]):
#     plt.plot(full_array[i,:])
#     plt.show()
#
# plt.close()
if True:
    X_tsne = TSNE(learning_rate=100).fit_transform(full_array)
    X_fit = PCA().fit(full_array)
    X_pca = X_fit.transform(full_array)
    # transformer = SparsePCA(n_components=2, normalize_components=True, random_state=0)
    # transformer.fit(full_array)
    # X_pca = transformer.transform(full_array)
    # print(X_tsne.shape)
    # print(X_pca.shape)
    print("explained ratio:")
    print(X_fit.explained_variance_ratio_)
    plt.plot(X_fit.explained_variance_ratio_)
    plt.title("PCA explained ratios")
    plt.show()
    print(list_of_labels.shape)
    plt.close()

    # figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=list_of_labels)
    plt.title("TSNE")

    plt.subplot(122)
    n=0
    plt.scatter(X_pca[n:,0], X_pca[n:,1], c=list_of_labels[n:])
    plt.title("PCA")
    plt.colorbar(ticks=range(10))
    # plt.clim(-0.5, 0.5)
    plt.show()
    print(intersection_array.shape[0])
    print(corridor_end_array.shape[0])

# cluster_labels = DBSCAN(eps=.025, min_samples=5,metric='cosine').fit_predict(full_array)# , min_samples=2
cluster_labels = DBSCAN(eps=30, min_samples=10).fit_predict(full_array)# , min_samples=2

# cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)


print(cluster_labels)

# dist = DistanceMetric.get_metric('cosine')
# dis_mat = np.zeros((sentence_embeddings.shape[0],sentence_embeddings.shape[0]))
# for i in range(sentence_embeddings.shape[0]):
#     for j in range(sentence_embeddings.shape[0]):
#         dis_mat[i][j] = paired_cosine_distances(sentence_embeddings[i], sentence_embeddings[j])
# print("distances:")
# print(dis_mat)

# dist = cosine_similarity(sentence_embeddings,sentence_embeddings)
# # distances = dist.pairwise(sentence_embeddings)
# print("distances:")
# print(dist)
# # try to run PCA on a 1D data extracted from data collected on similar and dis-similar states!
#
#
# print("done!")