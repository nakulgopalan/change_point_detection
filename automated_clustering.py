import numpy as np
import os
import glob
import bnpy.data.XData as XData
from scipy.signal import savgol_filter
from scipy.signal import decimate


from matplotlib import pylab
import seaborn as sns

from bnpy.data import GroupXData
from matplotlib import pyplot as plt


from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from helper_functions import read_data_movo
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from helper_functions import euclidean_and_cosine_similarity
from scipy.ndimage.filters import minimum_filter1d


import sys
import pickle


# open the npz file with all the segments!
npz_file_handler = np.load('/media/ng/LaCie SSD/all_cleaned_bags/watercooler_tintersection_right/segment_run_movo.npz')

print(npz_file_handler.files)
# z prediction from the change point detection algorithms
z_hat_list = npz_file_handler['z_hat_list']

# for z in z_hat_list:
#     print(np.where(z==4))


# print("any z's seen???")



# Paths for simulated domain!
# path_right = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_right/'
# path_left = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_left/'
# path_straight = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_right/'
# path_4 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_left/'
# path_5 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/end_of_corridor/'
#
# list_of_paths = [path_right,path_left,path_straight, path_4, path_5]

# Paths for movo domain!
path_right = '/media/ng/LaCie SSD/all_cleaned_bags/t_right/'
path_left = '/media/ng/LaCie SSD/all_cleaned_bags/t_left/'
path_straight = '/media/ng/LaCie SSD/all_cleaned_bags/atrium/'
path_3 = '/media/ng/LaCie SSD/all_cleaned_bags/library_right/'
path_4 = '/media/ng/LaCie SSD/all_cleaned_bags/427_left/'
path_5 = '/media/ng/LaCie SSD/all_cleaned_bags/clock_double_door/'
path_6 = '/media/ng/LaCie SSD/all_cleaned_bags/firealarm_double_door/'
path_7 = '/media/ng/LaCie SSD/all_cleaned_bags/kitchen_left/'
path_8 = '/media/ng/LaCie SSD/all_cleaned_bags/kitchen_right/'
path_9 = '/media/ng/LaCie SSD/all_cleaned_bags/straight_clock/'
path_10 = '/media/ng/LaCie SSD/all_cleaned_bags/straight_firealarm/'
path_11 = '/media/ng/LaCie SSD/all_cleaned_bags/t_intersection/'
path_12 = '/media/ng/LaCie SSD/all_cleaned_bags/watercooler_tintersection_left/'
path_13 = '/media/ng/LaCie SSD/all_cleaned_bags/watercooler_tintersection_right/'



list_of_paths = [path_right,path_left,path_straight, path_3,\
                 path_4, path_5, path_6, path_7, path_8, path_9, path_10, \
                 path_11, path_12, path_13]

read_from_pickle_file = True
pickle_path = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/movo_complete_data.pickle'

if not read_from_pickle_file:
    doc_range_const = 0
    z_const = 0

    list_of_empty_arrays = []
    list_of_action_indices = []
    file_names_list = []
    list_of_full_data = []

    for path in list_of_paths:
        x_temp,x_prev_temp, z_temp, doc_range_temp, list_of_action_indices_temp ,\
        list_of_empty_arrays_temp , file_names_list_temp, list_of_full_data_temp  = \
            read_data_movo(path,doc_range=doc_range_const,z_value=z_const)
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


    # For the simulated domain we used these params
    # dataset = GroupXData(X=x[:,-5:-2],doc_range=doc_range, Xprev=x_prev[:,-5:-2]) #, TrueZ=z

    dict_obj_to_save = {}
    dict_obj_to_save['0'] = list_of_empty_arrays
    dict_obj_to_save['1'] = list_of_action_indices
    dict_obj_to_save['2'] = file_names_list
    dict_obj_to_save['3'] = list_of_full_data
    dict_obj_to_save['4'] = x
    dict_obj_to_save['5'] = x_prev
    dict_obj_to_save['6'] = z
    dict_obj_to_save['7'] = doc_range

    with open(pickle_path,"wb") as f:
        pickle.dump(dict_obj_to_save, f)
else:
    with open(pickle_path, "r") as f:
        dict_obj_to_save = pickle.load(f)
        list_of_empty_arrays = dict_obj_to_save['0']
        list_of_action_indices = dict_obj_to_save['1']
        file_names_list = dict_obj_to_save['2']
        list_of_full_data = dict_obj_to_save['3']
        x = dict_obj_to_save['4']
        x_prev = dict_obj_to_save['5']
        z = dict_obj_to_save['6']
        doc_range = dict_obj_to_save['7']

# for movo these params!
dataset = GroupXData(X=x,doc_range=doc_range, Xprev=x_prev) #, TrueZ=z




list_of_old_skill_indices = []
list_of_new_skill_indices = []
list_of_skills = []
list_of_next_skills = []

filter_length = 50
state_window = 5

for trajectory_of_interest in range(doc_range.shape[0]-1):
    # trajectory_of_interest = 13
    print("-----------------"+str(trajectory_of_interest)+"----------------")
    print("file name:", file_names_list[trajectory_of_interest])
    print(z_hat_list[trajectory_of_interest].shape)

    # smooth z data

    # read old data
    # files 10 to 19 intersection and right or left and 20-25 are all at intersection!
    # 0 is straight, 1 is left, 2 is right
    # sometimes the earliest and last time point is marked 0. We should just remove these data points


    z_hat_temp = z_hat_list[trajectory_of_interest]
    z_hat_skill_change = np.ediff1d(z_hat_temp)
    print(z_hat_temp)
    terminal_skill_flag = False
    terminal_skill_index = None
    z_hat_skill_change_indices = np.nonzero(z_hat_skill_change)[0]
    # if(z_hat_skill_change_indices.shape[0]>0):
    #     z_hat_skill_change_indices = np.insert(z_hat_skill_change_indices,0,0)
    #     z_hat_skill_change_indices = np.insert(z_hat_skill_change_indices,z_hat_skill_change_indices.shape[0],z_hat_temp.shape[0]-1)

    print(z_hat_skill_change_indices)
    for _ in range(3):
        if(z_hat_skill_change_indices.shape[0]>0):
            if(z_hat_skill_change_indices[0]<filter_length):
                #   while(z_hat_skill_change_indices[0]<filter_length & z_hat_skill_change_indices.shape[0]>0):#change ifs to whiles?
                z_hat_skill_change_indices = np.delete(z_hat_skill_change_indices,0)
    if (z_hat_skill_change_indices.shape[0] > 0):
        if(z_hat_temp.shape[0] - z_hat_skill_change_indices[-1]<filter_length):
            # while((z_hat_temp.shape[0] - z_hat_skill_change_indices[-1]<filter_length) & z_hat_skill_change_indices.shape[0]>0):
            terminal_skill_flag = True
            terminal_skill_index = z_hat_skill_change_indices[-1]
            z_hat_skill_change_indices = np.delete(z_hat_skill_change_indices,-1)
    z_hat_filter_noise = np.ediff1d(z_hat_skill_change_indices)
    index_for_deletion = np.where(z_hat_filter_noise < filter_length)[0]
    # print(index_for_deletion)
    new_arr= index_for_deletion + 1
    # print new_arr
    index_for_deletion = np.append(index_for_deletion,new_arr)
    # delete occurs two times, once for the edge starting and the other for ending the skill
    # z_hat_skill_change_indices = np.delete(z_hat_skill_change_indices,index_for_deletion)
    z_hat_skill_change_indices = np.delete(z_hat_skill_change_indices,index_for_deletion)
    print(z_hat_skill_change_indices)
    z_hat_skill_change_indices = np.append(z_hat_skill_change_indices, z_hat_temp.shape[0]-1)
    z_hat_new_skill_indices = z_hat_skill_change_indices + 1


    # here the indices are adjusted to not just pick the last state but the true terminal skill
    list_of_skills_temp = z_hat_temp[z_hat_skill_change_indices[:]]
    if(terminal_skill_flag):
        list_of_skills_temp[-1] = z_hat_temp[terminal_skill_index-1]

    #remove terminal copies which tend to happen if the switch is to 0 and we put the final index

    # if(list_of_skills_temp[-1]==list_of_skills_temp[-2]):
    #     list_of_skills_temp = np.delete(list_of_skills_temp,-2)
    #     z_hat_skill_change_indices = np.delete(z_hat_skill_change_indices, -2)
    #     z_hat_new_skill_indices = np.delete(z_hat_new_skill_indices, -2)

    list_of_old_skill_indices.append(z_hat_skill_change_indices)
    list_of_new_skill_indices.append(z_hat_new_skill_indices)
    list_of_skills.append(list_of_skills_temp)
    print(z_hat_skill_change_indices)
    print(list_of_skills_temp)

# print(z_hat_temp[z_hat_skill_change_indices])
# # print(z_hat_temp[23])
#
# print(trajectory_of_interest)
# print(z_hat_skill_change_indices)


# get state data after the skill to go straight! Nothing smart just extract direct points!
trajectories_to_avoid = [19, 67, 68, 69, 71, 72, 73, 74, 75, 79]
if True:
    # trajectory_for_intersection = [10,12,14,16,18]
    # indices_for_intersection = [262,225,227,120,259]
    # trajectory_for_corridor_end = [21,22,23,24,25]
    # indices_for_corridor_end = [60,45,38,66,27]
    # print("----------")

    sentence_id_list = []
    list_of_trajectories_for_states = []
    full_array = None
    skill_chosen = 0

    for trajectory_of_interest in range(doc_range.shape[0] - 1):
        if(trajectory_of_interest in trajectories_to_avoid):
            continue



        state_array = list_of_full_data[trajectory_of_interest][:,2:]
        skills_present = list_of_skills[trajectory_of_interest]
        change_point_for_skills = list_of_old_skill_indices[trajectory_of_interest]

        if(skill_chosen not in skills_present):
            continue

        temp_states_array = None
        for count, skill in enumerate(skills_present):
            if skill ==skill_chosen:
                index_in_trajectory = list_of_action_indices[trajectory_of_interest]\
                    [change_point_for_skills[count]]
                slice_of_states_needed = state_array[index_in_trajectory \
                                                     - state_window:index_in_trajectory, :]
                # nan_array = np.argwhere(np.isnan(slice_of_states_needed[:, -1]))
                zero_arrays = np.where(~slice_of_states_needed.any(axis=1))[0]
                slice_of_states_needed = np.delete(slice_of_states_needed, zero_arrays, 0)
                # set_of_min_states = np.amin(slice_of_states_needed,axis=0,keepdims=True)
                # for counter in range(slice_of_states_needed.shape[0]):
                #     replacement_states = minimum_filter1d(set_of_min_states, size=10, axis=1, mode='wrap')
                #     argmax_value = np.amax(slice_of_states_needed[counter,:])
                #     locations_in_states = np.where(slice_of_states_needed[counter,:] > (argmax_value - 2))[0]
                #     slice_of_states_needed[counter, locations_in_states] = replacement_states[0,locations_in_states]


                # slice_of_states_needed = decimate(slice_of_states_needed, q=3, n=3, ftype='fir', axis=-1, zero_phase=True)

                if(slice_of_states_needed.shape[0]==0):
                    print('problems in trajectory: ', file_names_list[trajectory_of_interest])
                for _ in range(slice_of_states_needed.shape[0]):
                    if(count==skills_present.size-1):
                        next_elem = "_"
                    else:
                        next_elem = skills_present[count+1]
                    sentence_id_list.append(str(trajectory_of_interest) + "-" + \
                                            str(skill) + "-" + str(next_elem) )
                    list_of_trajectories_for_states.append(file_names_list[trajectory_of_interest])
                if temp_states_array is None: # was if count = 0
                    temp_states_array = slice_of_states_needed
                else:
                    temp_states_array = np.vstack((temp_states_array, slice_of_states_needed))
        if full_array is None:
            full_array = temp_states_array
        else:
            full_array = np.vstack((full_array,temp_states_array))

        # trajectory_num = trajectory_for_intersection[i]
        # print(len(list_of_action_indices[trajectory_num]))
        # print(len(z_hat_list[trajectory_num]))
        # index_in_trajectory = list_of_action_indices[trajectory_num][indices_for_intersection[i]]
        # print(state_array.shape)
        # print(index_in_trajectory)




        # plt.plot(slice_of_states_needed[0, :])
        # plt.show()
        # plt.plot(slice_of_states_needed[0, :])
        # plt.plot(slice_of_states_needed[-1, :])
        # plt.show()
        # print(i)
        # print(file_names_list[trajectory_for_intersection[i]])
        # plt.close()



    # for i in range(len(indices_for_corridor_end)):
    #     print("end number" + str(i))
    #     trajectory_num = trajectory_for_corridor_end[i]
    #     # print(len(list_of_action_indices[trajectory_num]))
    #     # print(len(z_hat_list[trajectory_num]))
    #     index_in_trajectory = list_of_action_indices[trajectory_num][indices_for_corridor_end[i]-1]
    #     state_array = list_of_full_data[trajectory_num]
    #     # print(state_array.shape)
    #     # print(index_in_trajectory)
    #     slice_of_states_needed = state_array[index_in_trajectory-window:index_in_trajectory,:]
    #
    #     # plt.plot(slice_of_states_needed[0, :])
    #     # plt.plot(slice_of_states_needed[-1, :])
    #     # plt.show()
    #
    #
    #     # for j in range(slice_of_states_needed.shape[0]):
    #     #     plt.plot(slice_of_states_needed[j,:])
    #     #     plt.show(block=False)
    #     #     plt.pause(0.5)
    #     #     plt.close()
    #
    #
    #     nan_array = np.argwhere(np.isnan(slice_of_states_needed[:, -1]))
    #     slice_of_states_needed = np.delete(slice_of_states_needed, nan_array, 0)
    #     for _ in range(slice_of_states_needed.shape[0]):
    #         sentence_id_list.append(i+len(indices_for_intersection))
    #     if i==0:
    #         corridor_end_array = slice_of_states_needed
    #     else:
    #         corridor_end_array = np.vstack((corridor_end_array,slice_of_states_needed))


    # run plain clustering to see what looks best...

    print("clustering next")
    full_array = full_array[:,410-50:680+50] # total span of 90deg works with min support of 2 with eps of ~90
    # full_array = full_array[:, 135:225]
    # full_array = np.vstack((intersection_array,corridor_end_array))


    # sentence_embeddings = np.load('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/lggltl/lggltl/models/torch/sentence_embeddings.npy')
    # print(sentence_embeddings.shape)
    # print(full_array.shape)
    # repeated_sentence_embeddings = np.zeros((full_array.shape[0],sentence_embeddings.shape[1]))
    # for i in range(len(sentence_id_list)):
    #     repeated_sentence_embeddings[i] = sentence_embeddings[sentence_id_list[i],:]

    # full_array = np.hstack((full_array,repeated_sentence_embeddings))
    # full_array = repeated_sentence_embeddings
    # list_of_labels = np.hstack((np.ones(intersection_array.shape[0]),np.zeros(corridor_end_array.shape[0])))

    print("all the zero states: ", full_array.shape)
    # for i in range(full_array.shape[0]):
    #     print(sentence_id_list[i])
    #     print(list_of_trajectories_for_states[i])
    #     plt.plot(full_array[i,:])
    #     plt.show()

    plt.close()
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
        # plt.show()
        # print(list_of_labels.shape)
        plt.close()

        # figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(X_tsne[:,0], X_tsne[:,1])
        for i, txt in enumerate(sentence_id_list):
            plt.annotate(txt, (X_tsne[i,0], X_tsne[i,1]))
        plt.title("TSNE")

        plt.subplot(122)
        n=0
        plt.scatter(X_pca[n:,0], X_pca[n:,1])
        plt.title("PCA")
        for i, txt in enumerate(sentence_id_list):
            plt.annotate(txt, (X_pca[i,0], X_pca[i,1]))
        # plt.colorbar(ticks=range(10))

        # plt.clim(-0.5, 0.5)
        plt.show()
        # print(intersection_array.shape[0])
        # print(corridor_end_array.shape[0])

    # cluster_labels = DBSCAN(eps=.025, min_samples=5,metric='cosine').fit_predict(full_array)# , min_samples=2

    eps0, min_samples0 = 0.0001, 5
    eps1, min_samples1 = 0.35, 5
    eps2, min_samples2 = 0.01, 5
    eps3, min_samples3 = 0.0001, 10
    eps4, min_samples4 = 0.001, 5

    cluster_labels = DBSCAN(eps=eps0, min_samples=min_samples0).fit_predict(full_array)# , min_samples=2
    #
    # cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)
    #
    #
    plt.close()


    print("cluster0: ", cluster_labels)
    n = 0
    plt.scatter(X_pca[n:, 0], X_pca[n:, 1], c=cluster_labels)
    plt.title(str(eps0) + ", " + str(min_samples0))
    for i, txt in enumerate(sentence_id_list):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.colorbar(ticks=range(10))
    plt.show()

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

    cluster_labels = DBSCAN(eps=eps1, min_samples=min_samples1).fit_predict(full_array)  # , min_samples=2
    #
    # cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)
    #
    #
    plt.close()
    print("cluster1: ", cluster_labels)
    n = 0
    plt.scatter(X_pca[n:, 0], X_pca[n:, 1], c=cluster_labels)
    plt.title(str(eps1) + ", " + str(min_samples1))
    for i, txt in enumerate(sentence_id_list):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.colorbar(ticks=range(10))
    plt.show()





    cluster_labels = DBSCAN(eps=eps2, min_samples=min_samples2).fit_predict(full_array)  # , min_samples=2
    #
    # cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)
    #
    #
    plt.close()
    print("cluster2: ", cluster_labels)
    n = 0
    plt.scatter(X_pca[n:, 0], X_pca[n:, 1], c=cluster_labels)
    plt.title(str(eps2) + ", " + str(min_samples2))
    for i, txt in enumerate(sentence_id_list):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.colorbar(ticks=range(10))
    plt.show()

    cluster_labels = DBSCAN(eps=eps3, min_samples=min_samples3).fit_predict(full_array)  # , min_samples=2
    #
    # cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)
    #
    #
    plt.close()
    print("cluster3: ", cluster_labels)
    n = 0
    plt.scatter(X_pca[n:, 0], X_pca[n:, 1], c=cluster_labels)
    plt.title(str(eps3) + ", " + str(min_samples3))
    for i, txt in enumerate(sentence_id_list):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.colorbar(ticks=range(10))
    plt.show()

    cluster_labels = DBSCAN(eps=eps4, min_samples=min_samples4).fit_predict(full_array)  # , min_samples=2
    #
    # cluster_labels = DBSCAN(metric=euclidean_and_cosine_similarity, eps=30, min_samples=10).fit_predict(full_array)
    #
    #
    plt.close()
    print("cluster4: ", cluster_labels)
    n = 0
    plt.scatter(X_pca[n:, 0], X_pca[n:, 1], c=cluster_labels)
    plt.title(str(eps4) + ", " + str(min_samples4))
    for i, txt in enumerate(sentence_id_list):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.colorbar(ticks=range(10))
    plt.show()

