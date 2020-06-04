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
from helper_functions import read_data_movo
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric
from helper_functions import euclidean_and_cosine_similarity

import csv
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


#all paths!
list_of_paths = [path_right,path_left,path_straight, path_3,\
                 path_4, path_5, path_6, path_7, path_8, path_9, path_10, \
                 path_11, path_12, path_13]

#without firealarm!
# list_of_paths = [path_right,path_left,path_straight, path_3,\
#                  path_4, path_5, path_7, path_8, path_9, \
#                  path_11, path_12, path_13]

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


    list_of_old_skill_indices.append(z_hat_skill_change_indices)
    list_of_new_skill_indices.append(z_hat_new_skill_indices)
    list_of_skills.append(list_of_skills_temp)
    print(z_hat_skill_change_indices)
    print(list_of_skills_temp)



# get state data after the skill to go straight! Nothing smart just extract direct points!
trajectories_to_avoid = [19, 67, 68, 69, 71, 72, 73, 74, 75, 79]
if True:
    pickle_path_0 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-0-symbol_data_without_firealarm_88.pickle'
    pickle_path_1 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-1-symbol_data_without_firealarm_88.pickle'
    pickle_path_2 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-2-symbol_data_without_firealarm_88.pickle'


    with open(pickle_path_0, "r") as f:
        dict_obj_0 = pickle.load(f)

    with open(pickle_path_1, "r") as f:
        dict_obj_1 = pickle.load(f)

    with open(pickle_path_2, "r") as f:
        dict_obj_2 = pickle.load(f)

    symbol_assignment_dict_0 = dict_obj_0['symbol_assignment']
    sentence_id_list_0 = dict_obj_0['sentence_id_list']

    symbol_assignment_dict_1 = dict_obj_1['symbol_assignment']
    sentence_id_list_1 = dict_obj_1['sentence_id_list']

    symbol_assignment_dict_2 = dict_obj_2['symbol_assignment']
    sentence_id_list_2 = dict_obj_2['sentence_id_list']

    dict_mapping_directories_to_possible_translations = {}


    for trajectory_of_interest in range(doc_range.shape[0] - 1):
        if(trajectory_of_interest in trajectories_to_avoid):
            continue
        if ('firealarm' in file_names_list[trajectory_of_interest]):
            # print(file_names_list[trajectory_of_interest])
            continue
        directory_name = file_names_list[trajectory_of_interest].split('/')[-2]
        print(directory_name)
        if directory_name not in dict_mapping_directories_to_possible_translations:
            dict_mapping_directories_to_possible_translations[directory_name] = []
        skills_present = list_of_skills[trajectory_of_interest]
        list_of_symbols = []
        for count, skill in enumerate(skills_present):
            next_elem = None
            if(count==skills_present.size-1):
                next_elem = "_"
            else:
                next_elem = skills_present[count+1]
            id_in_dict = str(trajectory_of_interest) + "-" + str(skill) + "-" + str(next_elem)
            if skill == 0:
                get_list = symbol_assignment_dict_0[id_in_dict]
                list_of_symbols.append(get_list)
            elif skill == 1:
                get_list = np.array(symbol_assignment_dict_1[id_in_dict]) + 1000
                list_of_symbols.append(get_list.tolist())
            elif skill == 2:
                get_list = np.array(symbol_assignment_dict_2[id_in_dict]) + 2000
                list_of_symbols.append(get_list.tolist())
            else:
                print("crap! weird ID WTH: ", id_in_dict)
            # list_of_trajectories_for_states.append(file_names_list[trajectory_of_interest])
        if len(skills_present) == 1:
            dict_mapping_directories_to_possible_translations[directory_name].extend(list_of_symbols[0])
            print(list_of_symbols[0])
        elif len(skills_present) == 2:
            list0 = list_of_symbols[0]
            list1 = list_of_symbols[1]
            cross_list = []
            for start_elem in list0:
                for end_elem in list1:
                    cross_list.append(str(start_elem) + " " + str(end_elem))
            dict_mapping_directories_to_possible_translations[directory_name].extend(cross_list)
            print(cross_list)
        elif len(skills_present) == 3:
            list0 = list_of_symbols[0]
            list1 = list_of_symbols[1]
            list2 = list_of_symbols[2]
            cross_list = []
            for start_elem in list0:
                for mid_elem in list1:
                    for end_elem in list2:
                        cross_list.append(str(start_elem) + " " + str(mid_elem) + " " + str(end_elem))
            dict_mapping_directories_to_possible_translations[directory_name].extend(cross_list)
            print(cross_list)


        lines = []
        for key in dict_mapping_directories_to_possible_translations:
            for sentence in dict_mapping_directories_to_possible_translations[key]:
                lines.append(key + ',' + str(sentence))


        with open ('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/targets_concentrated_without_firealarm.csv', 'w') as writeFile:
            writer = csv.writer(writeFile, delimiter=',')
            for line in lines:
                writer.writerow([line])
                print(line)
        writeFile.close()




# for the 0 cluster we chose eps = 88 and support of 2
# for all we are going to put in 88 with a support of 10
# skill 1 perfect is 130, 2
# skill 2 perfect is 150, 2





