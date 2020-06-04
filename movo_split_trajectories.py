"""
========================
Merge moves with HDP-HMM
========================

How to try merge moves efficiently for time-series datasets.

This example reviews three possible ways to plan and execute merge
proposals.

* try merging all pairs of clusters
* pick fewer merge pairs (at most 5 per cluster) in a size-biased way
* pick fewer merge pairs (at most 5 per cluster) in objective-driven way

"""
# sphinx_gallery_thumbnail_number = 2

import bnpy
import numpy as np
import os
import glob
import bnpy.data.XData as XData
from scipy.signal import savgol_filter

from helper_functions import moving_average

from matplotlib import pylab
import seaborn as sns

from bnpy.data import GroupXData
import matplotlib

FIG_SIZE = (10, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE
size_of_lidar_data = 1091

###############################################################################
#
# Setup: Load data
# ----------------


# Read bnpy's built-in "Mocap6" dataset from file.





def read_data(path, doc_range=0, z_value=0):
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
        file_names_list.append(path+i)

        data = np.load(path + i)#.transpose()

        # raw_data = np.load(path + i[:-3]+'full.npy').transpose()
        print(data.shape)
        # print(raw_data.shape)
        list_of_complete_data.append(data)
        # shape is samples x dim ( = 13) now

        # data cleaning:  removed all nan's, for actions the nans are zeros as there is no velocity
        # and for lidar data it is the previously noted lidar measurement!

        #check places where nan in the first elememnts and zeros in the last 6

        deleted_elem_array = np.array(range(data.shape[0]))


        elements_without_action_data = np.where(~data[:,0:2].any(axis=1))[0]
        # last_six_zeros = np.where(~data[:,-6:].any(axis=1))[0]
        #
        # common_elements = np.intersect1d(last_six_zeros, nan_array)

        data = np.delete(data, elements_without_action_data, 0)
        deleted_elem_array = np.delete(deleted_elem_array,elements_without_action_data,0)

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

        elements_greater_than_0 = np.argwhere(data[:,1]>0.)
        elements_lesser_than_0 = np.argwhere(data[:,1]<0.)
        action_data = data[:,0:3]
        action_data[:,2] = action_data[:,1] *1.
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



        action_data = np.cumsum(action_data,0)

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

    return (x,x_prev,z, np.array(doc_range), list_of_action_indices,list_of_empty_arrays, file_names_list, list_of_complete_data)






####### read elements
def read_data_old(path, doc_range=0, z_value=0):
    os.chdir(path)
    doc_range = [doc_range]
    extension = 'npy'
    eoc_files = glob.glob('*.{}'.format(extension))

    x = None
    x_prev = None
    z = None
    for i in eoc_files:
        # i = i+3
        # string_path = path+str(i)+'.npy'
        data = np.load(path + i).transpose()
        # shape is samples x dim ( = 13) now

        # data cleaning:  removed all nan's, for actions the nans are zeros as there is no velocity
        # and for lidar data it is the previously noted lidar measurement!

        #check places where nan in the first elememnts and zeros in the last 6

        nan_array = np.argwhere(np.isnan(data[:, 0]))
        last_six_zeros = np.where(~data[:,-6:].any(axis=1))[0]

        common_elements = np.intersect1d(last_six_zeros, nan_array)

        data = np.delete(data, common_elements, 0)

        nan_array = np.argwhere(np.isnan(data[:, 0]))

        if 0 in nan_array:
            nan_array = np.delete(nan_array, 0, 0)
            data = np.delete(data, 0, 0)
            nan_array = nan_array - 1



        # check if entire row is zeros delete them
        # empty_array = np.where(~data.any(axis=1))[0]

        data = np.nan_to_num(data)
        data[nan_array, 0:-6] = data[nan_array - 1, 0:-6]

        while True:
            lidar_state_zero = np.where(~data[:, :-6].any(axis=1))[0]
            if(lidar_state_zero.size == 0):
                break
            data[lidar_state_zero, 0:-6] = data[lidar_state_zero - 1, 0:-6]



        matplotlib.pyplot.plot(data[:,-10]) # 3 is for 0 deg -4 was interesting and -3

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
    matplotlib.pyplot.show()

    return (x,x_prev,z, doc_range)


####################################
#Show single sequence
def show_single_sequence(
        seq_id,
        zhat_T=None,
        z_img_cmap=None,
        ylim=[-10, 10],
        K=5,
        left=0.2, bottom=0.2, right=0.8, top=0.95):
    if z_img_cmap is None:
        z_img_cmap = matplotlib.cm.get_cmap('Set1', K)

    if zhat_T is None:
        nrows = 1
    else:
        nrows = 2
    fig_h, ax_handles = pylab.subplots(
        nrows=nrows, ncols=1, sharex=True, sharey=False)
    ax_handles = np.atleast_1d(ax_handles).flatten().tolist()

    start = dataset.doc_range[seq_id]
    stop = dataset.doc_range[seq_id + 1]
    # Extract current sequence
    # as a 2D array : T x D (n_timesteps x n_dims)
    curX_TD = dataset.X[start:stop]
    for dim in xrange(3):#fix range here!
        ax_handles[0].plot(curX_TD[:, dim], '.-')
    ax_handles[0].set_ylabel('x-axis')
    ax_handles[0].set_ylim(ylim)
    z_img_height = int(np.ceil(ylim[1] - ylim[0]))
    pylab.subplots_adjust(
        wspace=0.1,
        hspace=0.1,
        left=left, right=right,
        bottom=bottom, top=top)
    if zhat_T is not None:
        img_TD = np.tile(zhat_T, (z_img_height, 1))
        ax_handles[1].imshow(
            img_TD,
            interpolation='nearest',
            vmin=-0.5, vmax=(K-1)+0.5,
            cmap=z_img_cmap)
        ax_handles[1].set_ylim(0, z_img_height)
        ax_handles[1].set_yticks([])

        bbox = ax_handles[1].get_position()
        width = (1.0 - bbox.x1) / 3
        height = bbox.y1 - bbox.y0
        cax = fig_h.add_axes([right + 0.01, bottom, width, height])
        cbax_h = fig_h.colorbar(
            ax_handles[1].images[0], cax=cax, orientation='vertical')
        cbax_h.set_ticks(np.arange(K))
        cbax_h.set_ticklabels(np.arange(K))
        cbax_h.ax.tick_params(labelsize=9)

    ax_handles[-1].set_xlabel('time')
    return ax_handles




#########
#
# Read data
#--------------
# list_of_paths = []

# path_right = '/home/ng/workspace/corl_2019_all_code/data/final_files/left/'
# path_left = '/home/ng/workspace/corl_2019_all_code/data/final_files/right/'
path_straight = '/home/ng/workspace/corl_2019_all_code/data/final_files/full_behavior/'
# path_3 = '/media/ng/LaCie SSD/all_cleaned_bags/library_right/'
# path_4 = '/media/ng/LaCie SSD/all_cleaned_bags/427_left/'
# path_5 = '/media/ng/LaCie SSD/all_cleaned_bags/clock_double_door/'
# path_6 = '/media/ng/LaCie SSD/all_cleaned_bags/firealarm_double_door/'
# path_7 = '/media/ng/LaCie SSD/all_cleaned_bags/kitchen_left/'
# path_8 = '/media/ng/LaCie SSD/all_cleaned_bags/kitchen_right/'
# path_9 = '/media/ng/LaCie SSD/all_cleaned_bags/straight_clock/'
# path_10 = '/media/ng/LaCie SSD/all_cleaned_bags/straight_firealarm/'
# path_11 = '/media/ng/LaCie SSD/all_cleaned_bags/t_intersection/'
# path_12 = '/media/ng/LaCie SSD/all_cleaned_bags/watercooler_tintersection_left/'
# path_13 = '/media/ng/LaCie SSD/all_cleaned_bags/watercooler_tintersection_right/'



list_of_paths = [path_straight]#[path_right,path_left,path_straight]#, path_3,\
                 # path_4, path_5, path_6, path_7, path_8, path_9, path_10, \
                 # path_11, path_12, path_13]
doc_range_const = 0
z_const = 0

list_of_empty_arrays = []
list_of_action_indices = []
file_names_list = []

for path in list_of_paths:
    x_temp,x_prev_temp, z_temp, doc_range_temp, list_of_action_indices_temp ,list_of_empty_arrays_temp , file_names_list_temp, complete_data_list  = read_data(path,doc_range=doc_range_const,z_value=z_const)
    list_of_empty_arrays.extend(list_of_empty_arrays_temp)
    list_of_action_indices.extend(list_of_action_indices_temp)
    file_names_list.extend(file_names_list_temp)
    if z_const==0:
        x, x_prev, z, doc_range = x_temp, x_prev_temp, z_temp, doc_range_temp
    else:
        x = np.vstack((x, x_temp))
        x_prev = np.vstack((x_prev, x_prev_temp))
        z = np.hstack((z, z_temp))
        doc_range = np.hstack((doc_range[:-1], doc_range_temp))
    doc_range_const = doc_range_temp[-1]
    z_const=z_const+10

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

print("total trajectories: ", doc_range.shape[0]-1)



dataset = GroupXData(X=x,doc_range=doc_range, Xprev=x_prev) #, TrueZ=z

# output_path_starter = '/media/ng/Vetnari/nakul_old_thinkpad/corl_all_data/all_cleaned_bags/new_outputs/'
output_path_starter = '/home/ng/workspace/corl_2019_all_code/output/'


###############################################################################
#
# Setup: Initialization hyperparameters
# -------------------------------------

init_kwargs = dict(
    K=20,
    initname='randexamples',
    )

alg_kwargs = dict(
    nLap=100,
    nTask=5, nBatch=1, convergeThr=0.0001,
    )

###############################################################################
#
# Setup: HDP-HMM hyperparameters
# ------------------------------

hdphmm_kwargs = dict(
    gamma = 10.,       # top-level Dirichlet concentration parameter 10.
    transAlpha = 0.5,  # trans-level Dirichlet concentration parameter 0.5
)

###############################################################################
#
# Setup: Gaussian observation model hyperparameters
# -------------------------------------------------

gauss_kwargs = dict(
    sF = 1.0,          # Set prior so E[covariance] = identity
    ECovMat = 'eye', #'fromtruelabels',  #'eye',
    MMat ='eye'
    )


###############################################################################
#
# All-Pairs : Try all possible pairs of merges every 10 laps
# ----------------------------------------------------------
#
# This is expensive, but a good exhaustive test.

# allpairs_merge_kwargs = dict(
#     m_startLap = 10,
#     # Set limits to number of merges attempted each lap.
#     # This value specifies max number of tries for each cluster
#     # Setting this very high (to 50) effectively means try all pairs
#     m_maxNumPairsContainingComp = 50,
#     # Set "reactivation" limits
#     # So that each cluster is eligible again after 10 passes thru dataset
#     # Or when it's size changes by 400%
#     m_nLapToReactivate = 10,
#     m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
#     # Specify how to rank pairs (determines order in which merges are tried)
#     # 'total_size' and 'descending' means try largest combined clusters first
#     m_pair_ranking_procedure = 'total_size',
#     m_pair_ranking_direction = 'descending',
#     )
#
# allpairs_trained_model, allpairs_info_dict = bnpy.run(
#     dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
#     output_path=output_path_starter+'trymerge-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye-merge_strategy=all_pairs/',
#     moves='merge,shuffle',
#     **dict(
#         alg_kwargs.items()
#         + init_kwargs.items()
#         + hdphmm_kwargs.items()
#         + gauss_kwargs.items()
#         + allpairs_merge_kwargs.items()))

###############################################################################
#
# Large-Pairs : Try 5-largest-size pairs of merges every 10 laps
# --------------------------------------------------------------
#
# This is much cheaper than all pairs. Let's see how well it does.

# largepairs_merge_kwargs = dict(
#     m_startLap = 10,
#     # Set limits to number of merges attempted each lap.
#     # This value specifies max number of tries for each cluster
#     m_maxNumPairsContainingComp = 5,
#     # Set "reactivation" limits
#     # So that each cluster is eligible again after 10 passes thru dataset
#     # Or when it's size changes by 400%
#     m_nLapToReactivate = 10,
#     m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
#     # Specify how to rank pairs (determines order in which merges are tried)
#     # 'total_size' and 'descending' means try largest size clusters first
#     m_pair_ranking_procedure = 'total_size',
#     m_pair_ranking_direction = 'descending',
#     )


# largepairs_trained_model, largepairs_info_dict = bnpy.run(
#     dataset, 'HDPHMM', 'DiagGauss', 'memoVB',
#     output_path=output_path_starter+'trymerge-K=20-model=HDPHMM+DiagGauss-ECovMat=1*eye-merge_strategy=large_pairs/',
#     moves='merge,shuffle',
#     **dict(
#         alg_kwargs.items()
#         + init_kwargs.items()
#         + hdphmm_kwargs.items()
#         + gauss_kwargs.items()
#         + largepairs_merge_kwargs.items()))

###############################################################################
#
# Good-ELBO-Pairs : Rank pairs of merges by improvement to observation model
# --------------------------------------------------------------------------
#
# This is much cheaper than all pairs and perhaps more principled.
# Let's see how well it does.

goodelbopairs_merge_kwargs = dict(
    m_startLap = 10,
    # Set limits to number of merges attempted each lap.
    # This value specifies max number of tries for each cluster
    m_maxNumPairsContainingComp = 5,
    # Set "reactivation" limits
    # So that each cluster is eligible again after 10 passes thru dataset
    # Or when it's size changes by 400%
    m_nLapToReactivate = 10,
    m_minPercChangeInNumAtomsToReactivate = 400 * 0.01,
    # Specify how to rank pairs (determines order in which merges are tried)
    # 'obsmodel_elbo' means rank pairs by improvement to observation model ELBO
    m_pair_ranking_procedure = 'obsmodel_elbo',
    m_pair_ranking_direction = 'descending',
    )


#trymerge-K=20-model=HDPHMM+ARMA-ECovMat=1*eye-merge_strategy=good_elbo_pairs/
goodelbopairs_trained_model, goodelbopairs_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'AutoRegGauss', 'memoVB', #
    output_path=output_path_starter+'test/',
    moves='merge,shuffle',
    **dict(
        alg_kwargs.items()
        + init_kwargs.items()
        + hdphmm_kwargs.items()
        + gauss_kwargs.items()
        + goodelbopairs_merge_kwargs.items()))
K=goodelbopairs_trained_model.obsModel.K
start_prob_K = goodelbopairs_trained_model.allocModel.get_init_prob_vector()
trans_prob_KK = goodelbopairs_trained_model.allocModel.get_trans_prob_matrix()
prior = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost
post = goodelbopairs_trained_model.obsModel.Post


print("printing all data!")
print(goodelbopairs_trained_model.obsModel.Post.M)
print(goodelbopairs_trained_model.obsModel.Post.B)
print(goodelbopairs_trained_model.obsModel.Post)
print("printing finished!")

# log_lik_seq0_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
#     dataset.make_subset([0])
#     )
#
# zhat_seq0_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
#     log_lik_seq0_TK, np.log(start_prob_K), np.log(trans_prob_KK))
#
# show_single_sequence(0, zhat_T=zhat_seq0_T, K=K)
# pylab.show()
#
# log_lik_seq1_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
#     dataset.make_subset([1])
#     )
#
# zhat_seq1_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
#     log_lik_seq1_TK, np.log(start_prob_K), np.log(trans_prob_KK))
#
# show_single_sequence(1, zhat_T=zhat_seq0_T, K=K)
# pylab.show()


z_hat_list = []

for pq in range(doc_range.shape[0]-1):
    print(file_names_list[pq])
    log_lik_seq0_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
    dataset.make_subset([pq])
    )
    zhat_seq0_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
    log_lik_seq0_TK, np.log(start_prob_K), np.log(trans_prob_KK))
    z_hat_list.append(zhat_seq0_T)
    show_single_sequence(pq, zhat_T=zhat_seq0_T, K=K)
    pylab.show()



np.savez('segment_run_movo',dataset = dataset, z_hat_list = z_hat_list, file_names_list =file_names_list, list_of_action_indices = list_of_action_indices, list_of_empty_arrays =list_of_empty_arrays)


# paths = [
#         # '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_right/',
#         #  '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/take_left/',
#         #  '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/go_to_intersection/',
#          '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_left/',
#          '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/intersection_and_right/',
#          '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/turtlebot/data_collection/go_to_the_end_of_corridor/'
#          ]
#
#
# for path in paths:
#     print(path)
#     x_new, x_prev_new, z_new, doc_range_new = read_data(path, doc_range=0)
#
#     new_dataset = GroupXData(X=x_new[:, -5:-2], doc_range=doc_range_new, Xprev=x_prev_new[:, -5:-2], TrueZ=z_new)
#     for pq in range(len(doc_range_new)-1):
#         log_lik_seq0_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
#         new_dataset.make_subset([pq])
#         )
#         zhat_seq0_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
#         log_lik_seq0_TK, np.log(start_prob_K), np.log(trans_prob_KK))
#         show_single_sequence(pq, zhat_T=zhat_seq0_T, K=K)
#         pylab.show()


