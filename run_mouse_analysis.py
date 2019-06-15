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
import bnpy.data.XData as XData
from scipy.signal import savgol_filter


from matplotlib import pylab
import seaborn as sns

from bnpy.data import GroupXData
import matplotlib

FIG_SIZE = (10, 5)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
#
# Setup: Load data
# ----------------

# Read bnpy's built-in "Mocap6" dataset from file.

# dataset_path = os.path.join(bnpy.DATASET_PATH, 'mocap6')
# dataset = bnpy.data.GroupXData.read_npz(
#     os.path.join(dataset_path, 'dataset.npz'))
#
# print(dataset.dim)

# path = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/trajectory_'
path = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/test_'
files = range(0,2)
doc_range = [0]
x = None
x_prev = None
for i in files:
    string_path = path+str(i)+'.npy'
    data = np.load(string_path)
    data_prev = np.vstack([data[0,:],data[0:-1,:]])
    doc_range.append(doc_range[-1] + data.shape[0])
    if x is not None:
        x = np.vstack((x,data))
        x_prev = np.vstack((x_prev,data_prev))
    else:
        x = data
        x_prev = data_prev





# data0 = np.load('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/trajectory_observation_model.npy')
#
# # data0 = np.genfromtxt('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/PyUserInput/examples/data0.csv', delimiter=',')
# # data1 = np.genfromtxt('/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/PyUserInput/examples/data1.csv',delimiter=',')
# data_0_prev = np.vstack([data0[0,:],data0[0:-1,:]])
# # data_1_prev = np.vstack([data1[0,:],data1[0:-1,:]])
#
# # data_0_smooth = savgol_filter(data0, 51, 3, axis=0)
# # data_1_smooth = savgol_filter(data1, 51, 3, axis=0)
# # data0_diff = np.diff(data_0_smooth,axis=0)
# # data1_diff = np.diff(data_1_smooth,axis=0)
# # x = np.vstack((data0, data1))
# # x_prev = np.vstack((data_0_prev,data_1_prev))
# x = data0
# x_prev = data_0_prev
# # print(x.shape)
# # print (data0.shape)
#
# doc_range = [0,data0.shape[0]]#,data0.shape[0]+data1.shape[0]]
dataset = GroupXData(X=x,doc_range=doc_range, Xprev=x_prev)

output_path_starter = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/outputs/'


####################################
#Show single sequence
def show_single_sequence(
        seq_id,
        zhat_T=None,
        z_img_cmap=None,
        ylim=[0, 2500],
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
    for dim in xrange(2):#fix range here!
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

###############################################################################
#
# Setup: Initialization hyperparameters
# -------------------------------------

init_kwargs = dict(
    K=30,
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
    gamma = 10.0,       # top-level Dirichlet concentration parameter
    transAlpha = 0.5,  # trans-level Dirichlet concentration parameter
)

###############################################################################
#
# Setup: Gaussian observation model hyperparameters
# -------------------------------------------------

# gauss_kwargs = dict(
#     sF = 1.0,          # Set prior so E[covariance] = identity
#     ECovMat = 'eye',
#     MMat ='eye'
#     )


gauss_kwargs = dict(
    sF = 1.0,          # Set prior so E[covariance] = identity
    ECovMat = 'eye',
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



goodelbopairs_trained_model, goodelbopairs_info_dict = bnpy.run(
    dataset, 'HDPHMM', 'AutoRegGauss', 'memoVB', #
    output_path=output_path_starter+'trymerge-K=20-model=HDPHMM+ARMA-ECovMat=1*eye-merge_strategy=good_elbo_pairs/',
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

for pq in files:
    log_lik_seq0_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
    dataset.make_subset([pq])
    )
    zhat_seq0_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
    log_lik_seq0_TK, np.log(start_prob_K), np.log(trans_prob_KK))
    show_single_sequence(pq, zhat_T=zhat_seq0_T, K=K)
    pylab.show()

# log_lik_seq1_TK = goodelbopairs_trained_model.obsModel.calcLogSoftEvMatrix_FromPost(
#     dataset.make_subset([1])
#     )
#
# zhat_seq1_T = bnpy.allocmodel.hmm.HMMUtil.runViterbiAlg(
#     log_lik_seq1_TK, np.log(start_prob_K), np.log(trans_prob_KK))
#
# show_single_sequence(1, zhat_T=zhat_seq0_T, K=K)
# pylab.show()




#
# ###############################################################################
# #
# # Compare loss function vs wallclock time
# # ---------------------------------------
# #
# pylab.figure()
# for info_dict, color_str, label_str in [
#         (allpairs_info_dict, 'k', 'all_pairs'),
#         (largepairs_info_dict, 'g', 'large_pairs'),
#         (goodelbopairs_info_dict, 'b', 'good_elbo_pairs')]:
#     pylab.plot(
#         info_dict['elapsed_time_sec_history'],
#         info_dict['loss_history'],
#         '.-',
#         color=color_str,
#         label=label_str)
# pylab.legend(loc='upper right')
# pylab.xlabel('elapsed time (sec)')
# pylab.ylabel('loss')
#
#
# ###############################################################################
# #
# # Compare number of active clusters vs wallclock time
# # ---------------------------------------------------
# #
# pylab.figure()
# for info_dict, color_str, label_str in [
#         (allpairs_info_dict, 'k', 'all_pairs'),
#         (largepairs_info_dict, 'g', 'large_pairs'),
#         (goodelbopairs_info_dict, 'b', 'good_elbo_pairs')]:
#     pylab.plot(
#         info_dict['elapsed_time_sec_history'],
#         info_dict['K_history'],
#         '.-',
#         color=color_str,
#         label=label_str)
# pylab.legend(loc='upper right')
# pylab.xlabel('elapsed time (sec)')
# pylab.ylabel('num. clusters (K)')
#
# pylab.show()