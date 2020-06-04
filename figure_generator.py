import numpy as np
import pickle
from sklearn import svm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


from sklearn.ensemble import IsolationForest




pickle_path_0 = '/home/ng/workspace/corl_2019_all_code/all_code_backup/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-0-symbol_data_without_firealarm_88.pickle'

# pickle_path_0 = '/home/ng/workspace/corl_2019_all_code/data/symb_skill0.pkl'
# pickle_path_0 = '/home/ng/workspace/corl_2019_all_code/data/symb_skill2.pkl'


with open(pickle_path_0, "r") as f:
    dict_obj_0 = pickle.load(f)



labels = dict_obj_0['clusters']
states = dict_obj_0['states']
sentence_ids = dict_obj_0['sentence_id_list']


# test_lidar_data = np.load('/home/ng/workspace/corl_2019_all_code/data/map_points/final_forward3.bag.npy')
test_lidar_data = np.load('/home/ng/workspace/corl_2019_all_code/data/map_points/final_left1.bag.npy')
# lidar_data = x[:,2:-1]
lidar_data = test_lidar_data[:,2:]
lidar_data_cleaned = lidar_data[~np.all(lidar_data == 0, axis=1)]
test_points = lidar_data_cleaned[:,410:680]

X_fit = PCA().fit(states)
X_pca = X_fit.transform(states)
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=200)
plt.colorbar(ticks=range(10))
plt.show()

#0 is corridors
#-1 is corners

states_with_label_corridor = np.where(labels==0)[0]
states_with_label_end_of_corridor = np.where(labels == -1)[0]
if(False):
    states_with_label_end_of_corridor = states[states_with_label_end_of_corridor]
    states_with_label_corridor = test_points
    clf = svm.OneClassSVM(nu=0.1,kernel="sigmoid", gamma='scale')
    clf.fit(states_with_label_end_of_corridor)
    y_pred_train = clf.fit_predict(states_with_label_end_of_corridor)
    # for i in range(test_points.shape[0]):
    #     y_pred_test = clf.fit_predict(test_points[i,:].reshape(1,-1))
    #     print(y_pred_test)
    y_pred_test = clf.fit_predict(test_points)
    print(y_pred_test)
    print(y_pred_train)
    print(np.sum(y_pred_train))
    print(np.sum(y_pred_test))


# for i in states_with_label_corridor:
#     print(sentence_ids[int(i)])
if(False):
    states_with_label_end_of_corridor = np.where(labels == -1)[0]
    states_with_label_corridor = np.where(labels == 0)[0]
    for _ in range(20):
        labels_removed_for_test =  np.random.choice(states_with_label_end_of_corridor.shape[0], 10, replace=False)

        mask = np.ones(states_with_label_end_of_corridor.shape[0], dtype=bool)
        mask[labels_removed_for_test] = False

        full_data = states[states_with_label_end_of_corridor[mask],:]
        test_data = states[states_with_label_corridor,:]
        extra_test_data = states[states_with_label_end_of_corridor[~mask],:]


        # full_data = states[states_with_label_corridor,:]
        # test_data = states[states_with_label_end_of_corridor,:]

        print(full_data.shape)
        # fit the model
        scale_value = 1./(full_data.shape[1]*full_data.var())

        ## One class SVM models
        # clf = svm.OneClassSVM(nu=0.05, kernel="rbf")
        clf = svm.OneClassSVM(nu=0.1,kernel="sigmoid", gamma='scale') # sigmoid performed better
        # clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma='scale')  # linear is second best by a large margin
        # clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma='scale', degree=5)  # poly is crap


        ## Isolation forest
        # clf = IsolationForest(max_samples=50, bootstrap=True, contamination=0.15, max_features=40, n_estimators=300)


        clf.fit(full_data)
        y_pred_train = clf.fit_predict(full_data)
        y_pred_test = clf.predict(test_data)
        extra_pred = clf.predict(extra_test_data)

        # print(full_data.var())


        print(float(y_pred_train[y_pred_train == 1].size)/(y_pred_train.size))
        print(float(y_pred_test[y_pred_test == 1].size)/(y_pred_test.size))
        print(float(extra_pred[extra_pred == 1].size)/(extra_pred.size))

        print("+++++++++++++++++++++++++++++++++++++++++++++++")

        print(y_pred_train)
        print(y_pred_test)
        print(extra_pred)
        print("-----------------------------------------------")

        # print(clf.get_params())


        # for count, id in enumerate(states_with_label_end_of_corridor):
        #     print(sentence_ids[id], y_pred_train[count])

        # for count, id in enumerate(states_with_label_corridor):
        #     print(sentence_ids[id], y_pred_test[count])