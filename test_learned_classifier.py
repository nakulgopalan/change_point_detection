import numpy as np
import pickle
from sklearn import svm
import sys

# from sklearn.ensemble import IsolationForest


#####
#From what we have saved skill 0 has two labels 0 and -1. -1 is end of corridor and 0 is corridor itself

pickle_path_0 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-0-symbol_data_without_firealarm_88.pickle'

# pickle_path_1 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-1-symbol_data_without_firealarm_88.pickle'

# pickle_path_2 = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/skill-2-symbol_data_without_firealarm_88.pickle'


# test_george_path = '/media/ng/LaCie SSD/map_data/george/straight1/out.bag.npy'

# test_george_path = '/media/ng/LaCie SSD/map_data/george/intersection_mid/out.bag.npy'

# test_george_path = '/media/ng/LaCie SSD/map_data/george/right_straight/out.bag.npy'

# test_george_path = '/media/ng/LaCie SSD/map_data/george/left_straight/out.bag.npy'

test_george_path = '/media/ng/LaCie SSD/map_data/george/intersection_right/out.bag.npy'

pickle_path_for_test_data =  '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/movo_complete_data.pickle'

with open(pickle_path_for_test_data, "r") as f:
    dict_obj_to_save = pickle.load(f)
    list_of_empty_arrays = dict_obj_to_save['0']
    list_of_action_indices = dict_obj_to_save['1']
    file_names_list = dict_obj_to_save['2']
    list_of_full_data = dict_obj_to_save['3']
    x = dict_obj_to_save['4']
    x_prev = dict_obj_to_save['5']
    z = dict_obj_to_save['6']
    doc_range = dict_obj_to_save['7']

trajectory_needed = 17
print(file_names_list[trajectory_needed])
all_the_states = list_of_full_data[trajectory_needed][:,412:682]
zero_arrays = np.where(~all_the_states.any(axis=1))[0]
all_the_states = np.delete(all_the_states, zero_arrays, 0)

with open(pickle_path_0, "r") as f:
    dict_obj_0 = pickle.load(f)



labels = dict_obj_0['clusters']
states = dict_obj_0['states']
sentence_ids = dict_obj_0['sentence_id_list']

states_to_test_on = np.load(test_george_path)[:,412:682]
zero_arrays = np.where(~states_to_test_on.any(axis=1))[0]
states_to_test_on = np.delete(states_to_test_on, zero_arrays, 0)
print(states_to_test_on.shape)

#0 is corridors
#-1 is corners

# for right 0 is double door and -1 is narrow things

# for left 0 is mapping to kitchen and -1 is mapping to when it leads to an open space

states_with_label_negative_1 = np.where(labels == -1)[0]
states_with_label_0 = np.where(labels == 0)[0]
for i in states_with_label_0:
    print(sentence_ids[int(i)], labels[int(i)])

print("--------------------------------------------------------")

for i in states_with_label_negative_1:
    print(sentence_ids[int(i)], labels[int(i)])



# For right and left both symbols should be equal but because of open spaces are not equal
# print(labels)

# sys.exit()

train_data = states[states_with_label_negative_1, :]
test_data = states[states_with_label_0, :]
extra_test_data = states_to_test_on

# full_data = states[states_with_label_corridor,:]
# test_data = states[states_with_label_end_of_corridor,:]

print(train_data.shape)
# fit the model
scale_value = 1./(train_data.shape[1] * train_data.var())

## One class SVM models
# clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.000486157539339571)
clf = svm.OneClassSVM(nu=0.1,kernel="sigmoid", gamma='scale') # sigmoid performed better
# gamma values: 0 0 0.0008028497098985085
# clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma='scale')  # linear is second best by a large margin
# clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma='scale', degree=5)  # poly is crap


## Isolation forest
# clf = IsolationForest(max_samples=50, bootstrap=True, contamination=0.15, max_features=40, n_estimators=300)


clf.fit(train_data)
y_pred_train = clf.fit_predict(train_data)
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

print('++++++++++++++prediction for the entire state array of clocks')
y_pred_test_larger = clf.predict(all_the_states)
print(y_pred_test_larger)


sys.exit()
import pickle
# file_name = 'skill_2_sigmoid_classifier_corridor_-1.pkl'
file_name = 'crap.pkl'
classifier_model_pkl = open(file_name, 'wb')
pickle.dump(clf, classifier_model_pkl)
classifier_model_pkl.close()
# print(clf.get_params())


# for count, id in enumerate(states_with_label_end_of_corridor):
#     print(sentence_ids[id], y_pred_train[count])

# for count, id in enumerate(states_with_label_end_of_corridor):
#     print(sentence_ids[id], y_pred_test[count])