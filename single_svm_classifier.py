import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm


path = '/media/ng/7ccf8f98-7ab8-498b-b405-54df784c3191/ng/workspace/bayesian_changepoint_detection/bnpy/examples/08_mocap6/test_svm_'
files = range(0,5)
train_data = None


for i in files:
    string_path = path+str(i)+'.npy'
    data = np.load(string_path)
    conservative_data_points = data[np.greater(data[:,1],450)]
    # data_prev = np.vstack([data[0,:],data[0:-1,:]])
    # doc_range.append(doc_range[-1] + data.shape[0])
    if train_data is not None:
        train_data = np.vstack((train_data,conservative_data_points))
        # x_prev = np.vstack((x_prev,data_prev))
    else:
        train_data = conservative_data_points
        # x_prev = data_prev


files = range(5,7)
test_data = None

for i in files:
    string_path = path+str(i)+'.npy'
    data = np.load(string_path)
    # conservative_data_points = data[np.logical_and(data[:,1]>=450)]
    # data_prev = np.vstack([data[0,:],data[0:-1,:]])
    # doc_range.append(doc_range[-1] + data.shape[0])
    if test_data is not None:
        test_data = np.vstack((test_data,data))
        # x_prev = np.vstack((x_prev,data_prev))
    else:
        test_data = data
        # x_prev = data_prev

# fit the model
clf = svm.OneClassSVM(nu=0.001, kernel="rbf", gamma=0.000001)
clf.fit(train_data)
y_pred_train = clf.predict(train_data)
y_pred_test = clf.predict(test_data)

col = np.where(y_pred_test<1,'r','b')


s = 40
b1 = plt.scatter(train_data[:, 0], train_data[:, 1], c='k', s=s, edgecolors='k')
b2 = plt.scatter(test_data[:, 0], test_data[:, 1], c=col, s=s,
                 edgecolors='k')
plt.show()