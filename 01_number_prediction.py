#%%
from sklearn import datasets, svm

# load digits data from scikit learn
digits = datasets.load_digits()

# configure support vector classification
# gamma is kernel coefficient
# C is the penalty parameter, default is 1,
# more noisy data should be a lower value, less noisy a higher value
clf = svm.SVC(gamma=0.001, C=100.)

# train with all data except the last image
clf.fit(digits.data[:-1], digits.target[:-1])

# predict using the last image
clf.predict(digits.data[-1:])