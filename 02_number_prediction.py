#%%
from sklearn import datasets, svm


# load digits data from scikit learn
digits = datasets.load_digits()

# configure support vector classification
clf = svm.SVC(gamma=0.001, C=100.)

# train with all data except the last image
clf.fit(digits.data, digits.target)

sample_number = [
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
  [ 0.,  0.,  0.,  0.,  16.,  0.,  0.,  0. ],
]

# predict using the last image
clf.predict([[item for sublist in sample_number for item in sublist]])