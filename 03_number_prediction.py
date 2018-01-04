#%%
from sklearn import datasets, svm
import matplotlib.pyplot as plt

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

# create a new figure 3in by 3in
plt.figure(1, figsize=(3, 3))

# display the image using a grayscale image map
plt.imshow(sample_number, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()

# predict using the last image
clf.predict([[item for sublist in sample_number for item in sublist]])