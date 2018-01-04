
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

sample_number = [
  [ 0.,  0.,  10.,  14.,   8.,   1.,  0.,  0. ],
  [ 0.,  2.,  16.,  14.,   6.,   1.,  0.,  0. ],
  [ 0.,  0.,  15.,  15.,   8.,  15.,  0.,  0. ],
  [ 0.,  0.,   5.,  16.,  16.,  10.,  0.,  0. ],
  [ 0.,  0.,  12.,  15.,  15.,  12.,  0.,  0. ],
  [ 0.,  4.,  16.,   6.,   4.,  16.,  6.,  0. ],
  [ 0.,  8.,  16.,  10.,   8.,  16.,  8.,  0. ],
  [ 0.,  1.,   8.,  12.,  14.,  12.,  1.,  0. ],
]

# create a new figure 3in by 3in
plt.figure(1, figsize=(3, 3))

# display the image using a grayscale image map
plt.imshow(sample_number, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()

