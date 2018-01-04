#%%
import matplotlib.pyplot as plt

from sample_numbers import sample_number_1

sample_number = sample_number_1

# create a new figure 3in by 3in
plt.figure(1, figsize=(3, 3))

# display the image using a grayscale image map
plt.imshow(sample_number, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()