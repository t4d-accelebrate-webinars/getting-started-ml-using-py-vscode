#%%
from sklearn import datasets, svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# load digits data from scikit learn
digits = datasets.load_digits()

learning_algo = svm.SVC(kernel='linear', random_state=101)

search_space = [{'kernel': ['linear'], 
                 'C': np.logspace(-3, 3, 7)},
                {'kernel': ['rbf'], 
                 'C':np.logspace(-3, 3, 7), 
                 'gamma': np.logspace(-3, 2, 6)}]

gridsearch = GridSearchCV(learning_algo, 
                          param_grid=search_space, 
                          refit=True, cv=10)

gridsearch.fit(digits.data[:-1], digits.target[:-1])
print ('Best parameter: %s' % str(gridsearch.best_params_))

cv_performance = gridsearch.best_score_
test_performance = gridsearch.score(digits.data[-1:], digits.target[-1:])
print ('Cross-validation accuracy score: %0.3f, test accuracy score: %0.3f' %
       (cv_performance,test_performance))
