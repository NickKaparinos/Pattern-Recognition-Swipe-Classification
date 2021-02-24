# Project Recognission Grid Search 1
import time
import pandas as pd
import numpy as np
import sklearn.metrics as skm
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_selection import SelectKBest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# import matplotlib.pyplot as plt
from preprocess import read_and_preprocess

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # Read data
kFolds = 4
X,y = read_and_preprocess(True)


X.drop(labels="playerID", axis=1, inplace=True)

### Models ###
# KNN classifier
model = KNeighborsClassifier()

### Grid Search ###
start = time.perf_counter()
X = preprocessing.StandardScaler().fit_transform(X)

parameters = {'n_neighbors':[3,5,7,9]}
print(model)
gridSearch = GridSearchCV(model, parameters, cv=kFolds, n_jobs=kFolds).fit(X, y)
results = pd.DataFrame(gridSearch.cv_results_)
results = results.drop(labels=["std_fit_time","std_score_time","params"],axis=1)
print(results)

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start:.2f}")
