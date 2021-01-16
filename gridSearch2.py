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
# import matplotlib.pyplot as plt
from preprocess2 import read_and_preprocess2

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # Read data
kFolds = 4
X,y = read_and_preprocess2(kFolds)

# Feature selection kbest
# = SelectKBest(k=10).fit_transform(X, y)
# kbest = SelectKBest(k=7).fit(X, y)
# mask = kbest.get_support()
# features = list(X.columns)
# features = [f for m,f in zip(mask,features) if m]
#X = kbest.transform(X)

### Models ###
# SVM
model = svm.SVC()
# rbf_feature = RBFSampler(gamma=0.6, random_state=1).fit_transform(X_train)

# SGD
#model = SGDClassifier()

# MLPC multi layer perceptron
#model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=i, random_state=1)

# KNN classifier
#model = KNeighborsClassifier()

# Naibe bayes
#model = GaussianNB()

 # Gaussian process classifier
#kernel = i * RBF(i)
#model = GaussianProcessClassifier(random_state=0)

# Decision tree
#model = tree.DecisionTreeClassifier(random_state=42)

### Grid Search ###
start = time.perf_counter()
X = preprocessing.StandardScaler().fit_transform(X)
parameters = {'C':[1],'kernel':['poly'], 'gamma':[1]}
#parameters = {'C':[0.5,1,1.5],'kernel':['rbf'], 'gamma':[0.5,1,1.5]}
#parameters = {'n_neighbors':[3,5,7,9,11,13,15]}
#parameters = {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']}
#parameters = {'loss':['hinge', 'log']}
#parameters = {'kernel':[1.0 * RBF(1.0)]}
#parameters = {}
gridSearch = GridSearchCV(model, parameters, cv=kFolds, n_jobs=2).fit(X, y)
results = pd.DataFrame(gridSearch.cv_results_)
results = results.drop(labels=["std_fit_time","std_score_time","params"],axis=1)
print(results)

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start}")

debug = True
