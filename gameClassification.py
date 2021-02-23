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
X,y = read_and_preprocess(kFolds, True)


X.drop(labels="playerID", axis=1, inplace=True)

### Models ###
# SVM
#model = svm.SVC()
# rbf_feature = RBFSampler(gamma=0.6, random_state=1).fit_transform(X_train)

# SGD
#model = SGDClassifier()
sgd = SGDClassifier()

# MLPC multi layer perceptron
model = MLPClassifier(solver='adam', alpha=1e-5, random_state=0, max_iter = 400)
#mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(10,10,10), alpha=1e-5, random_state=0)

# KNN classifier
# model = KNeighborsClassifier()
#knn7 = KNeighborsClassifier(n_neighbors=7)

# Naibe bayes
#model = GaussianNB()

 # Gaussian process classifier
#kernel = i * RBF(i)
#model = GaussianProcessClassifier(random_state=0)

# Decision Tree
#model = tree.DecisionTreeClassifier(random_state=0)

# Random Forest
#forest = RandomForestClassifier(n_estimators=75, max_features = 6, random_state=0, n_jobs=1)

# Voting
#model = VotingClassifier(estimators = [('mlp',mlp),('knn7',knn7),('forest',forest),('sgd',sgd)], voting='hard')

### Grid Search ###
start = time.perf_counter()
X = preprocessing.StandardScaler().fit_transform(X)
#parameters = {'C':[1],'kernel':['poly'], 'gamma':[1]}
#parameters = {'C':[0.5,1,1.5],'kernel':['rbf'], 'gamma':[0.5,1,1.5]}
#parameters = {'n_neighbors':[7]}
#parameters = {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']}
#parameters = {'loss':['hinge', 'log']}
#parameters = {'kernel':[1.0 * RBF(1.0)]}
#parameters = {'hidden_layer_sizes':[(512,512,512),(100,100,100,100),(512,256,256),(100,80,70,60,50,40)]}
print(model)
parameters = {}
gridSearch = GridSearchCV(model, parameters, cv=kFolds, n_jobs=4).fit(X, y)
results = pd.DataFrame(gridSearch.cv_results_)
results = results.drop(labels=["std_fit_time","std_score_time","params"],axis=1)
print(results)

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start}")

debug = True
