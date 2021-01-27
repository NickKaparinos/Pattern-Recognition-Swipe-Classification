# Project Recognission two level classification
import os
import time

#explicitly require this experimental feature
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
#now you can import normally from ensemble
#from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomTreesEmbedding

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import GridSearchCV
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import SGDClassifier
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.feature_selection import SelectKBest
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
# # import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import OneHotEncoder

# import matplotlib.pyplot as plt
from preprocess import read_and_preprocess
from tfModel import tfModel

## OPTIONS ###
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
useTF = False
usePCA = False
useICA = False
useIsomap = False
nEpochs = 30
nComponents = 10
kNeighbors = 15
k = 2
eps = 6
minSamples = 5
pca1 = PCA(n_components=nComponents)
pca2 = PCA(n_components=nComponents)
ica1 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
ica2 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
iso1 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
iso2 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
print(f"Using tensorflow : {useTF}")
print(f"Using PCA : {usePCA}")
print(f"Using ICA : {useICA}")
print(f"Using Isomap : {useIsomap}")
if usePCA or useICA or useIsomap:
    print(f"Number of components {nComponents}")

### Read data ###
kFolds = 4

X, y = read_and_preprocess(kFolds, True)

### First Classification ###
# Train test split
start = time.perf_counter()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, stratify=X['playerID'], test_size=0.25, random_state=42)
# yl = list(y)
# distributionGame = {i: yl.count(i) for i in set(yl)}
x_train1PID = X_train1["playerID"]
X_train1 = X_train1.drop(labels="playerID", axis=1)
x_test1PID = X_test1["playerID"]
X_test1 = X_test1.drop(labels="playerID", axis=1)

# Standardise
scaler1 = preprocessing.StandardScaler().fit(X_train1)
X_train1 = pd.DataFrame(scaler1.transform(X_train1.values), columns=X_train1.columns, index=X_train1.index)
X_test1 = pd.DataFrame(scaler1.transform(X_test1.values), columns=X_test1.columns, index=X_test1.index)
model1 = KNeighborsClassifier(n_neighbors=7).fit(X_train1, y_train1)
y_predGameTrain = model1.predict(X_train1)
y_predGameTest = model1.predict(X_test1)

game_train_accuracy = skm.accuracy_score(y_train1, y_predGameTrain)
print(f"Game train accuracy = {game_train_accuracy}")
game_test_accuracy = skm.accuracy_score(y_test1, y_predGameTest)
print(f"Game test accuracy = {game_test_accuracy}")

# Second classification ###
X_train1["playerID"] = x_train1PID
gb = X_train1.groupby(y_predGameTrain)
groupedByGame = [gb.get_group(x) for x in gb.groups]
trainGame1 = groupedByGame[0]
trainGame2 = groupedByGame[1]

# Game 1: Training
X_train21 = trainGame1.loc[:, trainGame1.columns != "playerID"]
y_train21 = trainGame1["playerID"]
scaler21 = preprocessing.StandardScaler().fit(X_train21)
X_train21 = pd.DataFrame(scaler21.transform(X_train21.values), columns=X_train21.columns, index=X_train21.index)

if useTF:
    model21 = tfModel(len(set(y_train21)))
if usePCA:
    X_train21 = pca1.fit_transform(X_train21)
if useICA:
    X_train21 = ica1.fit_transform(X_train21)
if useIsomap:
    X_train21 = iso1.fit_transform(X_train21)

# Clustering
#clusteringTrain1 = KMeans(n_clusters=k, n_init=3, random_state=0)
clusteringTrain1 = DBSCAN(eps=eps, min_samples=minSamples,n_jobs=1)
clusteringTrain1.fit(X_train21)
silhouetteTrain1 = skm.silhouette_score(X_train21, clusteringTrain1.labels_)
print("")
print(clusteringTrain1)
print(f"Silhoutette train 1: {silhouetteTrain1}")


# y_train21 = list(y_train21CPY)
# distributionTrain11 = {i: y_train21.count(i) for i in set(y_train21)}

# Game 2: Clustering
X_train22 = trainGame2.loc[:, trainGame2.columns != "playerID"]
y_train22 = trainGame2["playerID"]
scaler22 = preprocessing.StandardScaler().fit(X_train22)
X_train22 = pd.DataFrame(scaler22.transform(X_train22.values), columns=X_train22.columns, index=X_train22.index)


if useTF:
    model22 = tfModel(len(set(y_train22)))
if usePCA:
    X_train22 = pca2.fit_transform(X_train22)
if useICA:
    X_train22 = ica2.fit_transform(X_train22)
if useIsomap:
    X_train22 = iso2.fit_transform(X_train22)

# CLustering
#clusteringTrain2= KMeans(n_clusters=k, n_init=3, random_state=0)
clusteringTrain2 = DBSCAN(eps=eps, min_samples=minSamples,n_jobs=1)
clusteringTrain2.fit(X_train22)
silhouetteTrain2 = skm.silhouette_score(X_train22, clusteringTrain2.labels_)
print(f"\nSilhoutette train 2: {silhouetteTrain2}")
#print(clusteringTrain2)

# y_train22 = list(y_train22)
# distributionTrain12 = {i: y_train22.count(i) for i in set(y_train22)}

# Second classification test
X_test1["playerID"] = x_test1PID
gb = X_test1.groupby(y_predGameTest)
groupedByGame = [gb.get_group(x) for x in gb.groups]
testGame1 = groupedByGame[0]
testGame2 = groupedByGame[1]

# Game 1: Test
X_test21 = testGame1.loc[:, trainGame1.columns != "playerID"]
y_test21 = testGame1["playerID"]
X_test21 = pd.DataFrame(scaler21.transform(X_test21.values), columns=X_test21.columns, index=X_test21.index)
if usePCA:
    X_test21 = pca1.transform(X_test21)
if useICA:
    X_test21 = ica1.transform(X_test21)
if useIsomap:
    X_test21 = iso1.transform(X_test21)


# Test Clustering
#clusteringTest1 = KMeans(n_clusters=k, n_init=3, random_state=0)
clusteringTest1 = DBSCAN(eps=eps, min_samples=minSamples,n_jobs=1)
clusteringTest1.fit(X_test21)
silhouetteTest1 = skm.silhouette_score(X_test21, clusteringTest1.labels_)
print(f"\nSilhoutette test 1: {silhouetteTest1}")
#print(clusteringTest1)



# Game 2: Test
X_test22 = testGame2.loc[:, trainGame2.columns != "playerID"]
y_test22 = testGame2["playerID"]
X_test22 = pd.DataFrame(scaler22.transform(X_test22.values), columns=X_test22.columns, index=X_test22.index)
if usePCA:
    X_test22 = pca2.transform(X_test22)
if useICA:
    X_test22 = ica2.transform(X_test22)
if useIsomap:
    X_test22 = iso2.transform(X_test22)


# Test Clustering
#clusteringTest2 = KMeans(n_clusters=k, n_init=3, random_state=0)
clusteringTest2 = DBSCAN(eps=eps, min_samples=minSamples,n_jobs=1)
clusteringTest2.fit(X_test22)
silhouetteTest2 = skm.silhouette_score(X_test22, clusteringTest2.labels_)
print(f"\nSilhoutette test2: {silhouetteTest2}")
#print(clusteringTest2)

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start:.2f} second(s)")

debug = True
