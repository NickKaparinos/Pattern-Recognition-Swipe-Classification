# Patter Recognition Project
# Nick Kaparinos
# Vasiliki Zarkadoula
# 2021
import time

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn import preprocessing
from sklearn.cluster import Birch
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from preprocess import read_and_preprocess

## OPTIONS ###
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
usePCA = False
useICA = False
useIsomap = False
useDBSCAN = False
nComponents = 10
kNeighbors = 15
eps = 2.2
minSamples = 10
nClusters = 6
linkType = 'complete'
pca1 = PCA(n_components=nComponents)
pca2 = PCA(n_components=nComponents)
ica1 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
ica2 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
iso1 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
iso2 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
print(f"Using PCA : {usePCA}")
print(f"Using ICA : {useICA}")
print(f"Using Isomap : {useIsomap}")
if usePCA or useICA or useIsomap:
    print(f"Number of components {nComponents}")

### Read data ###
kFolds = 4
silhouetteList = []

X, y = read_and_preprocess(True)

### First Classification ###
# Train test split
start = time.perf_counter()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, stratify=X['playerID'], test_size=0.25, random_state=0)
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

if usePCA:
    X_train21 = pca1.fit_transform(X_train21)
if useICA:
    X_train21 = ica1.fit_transform(X_train21)
if useIsomap:
    X_train21 = iso1.fit_transform(X_train21)

# Clustering
# clusteringTrain1 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
clusteringTrain1 = Birch(n_clusters=nClusters)
clusteringTrain1.fit(X_train21)
distributionTrain1 = {i: list(clusteringTrain1.labels_).count(i) for i in set(clusteringTrain1.labels_)}
silhouetteTrain1 = skm.silhouette_score(X_train21, clusteringTrain1.labels_)
silhouetteList.append(silhouetteTrain1)

print("")
print(clusteringTrain1)
if useDBSCAN:
    noisePercent = list(clusteringTrain1.labels_).count(0) / X_train21.shape[0]
    print(f"\nSilhoutette train 1: {silhouetteTrain1}\nNoise percent: {noisePercent}")
else:
    print(f"\nSilhoutette train 1: {silhouetteTrain1}")
print(distributionTrain1)

# Game 2: Clustering
X_train22 = trainGame2.loc[:, trainGame2.columns != "playerID"]
y_train22 = trainGame2["playerID"]
scaler22 = preprocessing.StandardScaler().fit(X_train22)
X_train22 = pd.DataFrame(scaler22.transform(X_train22.values), columns=X_train22.columns, index=X_train22.index)

if usePCA:
    X_train22 = pca2.fit_transform(X_train22)
if useICA:
    X_train22 = ica2.fit_transform(X_train22)
if useIsomap:
    X_train22 = iso2.fit_transform(X_train22)

# CLustering
# clusteringTrain2= KMeans(n_clusters=nClusters, n_init=3, random_state=0)
clusteringTrain2 = Birch(n_clusters=nClusters)
clusteringTrain2.fit(X_train22)
distributionTrain2 = {i: list(clusteringTrain2.labels_).count(i) for i in set(clusteringTrain2.labels_)}
silhouetteTrain2 = skm.silhouette_score(X_train22, clusteringTrain2.labels_)
silhouetteList.append(silhouetteTrain2)
if useDBSCAN:
    noisePercent = list(clusteringTrain2.labels_).count(0) / X_train22.shape[0]
    print(f"\nSilhoutette train 2: {silhouetteTrain2}\nNoise percent: {noisePercent}")
else:
    print(f"\nSilhoutette train 2: {silhouetteTrain2}")
print(distributionTrain2)

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
# clusteringTest1 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
clusteringTest1 = Birch(n_clusters=nClusters)
clusteringTest1.fit(X_test21)
distributionTest1 = {i: list(clusteringTest1.labels_).count(i) for i in set(clusteringTest1.labels_)}
silhouetteTest1 = skm.silhouette_score(X_test21, clusteringTest1.labels_)
silhouetteList.append(silhouetteTest1)
if useDBSCAN:
    noisePercent = list(clusteringTest1.labels_).count(0) / X_test21.shape[0]
    print(f"\nSilhoutette test1: {silhouetteTest1}\nNoise percent: {noisePercent}")
else:
    print(f"\nSilhoutette test1: {silhouetteTest1}")
print(distributionTest1)

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
# clusteringTest2 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
clusteringTest2 = Birch(n_clusters=nClusters)
clusteringTest2.fit(X_test22)
distributionTest2 = {i: list(clusteringTest2.labels_).count(i) for i in set(clusteringTest2.labels_)}
silhouetteTest2 = skm.silhouette_score(X_test22, clusteringTest2.labels_)
silhouetteList.append(silhouetteTest2)
if useDBSCAN:
    noisePercent = list(clusteringTest2.labels_).count(0) / X_test22.shape[0]
    print(f"\nSilhoutette test 2: {silhouetteTest2}\nNoise percent: {noisePercent}")
else:
    print(f"\nSilhoutette test 2: {silhouetteTest2}")
print(distributionTest2)

w = [X_train21.shape[0], X_train22.shape[0], X_test21.shape[0], X_test22.shape[0]]
weighted_silhouette = np.average(a=silhouetteList, weights=w)
print(f"\nTotal silhouette: {weighted_silhouette:.6f}")

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start:.2f} second(s)")
