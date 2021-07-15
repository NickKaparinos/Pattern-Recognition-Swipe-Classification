# Project Recognission two level classification
import time
from statistics import mean

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn import preprocessing, svm
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from preprocess import read_and_preprocess
from tfModel import tfModel

## OPTIONS ###
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
useTF = True
usePCA = False
useICA = False
useIsomap = False
useLLE = False
useClustering = False
standardiseClusterLabels = False
nEpochs = 10
nComponents = 14
kNeighbors = 15
nClusters = 6
linkType = 'ward'
pca1 = PCA(n_components=nComponents)
pca2 = PCA(n_components=nComponents)
ica1 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
ica2 = FastICA(n_components=nComponents, max_iter=800, random_state=0)
iso1 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
iso2 = Isomap(n_neighbors=kNeighbors, n_components=nComponents)
lle1 = LocallyLinearEmbedding(n_components=nComponents, random_state=0)
lle2 = LocallyLinearEmbedding(n_components=nComponents, random_state=0)
print(f"Using tensorflow : {useTF}")
print(f"Using PCA : {usePCA}")
print(f"Using ICA : {useICA}")
print(f"Using Isomap : {useIsomap}")
print(f"Using LLE : {useLLE}")
print(f"Using Clustering : {useClustering}")
if usePCA or useICA or useIsomap or useLLE:
    print(f"Number of components {nComponents}")
print("\n")

### Read data ###
X, y = read_and_preprocess(True)
kFolds = 4

### First Classification ###
# Train test split
start = time.perf_counter()
skf = StratifiedKFold(n_splits=kFolds, random_state=0, shuffle=True)
iterationNumber = 0
foldAccuracy = []

for train_index, test_index in skf.split(X, X['playerID']):
    # Partition sets
    X_train1 = X.iloc[train_index]
    y_train1 = y.iloc[train_index]
    X_test1 = X.iloc[test_index]
    y_test1 = y.iloc[test_index]

    iterationNumber += 1

    x_train1PID = X_train1["playerID"]
    X_train1 = X_train1.drop(labels="playerID", axis=1)
    x_test1PID = X_test1["playerID"]
    X_test1 = X_test1.drop(labels="playerID", axis=1)

    # Standardise
    scaler1 = preprocessing.StandardScaler().fit(X_train1)
    X_train1 = pd.DataFrame(scaler1.transform(X_train1.values), columns=X_train1.columns, index=X_train1.index)
    X_test1 = pd.DataFrame(scaler1.transform(X_test1.values), columns=X_test1.columns, index=X_test1.index)

    ### First Classification
    model1 = KNeighborsClassifier(n_neighbors=7).fit(X_train1, y_train1)
    y_predGameTrain = model1.predict(X_train1)
    y_predGameTest = model1.predict(X_test1)

    ### Second classification ###
    X_train1["playerID"] = x_train1PID

    # Group swipes by game
    gb = X_train1.groupby(y_predGameTrain)
    groupedByGame = [gb.get_group(x) for x in gb.groups]
    trainGame1 = groupedByGame[0]
    trainGame2 = groupedByGame[1]

    # Game 1: Training
    X_train21 = trainGame1.loc[:, trainGame1.columns != "playerID"]
    y_train21 = trainGame1["playerID"]
    scaler21 = preprocessing.StandardScaler().fit(X_train21)
    X_train21 = pd.DataFrame(scaler21.transform(X_train21.values), columns=X_train21.columns, index=X_train21.index)

    # Models
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    knn = KNeighborsClassifier(n_neighbors=11)
    svc = svm.SVC(gamma=1, kernel='poly')
    svc2 = svm.SVC(gamma=0.5, kernel='poly')
    svc3 = svm.SVC(gamma=1.5, kernel='poly')
    svc4 = svm.SVC(gamma=0.4, kernel='poly')
    svc5 = svm.SVC(gamma=2.5, kernel='poly')
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80), max_iter=300, random_state=0)
    mlp2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80, 80), max_iter=300, random_state=0)
    mlp3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80), max_iter=300, random_state=0)
    mlp4 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80, 80, 80), max_iter=300,
                         random_state=0)
    forest = RandomForestClassifier(n_estimators=75, max_features=8, random_state=0, n_jobs=1)
    bagging = BaggingClassifier(max_features=8, n_estimators=40, n_jobs=1, random_state=0)
    model21 = VotingClassifier(
        estimators=[('mlp', mlp), ('mlp2', mlp2), ('mlp3', mlp3), ('mlp4', mlp4), ('forest', forest), ('svc', svc)],
        voting='hard')
    if useClustering:
        # clusteringTrain1 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
        clusteringTrain1 = Birch(n_clusters=nClusters)
        labels = clusteringTrain1.fit_predict(X_train21)
        if standardiseClusterLabels:
            labels = preprocessing.StandardScaler().fit_transform(labels.reshape(-1, 1))
        X_train21['ClusterLabel'] = labels
    if useTF:
        model21 = tfModel(len(set(y_train21)))
    if usePCA:
        X_train21 = pca1.fit_transform(X_train21)
    if useICA:
        X_train21 = ica1.fit_transform(X_train21)
    if useIsomap:
        X_train21 = iso1.fit_transform(X_train21)
    if useLLE:
        X_train21 = lle1.fit_transform(X_train21)

    # Training
    if useTF:
        if not (usePCA or useICA or useIsomap or useLLE):
            X_train21 = X_train21.to_numpy()
        y_train21CPY = y_train21
        y_train21 = y_train21.to_numpy()

        LE1 = preprocessing.LabelEncoder()
        LE1.fit(y_train21)
        OneHot1 = OneHotEncoder()
        y_train21 = OneHot1.fit_transform(y_train21.reshape(-1, 1)).toarray()

        model21.fit(X_train21, y_train21, validation_split=0.1, epochs=nEpochs)
        if (iterationNumber == 1):
            print(model21.summary())
    else:
        model21.fit(X_train21, y_train21)
        if (iterationNumber == 1):
            print(model21)

    # Game 2: Training
    X_train22 = trainGame2.loc[:, trainGame2.columns != "playerID"]
    y_train22 = trainGame2["playerID"]
    scaler22 = preprocessing.StandardScaler().fit(X_train22)
    X_train22 = pd.DataFrame(scaler22.transform(X_train22.values), columns=X_train22.columns, index=X_train22.index)

    # Models
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    knn = KNeighborsClassifier(n_neighbors=11)
    svc = svm.SVC(gamma=1, kernel='poly')
    svc2 = svm.SVC(gamma=0.5, kernel='poly')
    svc3 = svm.SVC(gamma=1.5, kernel='poly')
    svc4 = svm.SVC(gamma=0.4, kernel='poly')
    svc5 = svm.SVC(gamma=2.5, kernel='poly')
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80), max_iter=300, random_state=0)
    mlp2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80, 80), max_iter=300, random_state=0)
    mlp3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80), max_iter=300, random_state=0)
    mlp4 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80, 80, 80, 80, 80), max_iter=300,
                         random_state=0)
    forest = RandomForestClassifier(n_estimators=75, max_features=8, random_state=0, n_jobs=1)
    bagging = BaggingClassifier(max_features=8, n_estimators=40, n_jobs=1, random_state=0)
    model22 = VotingClassifier(
        estimators=[('mlp', mlp), ('mlp2', mlp2), ('mlp3', mlp3), ('mlp4', mlp4), ('forest', forest), ('svc', svc)],
        voting='hard')
    if useClustering:
        clusteringTrain2 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
        labels = clusteringTrain2.fit_predict(X_train22)
        if standardiseClusterLabels:
            labels = preprocessing.StandardScaler().fit_transform(labels.reshape(-1, 1))
        X_train22['ClusterLabel'] = labels
    if useTF:
        model22 = tfModel(len(set(y_train22)))
    if usePCA:
        X_train22 = pca2.fit_transform(X_train22)
    if useICA:
        X_train22 = ica2.fit_transform(X_train22)
    if useIsomap:
        X_train22 = iso2.fit_transform(X_train22)
    if useLLE:
        X_train22 = lle2.fit_transform(X_train22)

    # Training
    if useTF:
        if not (usePCA or useICA or useIsomap or useLLE):
            X_train22 = X_train22.to_numpy()
        y_train22 = y_train22.to_numpy()

        LE2 = preprocessing.LabelEncoder()
        LE2.fit(y_train22)
        OneHot2 = OneHotEncoder()
        y_train22 = OneHot2.fit_transform(y_train22.reshape(-1, 1)).toarray()

        model22.fit(X_train22, y_train22, validation_split=0.1, epochs=nEpochs)
        if (iterationNumber == 1):
            print(model22.summary())
    else:
        model22.fit(X_train22, y_train22)
        if (iterationNumber == 1):
            print(model22)

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
    if useClustering:
        clusteringTest1 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
        labels = clusteringTest1.fit_predict(X_test21)
        if standardiseClusterLabels:
            labels = preprocessing.StandardScaler().fit_transform(labels.reshape(-1, 1))
        X_test21['ClusterLabel'] = labels
    if usePCA:
        X_test21 = pca1.transform(X_test21)
    if useICA:
        X_test21 = ica1.transform(X_test21)
    if useIsomap:
        X_test21 = iso1.transform(X_test21)
    if useLLE:
        X_test21 = lle1.transform(X_test21)
    if useTF:
        y_test21CPY = y_test21
        if not (usePCA or useICA or useIsomap or useLLE):
            X_test21 = X_test21.to_numpy()
        y_test21 = y_test21.to_numpy()
        y_pred21 = model21.predict(X_test21)

        y_pred21 = y_pred21.argmax(axis=-1)
        y_pred21 = LE1.inverse_transform(y_pred21)
    else:
        y_pred21 = model21.predict(X_test21)

    testing_accuracy1 = skm.accuracy_score(y_test21, y_pred21)
    print(f"Testing accuracy 1 = {testing_accuracy1}")

    # Game 2: Test
    X_test22 = testGame2.loc[:, trainGame2.columns != "playerID"]
    y_test22 = testGame2["playerID"]
    X_test22 = pd.DataFrame(scaler22.transform(X_test22.values), columns=X_test22.columns, index=X_test22.index)
    if useClustering:
        clusteringTest2 = KMeans(n_clusters=nClusters, n_init=3, random_state=0)
        labels = clusteringTest2.fit_predict(X_test22)
        if standardiseClusterLabels:
            labels = preprocessing.StandardScaler().fit_transform(labels.reshape(-1, 1))
        X_test22['ClusterLabel'] = labels
    if usePCA:
        X_test22 = pca2.transform(X_test22)
    if useICA:
        X_test22 = ica2.transform(X_test22)
    if useIsomap:
        X_test22 = iso2.transform(X_test22)
    if useLLE:
        X_test22 = lle2.transform(X_test22)
    if useTF:
        if not (usePCA or useICA or useIsomap or useLLE):
            X_test22 = X_test22.to_numpy()
        y_test22 = y_test22.to_numpy()
        y_pred22 = model22.predict(X_test22)
        y_pred22 = y_pred22.argmax(axis=-1)
        y_pred22 = LE2.inverse_transform(y_pred22)
    else:
        y_pred22 = model22.predict(X_test22)
    testing_accuracy2 = skm.accuracy_score(y_test22, y_pred22)
    print(f"Testing accuracy 2 = {testing_accuracy2}")

    # Calculate fold accuracy
    w = [len(y_test21), len(y_test22)]
    acc = [testing_accuracy1, testing_accuracy2]
    weighted_acc = np.average(a=acc, weights=w)
    print(f"Iteration {iterationNumber}: Total accuracy = {weighted_acc}\n")
    foldAccuracy.append(weighted_acc)

print(f"{kFolds}-fold accuracy = {mean(foldAccuracy):.8f}")
# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start:.2f} second(s)")
