# Project Recognission two level classification
import time
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
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
from preprocess import read_and_preprocess
from tfModel import tfModel
from sklearn.preprocessing import OneHotEncoder
import os

## OPTIONS ###
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
useTF = True
print(f"Use tensorflow = {useTF}")

### Read data ###
kFolds = 4
X, y = read_and_preprocess(kFolds, True)


# n = 9
# pca1 = PCA(n_components=n)
# pca2 = PCA(n_components=n)

### First Classification ###
# Train test split
start = time.perf_counter()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, stratify=X['playerID'], test_size=0.25, random_state=42)
yl = list(y)
distributionGame = {i: yl.count(i) for i in set(yl)}
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

### Second classification ###
X_train1["playerID"] = x_train1PID
gb = X_train1.groupby(y_predGameTrain)
groupedByGame = [gb.get_group(x) for x in gb.groups]
trainGame1 = groupedByGame[0]
trainGame2 = groupedByGame[1]

# Game 1: Training
X_train11 = trainGame1.loc[:, trainGame1.columns != "playerID"]
y_train11 = trainGame1["playerID"]
scaler11 = preprocessing.StandardScaler().fit(X_train11)
X_train11 = pd.DataFrame(scaler11.transform(X_train11.values), columns=X_train11.columns, index=X_train11.index)

# Models
model11 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,10,10), max_iter=100, random_state=1)
if(useTF):
    model11 = tfModel(len(set(y_train11)))
print(model11)
# pca1.fit(X_train11)
# X_train11 = pca1.transform(X_train11)

# Training
if(useTF):
    X_train11 = X_train11.to_numpy()
    y_train11CPY = y_train11
    y_train11 = y_train11.to_numpy()
    le1 = preprocessing.LabelEncoder()
    le1.fit(y_train11)
    enc1 = OneHotEncoder()
    y_train11 = enc1.fit_transform(y_train11.reshape(-1,1)).toarray()
    model11.fit(X_train11, y_train11, epochs=100)
else:
    model11.fit(X_train11, y_train11)

# y_train11 = list(y_train11CPY)
# distributionTrain11 = {i: y_train11.count(i) for i in set(y_train11)}

# Game 2: Training
X_train12 = trainGame2.loc[:, trainGame2.columns != "playerID"]
y_train12 = trainGame2["playerID"]
scaler12 = preprocessing.StandardScaler().fit(X_train12)
X_train12 = pd.DataFrame(scaler12.transform(X_train12.values), columns=X_train12.columns, index=X_train12.index)

# Models
model12 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,10,10), max_iter=100, random_state=1)
if(useTF):
    model12 = tfModel(len(set(y_train12)))
print(model12)
# pca2.fit(X_train12)
# X_train12 = pca2.transform(X_train12)

# Training
if(useTF):
    X_train12 = X_train12.to_numpy()
    y_train12 = y_train12.to_numpy()
    le2 = preprocessing.LabelEncoder()
    le2.fit(y_train12)
    enc2 = OneHotEncoder()
    y_train12 = enc2.fit_transform(y_train12.reshape(-1,1)).toarray()
    model12.fit(X_train12, y_train12, epochs=100)
else:
    model12.fit(X_train12, y_train12)

#y_train12 = list(y_train12)
#distributionTrain12 = {i: y_train12.count(i) for i in set(y_train12)}

# Second classification test
X_test1["playerID"] = x_test1PID
gb = X_test1.groupby(y_predGameTest)
groupedByGame = [gb.get_group(x) for x in gb.groups]
testGame1 = groupedByGame[0]
testGame2 = groupedByGame[1]

# Game 1: Test
X_test11 = testGame1.loc[:, trainGame1.columns != "playerID"]
y_test11 = testGame1["playerID"]
X_test11 = pd.DataFrame(scaler11.transform(X_test11.values), columns=X_test11.columns, index=X_test11.index)
if(useTF):
    X_test11 = X_test11.to_numpy()
    y_test11CPY = y_test11
    y_test11 = y_test11.to_numpy()
    y_pred11 = model11.predict(X_test11)

    y_pred11 = y_pred11.argmax(axis=-1)
    y_pred11 = le1.inverse_transform(y_pred11)
else:
    y_pred11 = model11.predict(X_test11)

testing_accuracy1 = skm.accuracy_score(y_test11, y_pred11)
print(f"Testing accuracy 11 = {testing_accuracy1}")
# y_test11 = list(y_test11CPY)
# distributionTest11 = {i: y_test11.count(i) for i in set(y_test11)}

# Game 2: Test
X_test12 = testGame2.loc[:, trainGame2.columns != "playerID"]
y_test12 = testGame2["playerID"]
X_test12 = pd.DataFrame(scaler12.transform(X_test12.values), columns=X_test12.columns, index=X_test12.index)
if(useTF):
    X_test12 = X_test12.to_numpy()
    y_test12 = y_test12.to_numpy()
    y_pred12 = model12.predict(X_test12)
    y_pred12 = y_pred12.argmax(axis=-1)
    y_pred12 = le2.inverse_transform(y_pred12)
else:
    y_pred12 = model12.predict(X_test12)
testing_accuracy2 = skm.accuracy_score(y_test12, y_pred12)
print(f"Testing accuracy 12 = {testing_accuracy2}")
# y_test12 = list(y_test12)
#distributionTest12 = {i: y_test12.count(i) for i in set(y_test12)}

w = [len(y_test11), len(y_test12)]
acc = [testing_accuracy1, testing_accuracy2]
weighted_acc = np.average(a=acc, weights=w)
print(f"Total accuracy = {weighted_acc}")

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start}")

debug = True
