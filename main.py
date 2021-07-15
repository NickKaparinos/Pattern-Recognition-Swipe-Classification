# Patter Recognition Project
# Nick Kaparinos
# 2021
import time

import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from preprocess import read_and_preprocess

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # Read data
X,y = read_and_preprocess(False)
kFolds = 4

### Model ###
# SVM
model = svm.SVC()

### Grid Search ###
start = time.perf_counter()
X = preprocessing.StandardScaler().fit_transform(X)
parameters = {'kernel':['rbf','poly'], 'gamma':[0.5,0.75,1,1.25,1.5]}
gridSearch = GridSearchCV(model, parameters, cv=kFolds, n_jobs=kFolds).fit(X, y)
results = pd.DataFrame(gridSearch.cv_results_)
results = results.drop(labels=["std_fit_time","std_score_time","params"],axis=1)
print(results)

# Execution Time
end = time.perf_counter()
print(f"\nExecution time = {end - start}")
