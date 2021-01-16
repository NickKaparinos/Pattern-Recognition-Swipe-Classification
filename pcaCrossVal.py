import time

import pandas as pd
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from preprocess import read_and_preprocess

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read and preprocess
kFolds = 4
X, y = read_and_preprocess(kFolds)

# Feature selection kbest
# = SelectKBest(k=10).fit_transform(X, y)
# kbest = SelectKBest(k=7).fit(X, y)
# mask = kbest.get_support()
# features = list(X.columns)
# features = [f for m,f in zip(mask,features) if m]
# X = kbest.transform(X)


X = preprocessing.StandardScaler().fit_transform(X)
pca = decomposition.PCA().fit(X)
Xpca = pca.transform(X)
model = svm.SVC(kernel='poly')

start = time.perf_counter()
print(model)
for i in range(9,10):
    print(f"i = {i}")
    model = svm.SVC(kernel='poly')
    XScores = Xpca[:, 0:i]
    skf = StratifiedKFold(n_splits=kFolds)
    # results = cross_val_score(estimator=model, X=Xpca, y=y, cv=skf)
    results = cross_val_score(estimator=model, X=XScores, y=y, cv=skf, n_jobs=kFolds)
    print(results)
    print(results.mean())
    print("\n")

    # parameters = {'C':[0.5],'kernel':['poly'], 'gamma':[1]}
    # gridSearch = GridSearchCV(model, parameters, cv=kFolds, n_jobs=1).fit(Xpca, y)
    # results = pd.DataFrame(gridSearch.cv_results_)
    # results = results.drop(labels=["std_fit_time","std_score_time","params","param_C","param_gamma","param_kernel", "rank_test_score"],axis=1)
    # print(results)

# print("Testing accuracy =  ", testing_accuracy)
end = time.perf_counter()
print(f"\nExecution time = {end - start}")

debug = True

# print("training accuracy")
# y_pred = svmModel.predict(X_train)
# training_accuracy = skm.accuracy_score(y_train, y_pred)
# print("Training accuracy = ", training_accuracy)
# print("testing accuracy")
# y_pred = model.predict(X_test)
# testing_accuracy = skm.accuracy_score(y_test, y_pred)

# Train Test split
# X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y , test_size=0.25, random_state=42)
# X_train = preprocessing.StandardScaler().fit_transform(X_train)
# X_test = preprocessing.StandardScaler().fit_transform(X_test)

# for i in modelList:
#     print(i)
#     skf = StratifiedKFold(n_splits=kFolds)
#     results = cross_val_score(i, X, y, cv=skf)
#     print(results)
#     print(results.mean())
#     print("\n")
