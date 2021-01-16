# Project Recognission assignment test1
import numpy as np
import pandas as pd


# import matplotlib.pyplot as plt

def read_and_preprocess(kFolds):
    # Read data
    data = pd.read_csv("Datasets/swipes.csv")
    data = data.iloc[:, :]

    ### Preprocessing ###
    # Choose features
    data = data.iloc[:, [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20]]
    # Delete players with only few swipes
    counts = data["playerID"].value_counts(ascending=True)
    IDsToDelete = [i for i, v in counts.iteritems() if v < kFolds]
    for i in IDsToDelete:
        data = data[data['playerID'] != i]

    # Rename playerIDs
    playerID = sorted(set(data["playerID"]))
    playerIDDict = dict(zip(playerID, list(range(len(playerID)))))
    data.loc[:, 'playerID'] = data.copy()["playerID"].map(playerIDDict)

    # Direction
    directionDictHorizontal = {"left": -1, "right": 1, "down": 0, "up": 0}
    directionDictVertical = {"left": 0, "right": 0, "down": -1, "up": 1}
    dataCopy = data.copy()
    data.loc[:, 'direction'] = data.copy()["direction"].map(directionDictHorizontal)
    data = data.rename(columns={"direction": "HorizontalDirection"})
    data["VerticalDirection"] = dataCopy.copy()["direction"].map(directionDictVertical)

    # Remove outliers from Horizontan acceleration, vertical acceleration, horizontal mean position
    data = data[data["horizontalMeanPosition"] >= 0]
    for i in ["horizontalAcceleration", "verticalAcceleration"]:
        data = data[data[i] > np.percentile(data[i], 0.1)]
        data = data[data[i] < np.percentile(data[i], 99.9)]

    # dataSVMCorrelation = data.corr()

    # for i in range(len(dataSVM.columns)):
    #     print(dataSVM.iloc[:, i].describe(percentiles=[0.001,0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]))

    # X and y
    X = data.loc[:, data.columns != "playerID"]
    y = data["playerID"]
    return X, y
