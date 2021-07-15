"""
Nick Kaparinos
User Swipe Classification
2021
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def read_and_preprocess(gameFlag):
    # Read data
    data = pd.read_csv("Datasets/swipes.csv")
    data = data.iloc[:50000, :]  # Use a subset of the full dataset

    # Preprocessing
    # Choose features
    if (gameFlag):
        data = data.iloc[:, [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20]]
    else:
        data = data.iloc[:, [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20]]

    le = preprocessing.LabelEncoder()
    temp = le.fit_transform(data.loc[:, 'playerID'])

    # Number of swipes per user distribution
    # sns.set_theme()
    # sns.color_palette("BrBG", 12)
    # ax = sns.histplot(temp,stat='density',bins=125,palette='tab10')
    # ax.set(xlabel="Number of swipes", ylabel = "Probability")
    # ax.set_title("Distribution of number of swipes per user")
    # plt.show()

    # Delete players with only few swipes
    counts = data["playerID"].value_counts(ascending=True)
    IDsToDelete = [i for i, v in counts.iteritems() if v <= 12]
    for i in IDsToDelete:
        data = data[data['playerID'] != i]

    le1 = preprocessing.LabelEncoder()
    data.loc[:, 'playerID'] = le1.fit_transform(data.loc[:, 'playerID'])

    # Transform screen column into game column
    if (gameFlag):
        # Games
        screen = data["screen"]
        uniqueScreens = screen.unique()
        uniqueGameTypes = set([i.split()[0] for i in uniqueScreens])

        screenDict = {}
        for screen in uniqueScreens:
            for game in uniqueGameTypes:
                if game in screen:
                    screenDict[screen] = game
                    break
        data["screen"] = data.copy()["screen"].map(screenDict)

    # Direction Feature split
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


    # Print Percentiles
    # for i in range(len(dataSVM.columns)):
    #     print(dataSVM.iloc[:, i].describe(percentiles=[0.001,0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]))

    # X and y
    if (gameFlag):
        X = data.loc[:, data.columns != "screen"]
        y = data["screen"]
    else:
        X = data.loc[:, data.columns != "playerID"]
        y = data["playerID"]
    return X, y
