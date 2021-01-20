# Project Recognission assignment test1
import numpy as np
import pandas as pd


# import matplotlib.pyplot as plt

def read_and_preprocess(kFolds, gameFlag):
    # Read data
    data = pd.read_csv("Datasets/swipes.csv")
    data = data.iloc[:, :]
    #data = data.drop([3264,3265,3266])

    ### Preprocessing ###
    # Choose features
    if(gameFlag):
        data = data.iloc[:, [ 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20]]
    else:
        data = data.iloc[:, [ 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20]]

    # Delete players with only few swipes
    counts = data["playerID"].value_counts(ascending=True)
    IDsToDelete = [i for i, v in counts.iteritems() if v < kFolds]
    for i in IDsToDelete:
        data = data[data['playerID'] != i]

    # Rename playerIDs
    playerID = sorted(set(data["playerID"]))
    playerIDDict = dict(zip(playerID, list(range(len(playerID)))))
    data.loc[:, 'playerID'] = data.copy()["playerID"].map(playerIDDict)

    if(gameFlag):
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

        # IDsToDelete = []
        # gb = data.groupby('playerID')
        # for i,df in gb:
        #     gamesPlayed = set(df['screen'])
        #     allGames = set(['MathisisGame','FocusGame'])
        #     bothGamesPlayed = (gamesPlayed == allGames)
        #     if(not bothGamesPlayed):
        #         IDsToDelete.append(i)
        # for i in IDsToDelete:
        #     data = data[data['playerID'] != i]

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
    if(gameFlag):
        X = data.loc[:, data.columns != "screen"]
        y = data["screen"]
    else:
        X = data.loc[:, data.columns != "playerID"]
        y = data["playerID"]
    return X, y
