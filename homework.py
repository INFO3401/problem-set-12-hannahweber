# Monday Problem
# Problem 1:

import pandas as pd

# import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def loadData(datafile):
    with open(datafile, 'r', encoding = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)
        
    #inspect the data
    print(data.columns.values)
    
    return data

def runKNN(dataset, prediction, ignore):
    
    # set up our dataset
    X = dataset.drop(columns = [prediction, ignore])
    Y = dataset[prediction].values
    
    # split the data into training and testing set
    # test size = what percent of the data do you want to test on
    # random_state = 1 = split them randomly
    # stratify = idk
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 1, stratify = Y)
    
    # run k-NN algorithm
    # n_neighbors = k-value
    knn = KNeighborsClassifier(n_neighbors = 5)
    
    # train the model
    knn.fit(X_train, Y_train)
    
    # test the model
    score = knn.score(X_test, Y_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy ")
    print("Chance is: " + str(1.0 / len(dataset.groupby(prediction))))
    
    return knn

nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
