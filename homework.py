import pandas as pd
import seaborn as sns
import numpy as np

# import ML support libraries
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

def loadData(datafile):
    with open(datafile, 'r', encoding = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)
        
    #inspect the data
    print(data.columns.values)
    
    return data

# MONDAY PROBLEM - worked with Taylor and Marissa

def runKNN(dataset, prediction, ignore, neighbors):
    
    # set up our dataset
    X = dataset.drop(columns = [prediction, ignore])
    Y = dataset[prediction].values
    
    # split the data into training and testing set
    # test size = what percent of the data do you want to test on
    # random_state = 1 = split them randomly
    # stratify = idk
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.4, random_state = 1, stratify = Y)
    
    # run k-NN algorithm
    # n_neighbors = k-value
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    
    # train the model
    knn.fit(X_train, Y_train)
    
    # test the model
    score = knn.score(X_test, Y_test)
    Y_prediction = knn.predict(X_test)
    print("Predicts " + prediction + " with " + str(score) + " accuracy ")
    print("Chance is: " + str(1.0 / len(dataset.groupby(prediction))))
    print("F1 score: " + str(f1_score(Y_test, Y_prediction, average = 'macro')))
    
    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    X = targetRow.drop(columns = [prediction, ignore])
    
    # determine the 5 closest neighbors
    neighbors = model.kneighbors(X, n_neighbors = 5, return_distance = False)
    
    # print out the neighbors data
    for neighbor in neighbors[0]:
        print(data.iloc[neighbor])
        
def runkNNCrossfold(dataset, prediction, ignore, neighbors):
    fold = 0 # make a counter
    accuracies = [] # make an empty list
    kf = KFold(n_splits=neighbors) # use the KFold built-in model from sklearn

    X = dataset.drop(columns=[prediction, ignore]) # set up the x and y like we did above
    Y = dataset[prediction].values

    for train,test in kf.split(X): # make a for loop for each k folds split
        fold += 1 # counts which fold it is working on
        knn = KNeighborsClassifier(n_neighbors=neighbors) # uses the kneighbors classifier for each of the 3 k inputs
        knn.fit(X[train[0]:train[-1]], Y[train[0]:train[-1]]) # train the classifier on each of the folds but removes the last one for testing

        pred = knn.predict(X[test[0]:test[-1]]) # makes a test prediction on the different fold variations
        accuracy = accuracy_score(pred, Y[test[0]:test[-1]]) # computes the accuracy for the fold
        accuracies.append(accuracy) # appends the accuracy into the list made above
        print("Fold " + str(fold) + ":" + str(accuracy)) # makes a print statement so it organizes the output into each fold number with the accuracy attached

    return np.mean(accuracies) # returns 
    

nbaData = loadData("nba_2013_clean.csv")

# WEDNESDAY PROBLEMS
knnModel = runKNN(nbaData, "pos", "player", 3)

# Problem 3 - Both the F1 score and accuracy score were around 45% which means that the classifier isn't very effective. In order for the classifier to be trustworthy we would want the scores to be closer to 1.

for k in [5,7,10]:
    print("Folds: " + str(k))
    runkNNCrossfold(nbaData,"pos", "player", k)
