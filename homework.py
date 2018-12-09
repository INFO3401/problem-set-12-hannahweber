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

# Problems 1-3: - worked with Taylor and Marissa
# Problem 3 - Both the F1 score and accuracy score were around 45% which means that the classifier isn't very effective. In order for the classifier to be trustworthy we would want the scores to be closer to 1.
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

# ******** PLEASE READ *********
# Problems 1-3 and 6, Taylor, Marissa, and I all worked together and got the code on our own. 
# For Problems 4, 5, and 7, Jacob walked us (Taylor, Marissa, and I) through his code, what he did, and how he got there. The code for these 3 problems are from Jacob but the comments are from Marissa, Taylor, and I to show that we understand what he was saying and that we get how the code works. The 3 questions' code is the same as Jacob, Taylor, and Marissa but we just wanted to let you know why that is. We thought that it would be more beneficial to copy his code and comment it out rather than just leave it all blank, but it is ultimately up to you if you want to grade it or not. 

# Problem 4:
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

    return np.mean(accuracies) # returns the average of the accuracies for each fold

# Problem 5:
def determineK(dataset, prediction, ignore, k_vals):
    best_k = 0 # creates an integer value that will be replaced in the for loop
    best_accuracy = 0 # creates an integer value that will be replaced in the for loop

    for k in k_vals: # for loop to loop through each k (5, 7, 10) in the input
        current_k = runkNNCrossfold(dataset, prediction, ignore, k) # runs the kNN cross validation function that we created in problem 4 for each k value in k_vals
        if current_k > best_accuracy: # asks if the current k value (computed using runKNNCrossfold) was better than the best accuracy that is stored
            best_k = k # if that k is better than the stored best accuracy then set the variable k to the k value being looped through
            best_accuracy = current_k # stores the current k value as best accuracy

    print("Best k, accuracy = " + str(best_k) + ", " + str(best_accuracy)) # prints the best k and current k value accuracy as the output in the terminal

# Problem 6 - worked on with Taylor and Marissa (without Jacob's help)
def runKMeans(dataset, ignore, neighbors):
    # set up dataset
    X = dataset.drop(columns = ignore)
    
    # run k-means algorithm
    kmeans = KMeans(n_clusters = neighbors)
    
    # train the model
    kmeans.fit(X)
    
    # add the predictions to the dataframe
    dataset['cluster'] = pd.Series(kmeans.predict(X), index = dataset.index)
    
    return kmeans

# Problem 7:
#Adapted from: https://datascience.stackexchange.com/a/41125
def findClusterK(dataset, ignore):
    
    mean_distances = {} # creates an empty dictionary
    X = dataset.drop(columns=ignore) # sets up the dataset
    
    for n in np.arange(4,12):
        model = runKMeans(dataset, ignore, n) #run the model from problem 6
        mean_distances[n] = np.mean([np.min(x) for x in model.transform(X)]) # use .transform() to get the distances of the points from all clusters. Then use list comprehension to get the min of those distances for each point to get the distance from the cluster the point belongs to. Take the mean of that list to get average distance.

    print("Best k by average distance: " + str(min(mean_distances, key=mean_distances.get))) # prints the best k based on the average distance to the other points, then use .get to return the value in the mean_distances key
    
nbaData = loadData("nba_2013_clean.csv")

# Run the code:
knnModel = runKNN(nbaData, "pos", "player", 3)

for k in [5,7,10]:
    print("Folds: " + str(k))
    runkNNCrossfold(nbaData,"pos", "player", k)
    
determineK(nbaData,"pos", "player", [5,7,10])

kmeansModel = runKMeans(nbaData, ['pos', 'player'], 5)

findClusterK(nbaData, ['pos', 'player'])