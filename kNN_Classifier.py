import math
from collections import Counter


class KNNClassifier:

    #   K-Nearest Neighbour (KNN) Classifier

    #   This classifier predicts the class of a given data point by identifying the most common class among
    #   the 'k' nearest training data points within the feature space. This occurs during the prediction phase
    #   by taking in one given data point at a time, and calculating the distance from the input to all existing
    #   training data points. Distance is calculated using the Euclidean distance function, then sorted in
    #   ascending order, and by considering the classes among the 'k' number of training data points,
    #   retrieving the most common class for the class prediction. Optionally, it can also provide the confidence score
    #   for the class predictions.

    def __init__(self, number_of_neighbours):

        #   KNN Class Constructor

        #   The class needs to be constructed with the wanted number of neighbours (k), as that is required by
        #   the KNN algorithm to know how many neighbouring data points to consider when calculating the most common
        #   class. This value is also required for calculating the confidence score as that is calculated by dividing
        #   the number of neighbours that had the most common class by how many neighbours were considered.

        self.k = number_of_neighbours
        self.trainData = None
        self.trainLabels = None
        self.predictionResults = []

    def fit(self, training_data, training_labels):

        #   Fit Function

        #   KNN is a non-parametric learning algorithm, which means that when an unseen data point is given the
        #   algorithm needs to compare that data point to all training data points, therefore the training data has
        #   to be kept in memory, and cannot be discarded.

        self.trainData = training_data
        self.trainLabels = training_labels

    def predict(self, testing_data, confidence=False):

        #   Predict Function

        #   This function takes in a list of unseen data points, and returns the predicted class for every data point.
        #   Optionally, will also return the confidence score if the default False is set to True.

        if len(testing_data) == 0:
            return -1

        for test_value in testing_data:
            self.predictionResults.append(self._predict(test_value, confidence))
        return self.predictionResults

    def _predict(self, test_value, confidence):

        #   Internal Predict Function

        #   This function calculates the Euclidean distance from the passed data point to all training data points, and
        #   returns the most common class for that data point. This is the key function that contains all required
        #   calculations to make the KNN algorithm work correctly. Optionally, it also takes a confidence which is
        #   calculated by taking how many nearest data points have the majority class divided by how many data points
        #   where considered (k).

        distances = []
        for train_value in self.trainData:
            euclidean = 0
            for feature in range(train_value.shape[0]):
                #   Euclidean distance equation: Square root ( Sum of ( (test feature - train feature) ^ 2) )
                euclidean += (test_value[feature] - train_value[feature]) ** 2
            distances.append(math.sqrt(euclidean))

        #   Combine the calculated distances with the training classes, as to be able to sort using the distance values
        #   as the key, and then extract just the most common class from the combined list.
        distances = list(zip(distances, self.trainLabels))
        distances = sorted(distances, key=lambda x: x[0], reverse=False)
        distances = distances[:self.k]
        labels = []
        for distance in distances:
            labels.append(distance[1])
        most_common = Counter(labels).most_common(1)

        if confidence:
            return most_common[0][0], (most_common[0][1] / self.k)
        return most_common[0][0]

    def accuracy(self, testing_labels):

        #   Accuracy Function

        #   This function is needed to evaluate the performance of the KNN algorithm, providing the testing classes this
        #   function checks how many of the predicted classes match the testing classes. Returning the decimal value of
        #   how many were correct. Based on this accuracy information you can tell if the model needs more optimisation
        #   by adjusting the number of nearest neighbours considered.

        if not self.predictionResults or not testing_labels.any():
            return -1

        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            #   Makes an accuracy list without the confidence values before trying the comparison again,
            #   it turns the 2D == 1D into 1D == 1D which Python can do.
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)
