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
        self.k = number_of_neighbours
        self.trainData = None
        self.trainLabels = None
        self.predictionResults = []

    # Fit function that takes the training data and the training labels,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_labels):
        self.trainData = training_data
        self.trainLabels = training_labels

    # Predict function that takes the testing data,
    # and iterates through the list passing each value to the prediction algorithm,
    # returns a list of predictions, and the confidence if set to True.
    def predict(self, testing_data, confidence=False):
        if len(testing_data) == 0:
            return -1

        for test_value in testing_data:
            self.predictionResults.append(self._predict(test_value, confidence))
        return self.predictionResults

    def _predict(self, test_value, confidence):
        distances = []
        for train_value in self.trainData:
            euclidean = 0
            for feature in range(train_value.shape[0]):
                euclidean += (test_value[feature] - train_value[feature]) ** 2
            distances.append(math.sqrt(euclidean))

        # Match the Euclidean distances with the training labels,
        # sort based on the distance - lowest to highest.
        distances = list(zip(distances, self.trainLabels))
        distances = sorted(distances, key=lambda x: x[0], reverse=False)
        distances = distances[:self.k]

        # Iterate through the list and save only the training labels.
        labels = []
        for distance in distances:
            labels.append(distance[1])

        most_common = Counter(labels).most_common(1)

        # Returns either 1D or 2D array
        if confidence:
            return most_common[0][0], (most_common[0][1] / self.k)
        return most_common[0][0]

    # Simple function to quick return the accuracy of the model as a decimal.
    def accuracy(self, testing_labels):
        # Check to see if the predictions list or the testing labels list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_labels.any():
            return -1

        # Try to calculate accuracy from the predicted results and provided testing labels,
        # and if the prediction results also contain the confidence values,
        # catch the value error (comparing 2D to 1D list),
        # then extract the labels and save as a new accuracy list,
        # calculate accuracy using new list and return accuracy value.
        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)
