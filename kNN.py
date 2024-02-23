import math
import pandas as pd
from collections import Counter


class kNN:
    # kNN Classifier Constructor
    # To initialise, the number of neighbours the model will use needs to be provided.
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
    # and iterates through the list passing each row to the prediction algorithm,
    # returns a list of predictions.
    def predict(self, testing_data):
        for test_value in testing_data:
            self.predictionResults.append(self._predict(test_value))
        return self.predictionResults

    # kNN Prediction Algorithm,
    # takes 1 testing data row with the same number of features as training data,
    # returns predicted label.
    def _predict(self, test_value):
        distances = []

        # For each row in the training data,
        # do the Euclidean distance equation with the testing row passed,
        # add the results to the distances list.
        for train_value in self.trainData:
            euclidean = 0
            for feature in range(train_value.shape[0]):
                euclidean += (test_value[feature] - train_value[feature]) ** 2
            distances.append(math.sqrt(euclidean))

        # Convert the distances list to a pandas dataframe
        distances = pd.DataFrame({'distance': distances})
        # Match each row of the calculated distances to the correct training labels
        distances["labels"] = self.trainLabels
        # Sort the calculated distance values from lowest to largest
        distances = distances.sort_values('distance')
        # Keep only the k number of lowest distance values
        distances = distances.iloc[:self.k]
        # Convert the leftover labels column to a python list
        distances = distances['labels'].tolist()
        # Get the most common label in the list and return it as the predicted value
        most_common = Counter(distances).most_common(1)
        return most_common[0][0]

    # Simple function to quick return the accuracy of the model as a decimal.
    def accuracy(self, testing_labels):
        # Check to see if the predictions list or the testing labels list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_labels.any():
            return -1
        return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
