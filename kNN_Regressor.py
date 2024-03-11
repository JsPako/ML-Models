import math


class KNNRegressor:

    # kNN Regressor Constructor
    # To initialise, the number of neighbours the model will use needs to be provided.
    def __init__(self, number_of_neighbours):
        self.k = number_of_neighbours
        self.trainData = None
        self.trainValues = None
        self.predictionResults = []

    # Fit function that takes the training data and the training target values,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_values):
        self.trainData = training_data
        self.trainValues = training_values

    # Predict function that takes the testing data,
    # and iterates through the list passing each value to the prediction algorithm,
    def predict(self, testing_data):
        if len(testing_data) == 0:
            return -1

        for test_value in testing_data:
            self.predictionResults.append(self._predict(test_value))
        return self.predictionResults

    def _predict(self, test_value):
        distances = []
        for train_value in self.trainData:
            euclidean = 0
            for feature in range(train_value.shape[0]):
                euclidean += (test_value[feature] - train_value[feature]) ** 2
            distances.append(math.sqrt(euclidean))

        # Match the Euclidean distances with the training values,
        # sort based on the distance - lowest to highest.
        distances = list(zip(distances, self.trainValues))
        distances = sorted(distances, key=lambda x: x[0], reverse=False)
        distances = distances[:self.k]

        # Iterate through the list and save only the training values.
        distance_values = []
        for distance in distances:
            distance_values.append(distance[1])

        # Take the k number of nearest neighbours target values,
        # average the values and return that as the prediction.
        target_value = 0
        for value in distance_values:
            target_value += value
        target_value = target_value / len(distances)

        # Returns the predicted value.
        return target_value

    # Simple function to quick return the mean squared error of the model.
    def mean_squared_error(self, testing_values):
        # Check to see if the predictions list or the testing values list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_values.any():
            return -1

        # Try to calculate mean squared error from the predicted results and provided testing values,
        # then return the mean squared error.
        total = 0
        for predicted, true in zip(self.predictionResults, self.trainValues):
            total += (predicted - true) ** 2

        mse = total / len(testing_values)

        return mse
