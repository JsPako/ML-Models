import math


class KNNRegressor:

    #   K-Nearest Neighbour (KNN) Regressor

    #   This regressor predicts the value of a given data point by identifying what the average value among the 'k'
    #   nearest training data points within the feature space. This occurs during the prediction phase
    #   by taking in one given data point at a time, and calculating the distance from the input to all existing
    #   training data points. Distance is calculated using the Euclidean distance function, then sorted in
    #   ascending order, and by considering all the values among the 'k' number of training data points,
    #   calculating the average value for the prediction.

    def __init__(self, number_of_neighbours):

        #   KNN Class Constructor

        #   The class needs to be constructed with the wanted number of neighbours (k), as that is required by
        #   the KNN algorithm to know how many neighbouring data points to consider when calculating the average value.

        self.k = number_of_neighbours
        self.trainData = None
        self.trainValues = None
        self.predictionResults = []

    def fit(self, training_data, training_values):

        #   Fit Function

        #   KNN is a non-parametric learning algorithm, which means that when an unseen data point is given the
        #   algorithm needs to compare that data point to all training data points, therefore the training data has
        #   to be kept in memory, and cannot be discarded.

        self.trainData = training_data
        self.trainValues = training_values

    def predict(self, testing_data):

        #   Predict Function

        #   This function takes in a list of unseen data points, and returns the predicted value for every data point.

        if len(testing_data) == 0:
            return -1

        for test_value in testing_data:
            self.predictionResults.append(self._predict(test_value))
        return self.predictionResults

    def _predict(self, test_value):

        #   Internal Predict Function

        #   This function calculates the Euclidean distance from the passed data point to all training data points, and
        #   returns the averaged value for that data point. This is the key function that contains all required
        #   calculations to make the KNN algorithm work correctly.

        distances = []
        for train_value in self.trainData:
            euclidean = 0
            for feature in range(train_value.shape[0]):
                #   Euclidean distance equation: Square root ( Sum of ( (test feature - train feature) ^ 2) )
                euclidean += (test_value[feature] - train_value[feature]) ** 2
            distances.append(math.sqrt(euclidean))

        #   Combine the calculated distances with the training values, as to be able to sort using the distance values
        #   as the key, and then calculate the average value.
        distances = list(zip(distances, self.trainValues))
        distances = sorted(distances, key=lambda x: x[0], reverse=False)
        distances = distances[:self.k]
        distance_values = []
        for distance in distances:
            distance_values.append(distance[1])

        #   Calculate the average value and return that as the prediction.
        target_value = 0
        for value in distance_values:
            target_value += value
        target_value = target_value / len(distances)
        return target_value

    def mean_squared_error(self, testing_values):

        #   Mean Squared Error (MSE) Function

        #   This function is needed to evaluate the performance of the KNN algorithm, providing the testing values this
        #   function calculates the MSE of the predicted values against the true testing values. Based on the MSE value
        #   this function returns, you can tell how well the model is performing, and if the model needs adjusting
        #   which is done by changing the number of nearest neighbours considered.

        if not self.predictionResults or not testing_values.any():
            return -1

        total = 0
        for predicted, true in zip(self.predictionResults, testing_values):
            total += (predicted - true) ** 2
        mse = total / len(testing_values)

        return mse
