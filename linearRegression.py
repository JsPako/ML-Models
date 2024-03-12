class LinearRegressor:

    # Linear Regressor Constructor
    def __init__(self):
        self.trainData = None
        self.trainValues = None
        self.coefficients = []
        self.intercepts = []
        self.predictionResults = []

    # Fit function that takes the training data and the training target values,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_values, epoch=5000, alpha=0.01):
        self.trainData = training_data
        self.trainValues = training_values

        for feature in range(self.trainData.shape[1]):
            gradient = 0
            intercept = 0

            for iteration in range(epoch):
                dl_gradient = 0
                dl_intercept = 0

                for x, y in zip(self.trainData, self.trainValues):
                    dl_gradient += -2 * x[feature] * (y - (gradient * x[feature] + intercept))
                    dl_intercept += -2 * (y - (gradient * x[feature] + intercept))

                gradient = gradient - (1 / len(self.trainData)) * dl_gradient * alpha
                intercept = intercept - (1 / len(self.trainData)) * dl_intercept * alpha

            self.coefficients.append(gradient)
            self.intercepts.append(intercept)


# Iterate through the provided testing values,
# for each feature in the row calculate the target value using the appropriate line of best fit,
# sum the predicted target values,
# average the final sum,
# and return the average as the final prediction.
def predict(self, testing_values):
    for row in testing_values:
        target_value = 0
        for feature in range(testing_values.shape[1]):
            target_value += self.coefficients[feature] * row[feature] + self.intercepts[feature]

        self.predictionResults.append(target_value / testing_values.shape[1])

    return self.predictionResults


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
