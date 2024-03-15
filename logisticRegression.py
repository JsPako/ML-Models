import math


class LogisticRegression:
    def __init__(self):
        self.trainData = None
        self.trainValues = None
        self.coefficients = []
        self.intercepts = []
        self.predictionResults = []

    def fit(self, training_data, training_values, epoch, alpha):
        self.trainData = training_data
        self.trainValues = training_values

        for feature in range(self.trainData.shape[1]):
            gradient = 0
            intercept = 0

            # Gradient Descent
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

    def predict(self, testing_data):
        lower_bound = min(self.trainValues)
        upper_bound = max(self.trainValues)

        for row in testing_data:
            prediction = 0
            for feature in range(testing_data.shape[1]):
                probability = self.standard_logistic_function(row[feature], self.coefficients[feature],
                                                              self.intercepts[feature])
                prediction = math.log(probability / (1 - probability))
                prediction += max(min(prediction, upper_bound), lower_bound)
            self.predictionResults.append(prediction / testing_data.shape[1])
        return self.predictionResults

    @staticmethod
    def standard_logistic_function(value, coefficient, intercept):
        fx = 1 / (1 + math.exp(-(coefficient * value + intercept)))
        return fx

    # Simple function to quick return the mean squared error of the model.
    def mean_squared_error(self, testing_values):
        # Check to see if the predictions list or the testing values list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_values.any():
            return -1

        # Try to calculate mean squared error from the predicted results and provided testing values,
        # then return the mean squared error.
        total = 0
        for predicted, true in zip(self.predictionResults, testing_values):
            total += (predicted - true) ** 2

        mse = total / len(testing_values)

        return mse
