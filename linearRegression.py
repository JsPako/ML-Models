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
    def fit(self, training_data, training_values):
        self.trainData = training_data
        self.trainValues = training_values

        # Variables needed to perform the least squares calculation.
        sums_of_x = []
        sums_of_x2 = []
        sum_of_y = 0
        sums_of_xy = []

        # Iterate through the dependent variable,
        # and set the sum of all the values as 'sum_of_y'.
        for value in self.trainValues:
            sum_of_y += value

        # Iterate through each feature(independent variables),
        # and sum all the x values, x^2 values, and x * y values,
        # save each feature values separately by appending the summed values to an appropriate array.
        for feature in range(self.trainData.shape[1]):
            sum_x = 0
            sum_x2 = 0
            sum_xy = 0
            for x, y in zip(self.trainData, self.trainValues):
                sum_x += x[feature]
                sum_x2 += x[feature] ** 2
                sum_xy += x[feature] * y

            sums_of_x.append(sum_x)
            sums_of_x2.append(sum_x2)
            sums_of_xy.append(sum_xy)

        # Iterate through the prepared arrays,
        # and calculate a coefficient for each feature using the least squares method,
        # then use the coefficient to calculate the intercept for each feature.
        for feature in range(self.trainData.shape[1]):
            n = self.trainData.shape[0]
            self.coefficients.append(
                (n * sums_of_xy[feature]) - (sums_of_x[feature] * sum_of_y)
                / (n * sums_of_x2) - (sums_of_x2[feature]))

            self.intercepts.append(sum_of_y - (self.coefficients[feature] * sums_of_x[feature]) / n)

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




