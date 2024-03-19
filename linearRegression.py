class LinearRegressor:

    #   Linear Regressor

    #   This algorithm predicts the value of a given data point by fitting a linear function per feature. The linear
    #   functions are determined through gradient descent optimisation, which occurs during the fitting phase. The
    #   calculated coefficients (gradients) and intercepts are later used in the prediction phase when given an unseen
    #   data point, the target value (y) is predicted using each available feature (x). The target values that are
    #   calculated are then averaged and returned as the prediction.

    def __init__(self):

        #   Linear Regression Class Constructor

        #   The constructor does not require any user input, however it is essential that the coefficients array and
        #   intercepts array are initialised before the fit function is called.

        self.trainData = None
        self.trainValues = None
        self.coefficients = []
        self.intercepts = []
        self.predictionResults = []

    def fit(self, training_data, training_values, epoch=5000, alpha=0.01):

        #   Fit Function

        #   This is the key function in the linear regression algorithm, it takes the training data points and using
        #   gradient descent the algorithm calculates the coefficients and intercepts one per feature. The gradient
        #   descent can be optimised with the epoch (amount of iteration) and alpha (learning rate). Gradient descent
        #   optimisation is necessary because without it, the point at which the function converges will never be found
        #   with too little iterations, it is important to set the epoch value quite high. The convergence point might
        #   also be skipped over and never reached if the learning rate is set too high, therefore is it important for
        #   the learning rate to be a small value.

        self.trainData = training_data
        self.trainValues = training_values

        for feature in range(self.trainData.shape[1]):
            gradient = 0
            intercept = 0

            #   Start the gradient descent with a cost function being partial derivative.
            for iteration in range(epoch):
                dl_gradient = 0
                dl_intercept = 0

                #   Find the partial derivative of the gradient and of the intercept, which will tell the algorithm if
                #   the function is decreasing or increasing. If the function is decreasing the algorithm knows the
                #   linear function is too low, so the gradient and/or the intercept is increased, and vice versa if the
                #   functions is increasing the gradient and/or intercept is decreased. If the partial derivative is a
                #   large number the amount of increase or decrease is large, and when the partial derivative is small,
                #   the algorithm knows the convergence point is approaching and the amount of increase or decrease is
                #   small as to not skip over it.
                for x, y in zip(self.trainData, self.trainValues):
                    dl_gradient += -2 * x[feature] * (y - (gradient * x[feature] + intercept))
                    dl_intercept += -2 * (y - (gradient * x[feature] + intercept))

                #   Update the gradient and intercept based on the partial derivative and control the size of the update
                #   using the alpha variable.
                gradient = gradient - (1 / len(self.trainData)) * dl_gradient * alpha
                intercept = intercept - (1 / len(self.trainData)) * dl_intercept * alpha

            self.coefficients.append(gradient)
            self.intercepts.append(intercept)

    def predict(self, testing_data):

        #   Predict Function

        #   This function takes unseen data points and calculates the target value (y) for each of the features using
        #   a linear function with the appropriate coefficient and intercept. The values from those calculations are
        #   averaged and then returned as the predicted value.

        for row in testing_data:
            target_value = 0
            for feature in range(testing_data.shape[1]):
                target_value += self.coefficients[feature] * row[feature] + self.intercepts[feature]

            self.predictionResults.append(target_value / testing_data.shape[1])

        return self.predictionResults

    def mean_squared_error(self, testing_values):

        #   Mean Squared Error (MSE) Function

        #   This function is needed to evaluate the performance of the linear regression, provided the testing values this
        #   function calculates the MSE of the predicted values against the true testing values. Based on the MSE value
        #   this function returns, you can evaluate how well the model is performing, and if the model needs adjusting
        #   which is done by changing the epoch iteration number or by changing the alpha learning rate.

        if not self.predictionResults or not testing_values.any():
            return -1

        total = 0
        for predicted, true in zip(self.predictionResults, testing_values):
            total += (predicted - true) ** 2

        mse = total / len(testing_values)

        return mse
