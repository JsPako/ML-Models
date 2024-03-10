class LinearRegressor:

    # Linear Regressor Constructor
    def __init__(self):
        self.trainData = None
        self.trainValues = None
        self.predictionResults = []

    # Fit function that takes the training data and the training target values,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_values):
        self.trainData = training_data
        self.trainValues = training_values
