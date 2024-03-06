class NaiveBayes:

    # Naive Bayes Classifier Constructor
    def __init__(self):
        self.trainData = None
        self.trainLabels = None
        self.predictionResults = []

    # Fit function that takes the training data and the training labels,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_labels):
        self.trainData = training_data
        self.trainLabels = training_labels
