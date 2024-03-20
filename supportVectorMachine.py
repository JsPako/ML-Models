from itertools import combinations


class SupportVectorMachine:
    def __init__(self):
        self.trainData = None
        self.trainLabels = None
        self.coefficients = []
        self.intercepts = []
        self.predictionResults = []

    def fit(self, training_data, training_labels):
        self.trainData = training_data
        self.trainLabels = training_labels

        unique_classes = list(set(self.trainLabels))

        data_dictionary = {}
        for class_name in unique_classes:
            data_dictionary[class_name] = []

        for class_name, data in zip(self.trainLabels, self.trainData):
            data_dictionary[class_name].append(data)

        pairings = list(combinations(unique_classes, 2))
    def predict(self):
        pass

    def accuracy(self, testing_labels):

        #   Accuracy Function

        #   This function is needed to evaluate the performance of the SVM algorithm, provided the testing classes this
        #   function checks how many of the predicted classes match the testing classes. Returning the decimal value of
        #   how many were correct. Based on this accuracy information you can tell if the model needs more optimisation
        #   which is done by changing the epoch iteration number or by changing the alpha learning rate.

        if not self.predictionResults or not testing_labels.any():
            return -1

        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            #   Makes an accuracy list without the confidence values before trying the comparison again,
            #   it turns the 2D == 1D into 1D == 1D which Python can do.
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)