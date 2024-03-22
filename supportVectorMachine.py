from itertools import combinations

import numpy as np


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

        pairings = list(combinations(unique_classes, 2))

        data_pairing = []
        for pairing in pairings:
            value_pairing = []
            for data in zip(self.trainData, self.trainLabels):
                if data[1] in pairing:
                    data = list(data)
                    if data[1] == pairing[0]:
                        data[1] = -1
                    else:
                        data[1] = 1
                    value_pairing.append(data)
            data_pairing.append(value_pairing)

        for data in data_pairing:
            self._gradient_descent(data)

    def _gradient_descent(self, data, epoch=5000, alpha=0.01):
        gradient = [0] * len(data[0][0])
        intercept = 0
        for iteration in range(epoch):
            for x, y in data:
                hinge_loss = max(0, 1 - y * (np.dot(x, gradient) - intercept))
                if hinge_loss > 0:
                    for index in range(len(x)):
                        gradient[index] -= (-y * x[index]) * alpha
                    intercept -= y * alpha

        self.coefficients.append(gradient)
        self.intercepts.append(intercept)

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
