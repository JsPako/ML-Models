from collections import Counter
from itertools import combinations

import numpy as np


class SupportVectorMachine:

    #   Support Vector Machine (SVM) Classifier

    #   This classifier predicts the class of a given data point by identifying a hyperplane that best separates the
    #   classes in the feature space. The hyperplane is generated during the fitting phase, with the use of a gradient
    #   descent and optimised using hyperparameters. During the prediction phase it calculates the distance of an
    #   unseen data point to the hyperplane and if the distance is positive the 'Class 1' is saved, otherwise 'Class -1'
    #   is saved, after all One Versus One predictions for the data point are made the majority class is returned as the
    #   prediction. Optionally, it can also provide the confidence score for the class predictions.

    def __init__(self):

        #   SVM Class Constructor

        #   The constructor does not require any user input, however it is essential that the coefficients array and
        #   intercepts array are initialised before the fit function is called. This constructor also initialises the
        #   unique classes array and pairings array which are required for the prediction phase to function correctly.

        self.trainData = None
        self.trainLabels = None
        self.uniqueClasses = []
        self.pairings = []
        self.coefficients = []
        self.intercepts = []
        self.predictionResults = []

    def fit(self, training_data, training_labels):

        #   Fit Function

        #   This is the key function in the SVM algorithm this function takes in all training data and training classes,
        #   separates the classes into pairings of 2 and converts the classes to be in the binary format (1, -1), as to
        #   prepare the data for the One Versus One hyperplane fitting.

        self.trainData = training_data
        self.trainLabels = training_labels

        self.uniqueClasses = list(set(self.trainLabels))
        self.pairings = list(combinations(self.uniqueClasses, 2))

        data_pairing = []
        for pairing in self.pairings:
            value_pairing = []

            for data in zip(self.trainData, self.trainLabels):
                if data[1] in pairing:
                    data = list(data)

                    #   Convert class names into binary 1 and -1.
                    if data[1] == pairing[0]:
                        data[1] = 1
                    else:
                        data[1] = -1

                    #   Return the new class name and the data that belongs to that class.
                    value_pairing.append(data)
            data_pairing.append(value_pairing)

        #   Call the gradient descent function for each of the pairings and fit a hyperplane for each OVO model. This
        #   is necessary as each pairing will have a different separation boundary.
        for data in data_pairing:
            self._gradient_descent(data)

    def _gradient_descent(self, data, epoch=5000, alpha=0.01):

        #   Internal Gradient Descent Function

        #   This calculates all hyperplane coefficients and intercepts for the data provided, using a gradient descent.
        #   The gradient descent can be optimised with the epoch (amount of iteration) and alpha (learning rate).
        #   Gradient descent optimisation is necessary because without it, the point at which the function converges
        #   will never be found with too little iterations, it is important to set the epoch value quite high. The
        #   convergence point might also be skipped over and never reached if the learning rate is set too high,
        #   therefore is it important for the learning rate to be a small value.

        gradient = [0] * len(data[0][0])
        intercept = 0
        for iteration in range(epoch):
            for x, y in data:
                #   Find the hinge loss of current data point. If it is above 0, which means the class was incorrectly
                #   identified update each feature coefficient and update the intercept.
                hinge_loss = max(0, 1 - y * (np.dot(x, gradient) - intercept))
                if hinge_loss > 0:
                    for index in range(len(x)):
                        gradient[index] -= (-y * x[index]) * alpha
                    intercept -= y * alpha

        #   Save the coefficients and intercept for the current One Versus One pairing.
        self.coefficients.append(gradient)
        self.intercepts.append(intercept)

    def predict(self, testing_data, confidence=False):

        #   Predict Function

        #   This function calculates the margin from the hyperplane to the unseen data point for each of the One Versus
        #   One pairings, and determines if that value is positive or negative. When the margin is positive the first
        #   class in the pairing is appended to the predictions otherwise the second class in the pairing is appended.
        #   Once all One Versus One margin calculations are made the majority class in the predictions is returned as
        #   the final prediction. Optionally, it can also return the confidence of the prediction, which is calculated
        #   by normalising the margin values.

        for sample in testing_data:
            class_prediction = []
            margins = []
            for index, pairing in enumerate(self.pairings):
                margin = np.dot(sample, self.coefficients[index]) - self.intercepts[index]
                margins.append(margin)
                if margin > 0:
                    class_prediction.append(pairing[0])
                else:
                    class_prediction.append(pairing[1])

            most_common = Counter(class_prediction).most_common(1)
            if confidence:
                max_margin = max(margins)
                min_margin = min(margins)
                confidence_value = (max_margin - min_margin) / max_margin
                if len(self.pairings) == 1:
                    confidence_value = 1.0
                self.predictionResults.append([most_common[0][0], confidence_value])
            else:
                self.predictionResults.append(most_common[0][0])

        return self.predictionResults

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
