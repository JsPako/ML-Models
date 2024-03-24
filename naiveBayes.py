import math
from collections import Counter


class NaiveBayes:

    #   Naive Bayes Classifier.

    #   This classifier implements the Naive Bayes algorithm for classification tasks. It calculates the prior
    #   probabilities of each class and the mean and standard deviation of each feature within each class during the
    #   training phase. During prediction, it calculates the likelihood of a data point belonging to each of the
    #   classes and selects the class with the highest likelihood as the predicted class.

    def __init__(self):

        #   Naive Bayes Classifier Constructor.

        #   The constructor does not require any user input, however it is essential that the instance variables
        #   including training data, training labels, class probabilities, feature mean, and standard deviation are
        #   initialised before the fit function is called.

        self.trainData = None
        self.trainLabels = None
        self.classProbability = {}
        self.featureMeanStandardDeviation = {}
        self.predictionResults = []

    def fit(self, training_data, training_labels):

        #   Fit Function

        #   This is the key function that trains the Naive Bayes classifier by calculating class prior probabilities
        #   and feature mean and standard deviation for each class. The calculated values are then used to form a
        #   normal distributions for each of the classes, which is later used for calculating the probabilities.

        self.trainData = training_data
        self.trainLabels = training_labels

        unique_classes = list(Counter(self.trainLabels).keys())
        count_of_unique_classes = list(Counter(self.trainLabels).values())
        size_of_classes = len(self.trainLabels)

        #   Iterate through the class name, and set the dictionary values to be "class_name" : "probability".
        for class_name, count in zip(unique_classes, count_of_unique_classes):
            self.classProbability[class_name] = count / size_of_classes

        #   Set up a dictionary and initialise the class names as keys, then filter training data into the dictionary,
        #   to make each class contain the appropriate data. This is used to separate the data so that each feature is
        #   analysed independently.
        data_dictionary = {}
        for class_name in unique_classes:
            data_dictionary[class_name] = []

        for class_name, data in zip(self.trainLabels, self.trainData):
            data_dictionary[class_name].append(data)

        #   Data dictionary that will be used to store the mean and standard deviation, those values are later used in
        #   the prediction phase to calculate likelihood.
        for class_name in unique_classes:
            self.featureMeanStandardDeviation[class_name] = []

        index = 0
        for data in data_dictionary.values():
            for feature in range(len(data[0])):
                mean = 0
                for row in data:
                    mean += row[feature]
                mean = mean / len(data)

                std = 0
                for row in data:
                    std += (row[feature] - mean) ** 2
                std = math.sqrt(std / len(data))

                self.featureMeanStandardDeviation[unique_classes[index]].append([mean, std])
            index += 1

    def predict(self, testing_data, confidence=False):

        #   Predict Function

        #   This function takes in a list of unseen data points, and returns the predicted class with the highest
        #   calculated likelihood for every data point. Optionally, will also return the confidence score if the
        #   default False is set to True. The confidence is calculated by taking the likelihood values and normalising
        #   them to be between 0 and 1.0.

        for row in testing_data:
            predictions = {}

            for class_name, class_probability in self.classProbability.items():
                predictions[class_name] = 1

                for feature, (mean, std) in enumerate(self.featureMeanStandardDeviation[class_name]):
                    predictions[class_name] *= self.probability_density_function(row[feature], mean, std)
                predictions[class_name] *= class_probability

            prediction = max(predictions, key=predictions.get)

            if confidence:
                normalisation = 0
                for value in predictions.values():
                    normalisation += value
                values = max(predictions.items(), key=lambda item: item[1])
                confidence = values[1] / normalisation

                prediction = max(predictions, key=predictions.get)
                self.predictionResults.append([prediction, confidence])
            else:
                self.predictionResults.append(prediction)

        return self.predictionResults

    @staticmethod
    def probability_density_function(value, mean, std):

        #   Probability Density Function

        #   Calculates the likelihood that a provided value is within the normal distribution. The calculated output
        #   from this function is used during the prediction phase to calculate the likelihood of the prediction
        #   belonging to that class.

        exponent = math.exp(- (value - mean) ** 2 / (2 * (std ** 2)))
        base = 1 / math.sqrt(2 * math.pi * (std ** 2))
        return base * exponent

    def accuracy(self, testing_labels):

        #   Accuracy Function

        #   This function is needed to evaluate the performance of the KNN algorithm, provided the testing classes this
        #   function checks how many of the predicted classes match the testing classes. Returning the decimal value of
        #   how many were correct. Based on this accuracy information you can tell if the model needs more optimisation
        #   by adjusting the number of nearest neighbours considered.

        if not self.predictionResults or not testing_labels.any():
            return -1

        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            #   Makes an accuracy list without the confidence values before trying the comparison again,
            #   it turns the 2D == 1D into 1D == 1D which Python can do.
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)
