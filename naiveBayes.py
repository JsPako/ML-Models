import math
from collections import Counter


class NaiveBayes:

    # Naive Bayes Classifier Constructor
    def __init__(self):
        self.trainData = None
        self.trainLabels = None
        self.classProbability = {}
        self.featureMeanStandardDeviation = {}
        self.predictionResults = []

    # Fit function that takes the training data and the training labels,
    # and sets them as variable instances inside the class.
    def fit(self, training_data, training_labels):
        self.trainData = training_data
        self.trainLabels = training_labels

        # Get the required variables to calculate the prior probabilities.
        unique_classes = list(Counter(self.trainLabels).keys())
        count_of_unique_classes = list(Counter(self.trainLabels).values())
        size_of_classes = len(self.trainLabels)

        # Iterate through the class name,
        # and set the dictionary values to be "class_name" : "probability".
        for class_name, count in zip(unique_classes, count_of_unique_classes):
            self.classProbability[class_name] = count / size_of_classes

        # Set up a dictionary and initialise key values,
        # the filter training data into the dictionary,
        # keys are the label names, and the value is a 2D array containing the appropriate data,
        # 1D is the row, 2D is the feature within that row.
        data_dictionary = {}
        for class_name in unique_classes:
            data_dictionary[class_name] = []

        for class_name, data in zip(self.trainLabels, self.trainData):
            data_dictionary[class_name].append(data)

        # Set up the data dictionary that will be used to store the mean and standard deviation
        for class_name in unique_classes:
            self.featureMeanStandardDeviation[class_name] = []

        # Calculate the mean and standard deviation for each feature,
        # Each class should have N feature number of means and standard deviations,
        # E.G. - (4 Features) = Class_name : 4 means, 4 standard deviations per class.
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

    # Predict function that takes the testing data,
    # and iterates through the testing data,
    # and for each row calculates the likelihood that it belongs in each class,
    # gets the class label that has the highest likelihood and saves that as the prediction result,
    # returns a list of predictions, and a fake confidence if set to True.
    def predict(self, testing_data, confidence=False):
        for row in testing_data:
            predictions = {}

            for class_name, class_probability in self.classProbability.items():
                predictions[class_name] = 1
                for feature, (mean, std) in enumerate(self.featureMeanStandardDeviation[class_name]):
                    predictions[class_name] *= self.probability_density_function(row[feature], mean, std)
                predictions[class_name] *= class_probability

            prediction = max(predictions, key=predictions.get)
            if confidence:
                # Fake confidence value of 0.5,
                # it is 0.5 because I want the model to be treated as being unsure.
                prediction = [prediction, 0.5]
            self.predictionResults.append(prediction)

        return self.predictionResults

    # Calculate the likelihood that a provided value is within the normal distribution.
    @staticmethod
    def probability_density_function(value, mean, std):
        exponent = math.exp(- (value - mean) ** 2 / (2 * (std ** 2)))
        base = 1 / math.sqrt(2 * math.pi * (std ** 2))
        return base * exponent

# Simple function to quick return the accuracy of the model as a decimal.
    def accuracy(self, testing_labels):
        # Check to see if the predictions list or the testing labels list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_labels.any():
            return -1

        # Try to calculate accuracy from the predicted results and provided testing labels,
        # and if the prediction results also contain the confidence values,
        # catch the value error (comparing 2D to 1D list),
        # then extract the labels and save as a new accuracy list,
        # calculate accuracy using new list and return accuracy value.
        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)
