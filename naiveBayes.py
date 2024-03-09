from collections import Counter
import math


class NaiveBayes:

    # Naive Bayes Classifier Constructor
    def __init__(self):
        self.trainData = None
        self.trainLabels = None
        self.classProbability = {}
        self.featureProbability = {}
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
        mean_std = {}
        for class_name in unique_classes:
            mean_std[class_name] = []

        # Calculate the mean and standard deviation for each feature,
        # Each class should have N feature number of means and standard deviations,
        # E.G. - (4 Features) = Class_name : 4 means, 4 standard deviations per class.
        index = 0
        for data in data_dictionary.values():
            for feature in range(len(data[0])):
                mean = 0
                std = 0
                for row in data:
                    mean += row[feature]
                    std += row[feature]

                mean = mean / len(data)
                std = math.sqrt((std - mean) ** 2 / len(data))

                mean_std[unique_classes[index]].append([mean, std])

            index += 1
