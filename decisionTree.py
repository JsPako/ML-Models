from collections import Counter


class decisionTree:

    # Decision Tree Classifier Constructor
    # To initialise, the minimum allowed subset size needs to be provided,
    # The model will use this as the minimum size before forcing a leaf to be created.
    def __init__(self, minimum_subset_size):
        self.minSize = minimum_subset_size
        self.trainData = None
        self.trainLabels = None
        self.root = None
        self.predictionResults = []

    # Fit function that takes the training data and the training labels,
    # and sets them as variable instances inside the class,
    # then starts to build the decision tree.
    def fit(self, training_data, training_labels):
        self.trainData = training_data
        self.trainLabels = training_labels
        self.root = self._build(self.trainData, self.trainLabels)

    def predict(self, testing_data):
        for sample in testing_data:
            node = self.root
            while not node.leaf:
                if sample[node.featureIndex] <= node.value:
                    node = node.left
                else:
                    node = node.right
            self.predictionResults.append(node.prediction)
        return self.predictionResults

    @staticmethod
    def _gini_diversity_index(left_subset_labels, right_subset_labels):
        # Find the sizes of each subset and the total combined size.
        left_size = len(left_subset_labels)
        right_size = len(right_subset_labels)
        total_size = left_size + right_size
        
        left_proportion = 0
        for label in set(left_subset_labels):
            count = 0
            for sample in left_subset_labels:
                if sample == label:
                    count += 1
            left_proportion = left_proportion + ((count / left_size) ** 2)

        right_proportion = 0
        for label in set(right_subset_labels):
            count = 0
            for sample in right_subset_labels:
                if sample == label:
                    count += 1
            right_proportion = right_proportion + ((count / right_size) ** 2)

        return (((left_size / total_size) * (1.0 - left_proportion)) +
                ((right_size / total_size) * (1.0 - right_proportion)))

    def _build(self, subset, subset_labels):
        numRows, numColumns = subset.shape
        numLabels = len(np.unique(subset_labels))

        if numRows <= self.minSize or numLabels == 1:
            leaf = decisionTreeNode(is_leaf=True)
            leaf.prediction = Counter(subset_labels).most_common(1)[0][0]
            return leaf

        bestGini = 1.1
        bestDecisionFeature = None
        bestDecisionValue = None

        for feature in range(numColumns):
            for value in range(1, numRows - 1):

                leftSplit = np.where(subset[:, feature] <= value)[0]
                rightSplit = np.where(subset[:, feature] > value)[0]

                print(rightSplit)
                gini = self._gini_diversity_index(subset_labels[leftSplit], subset_labels[rightSplit])

                if gini < bestGini:
                    bestGini = gini
                    bestDecisionFeature = feature
                    bestDecisionValue = value

        if bestDecisionFeature is None:
            leaf = decisionTreeNode(is_leaf=True)
            leaf.prediction = Counter(subset_labels).most_common(1)[0][0]
            return leaf

        leftSplit = np.where(subset[:, bestDecisionFeature] <= bestDecisionValue)[0]
        rightSplit = np.where(subset[:, bestDecisionFeature] > bestDecisionValue)[0]

        leftTree = self._build(subset[leftSplit], subset_labels[leftSplit])
        rightTree = self._build(subset[rightSplit], subset_labels[rightSplit])

        return decisionTreeNode(decision_feature_index=bestDecisionFeature, decision_value=bestDecisionValue,
                                gini=bestGini, left=leftTree, right=rightTree)


class decisionTreeNode:

    # Decision Tree Node Constructor
    # It is an object that holds all the data required by the decision tree classifier.
    def __init__(self):
        self.featureIndex = None
        self.featureName = None
        self.valueIndex = None
        self.valueName = None
        self.gini = None
        self.leaf = False
        self.prediction = None
        self.confidence = None
        self.left = None
        self.right = None

    # Recursive function that prints a visual representation of the decision tree classifier.
    def display(self, indent=0, comparison="<=", prefix="Left node -"):
        if self.leaf:
            print(" " * indent + prefix + " Predicted Class:", self.prediction)
        else:
            print(" " * indent + f"Feature {self.featureName} {comparison} {self.valueName}, Gini: {self.gini}")
            self.left.display(indent + 5, comparison="<=", prefix="Left Node -")
            self.right.display(indent + 5, comparison=">", prefix="Right Node -")
