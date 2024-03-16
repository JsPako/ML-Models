from collections import Counter


class DecisionTreeClassifier:

    #   Decision Tree Classifier

    #   This classifier predicts the class of a given data point by making an acyclic graph, in each branching node of
    #   the graph, the data point feature is examined and if the feature is below a specific threshold, then the left
    #   branch is followed; otherwise the right branch is followed. When a leaf node is reached, it contains the class
    #   that data point belongs to. The specific thresholds are determined by splitting the training data samples into
    #   two, and calculating the Gini Diversity Index (GDI). The best GDI result is used as a threshold, this process is
    #   repeated recursively until all branches lead to leaf node. Leaf nodes are saved when they are below the provided
    #   minimum split size or if the split only contains one unique class. The class prediction is made by identifying
    #   the majority class in the branch split. Optionally, this class can also provide the confidence score for the
    #   class prediction.

    def __init__(self, minimum_subset_size):

        #   Decision Tree Constructor

        #   The class needs to be constructed with the minimum allowed subset size. The decision tree uses this value
        #   to force a leaf to be created.

        self.minSize = minimum_subset_size
        self.trainData = None
        self.trainLabels = None
        self.root = None
        self.predictionResults = []

    def fit(self, training_data, training_labels):

        #   Fit Function

        #   This function starts the recursive calling of the internal build function, this creates the tree graph that
        #   the model will use to follow in order to determine the class of an unseen data point.

        self.trainData = training_data
        self.trainLabels = training_labels
        self.root = self._build(self.trainData, self.trainLabels)

    def predict(self, testing_data, confidence=False):

        #   Predict Function

        #   This function takes in a list of unseen data points, and returns the predicted class for every data point.
        #   This is done by following the tree graph until a leaf is reached. Optionally, will also return the
        #   confidence score if the default False is set to True.

        for sample in testing_data:
            #   Start at the root of the tree graph.
            node = self.root
            while not node.leaf:
                #   Examine the data point value and if it is below the threshold follow the left node, otherwise
                #   follow the right node.
                if sample[node.featureIndex] <= node.valueName:
                    node = node.left
                else:
                    node = node.right
            if confidence:
                self.predictionResults.append([node.prediction, node.confidence])
            else:
                self.predictionResults.append(node.prediction)
        return self.predictionResults

    @staticmethod
    def _gini_diversity_index(left_subset_labels, right_subset_labels):

        #   Gini Diversity Index (GDI) Function

        #   This function calculates the Gini Diversity Index, provided the classes in the left and right split. The GDI
        #   is a necessary value because the algorithm uses this value to evaluate how good a specific split is.
        #   The lower the GDI value is the better the decision threshold, and vice versa.

        left_size = len(left_subset_labels)
        right_size = len(right_subset_labels)
        total_size = left_size + right_size

        #   Calculate the Gini Diversity Index for the left split.
        left_proportion = 0
        for label in set(left_subset_labels):
            count = 0
            for sample in left_subset_labels:
                if sample == label:
                    count += 1
            left_proportion = left_proportion + ((count / left_size) ** 2)

        #   Calculate the Gini Diversity Index for the right split.
        right_proportion = 0
        for label in set(right_subset_labels):
            count = 0
            for sample in right_subset_labels:
                if sample == label:
                    count += 1
            right_proportion = right_proportion + ((count / right_size) ** 2)

        #   Combine the two GDI calculations and weight the results based on the proportion of the split size.
        return (((left_size / total_size) * (1.0 - left_proportion)) +
                ((right_size / total_size) * (1.0 - right_proportion)))

    def _build(self, subset, subset_labels):

        #   Internal Build Function

        #   This is the key function in the Decision Tree algorithm as this function recursively calls itself, until
        #   all the training data has been processed and all decision branches lead to a leaf.

        num_rows, num_columns = subset.shape
        unique_labels = set(subset_labels)

        #   A predefined condition to force a leaf to be created, when either the minimum subset size has been or
        #   all of the classes in the split are the same. This is necessary so that the decision tree algorithm does not
        #   overfit the training data.
        if num_rows <= self.minSize or len(unique_labels) == 1:
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True
            #   The leaf node prediction is the majority class in the current split.
            leaf_node.prediction = Counter(subset_labels).most_common(1)[0][0]
            #   The confidence score is how many samples of the majority class there are in split divided by all samples
            #   present in the split.
            leaf_node.confidence = (Counter(subset_labels).most_common(1)[0][1] / num_rows)
            return leaf_node

        #   GDI is set to 1.1 as the max value the _gini_diversity_index function can return is 1.
        best_gini_diversity_index = 1.1
        best_decision_feature_index = None
        best_decision_value_index = None
        best_decision_value = None

        for feature in range(num_columns):
            #   Sorting the subset before splitting makes the classes be contiguous which improves the efficiency of
            #   the algorithm.
            sorted_subset_indices = subset[:, feature].argsort()

            for sample in range(1, num_rows):
                left_split_labels = subset_labels[sorted_subset_indices[:sample]]
                right_split_labels = subset_labels[sorted_subset_indices[sample:]]

                gini = self._gini_diversity_index(left_split_labels, right_split_labels)

                #   The best available decision threshold is saved until a better split is found, this is done in the
                #   building phase of the decision tree to ensure that an optimal decision threshold is found, which
                #   optimises the tree building process.
                if gini < best_gini_diversity_index:
                    best_gini_diversity_index = gini
                    best_decision_feature_index = feature
                    best_decision_value_index = sample
                    best_decision_value = subset[sorted_subset_indices[sample], feature]

        if best_decision_value_index is None:
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True
            leaf_node.prediction = Counter(subset_labels).most_common(1)[0][0]
            leaf_node.confidence = (Counter(subset_labels).most_common(1)[0][1] / num_rows)
            return leaf_node

        #   Sort the remaining training data and recursively call the build function until all branches lead to a leaf.
        sorted_subset_indices = subset[:, best_decision_feature_index].argsort()
        left_split_data = subset[sorted_subset_indices[:best_decision_value_index]]
        left_split_labels = subset_labels[sorted_subset_indices[:best_decision_value_index]]
        right_split_data = subset[sorted_subset_indices[best_decision_value_index:]]
        right_split_labels = subset_labels[sorted_subset_indices[best_decision_value_index:]]

        left_tree = self._build(left_split_data, left_split_labels)
        right_tree = self._build(right_split_data, right_split_labels)

        #   This creates a decision tree decision split node with information about the threshold condition and a
        #   necessary pointer to the left and right node so that the decision tree can be traversed during the
        #   prediction phase.
        tree_node = DecisionTreeNode()
        tree_node.featureIndex = best_decision_feature_index
        tree_node.valueIndex = best_decision_value_index
        tree_node.valueName = best_decision_value
        tree_node.gini = best_gini_diversity_index
        tree_node.left = left_tree
        tree_node.right = right_tree

        return tree_node

    def accuracy(self, testing_labels):

        #   Accuracy Function

        #   This function is needed to evaluate the performance of the Decision Tree algorithm, provided the testing
        #   classes this function checks how many of the predicted classes match the testing classes. Returning the
        #   decimal value of how many were correct. Based on this accuracy information you can tell if the model needs
        #   more optimisation by adjusting the minimum subset size.

        if not self.predictionResults or not testing_labels.any():
            return -1

        try:
            return (sum(self.predictionResults == testing_labels)) / len(testing_labels)
        except ValueError:
            #   Makes an accuracy list without the confidence values before trying the comparison again,
            #   it turns the 2D == 1D into 1D == 1D which Python can do.
            accuracy_list = [row[0] for row in self.predictionResults]
            return (sum(accuracy_list == testing_labels)) / len(testing_labels)


class DecisionTreeNode:

    #   Decision Tree Node

    #   This class object holds all the calculated information about a decision split or a leaf prediction. This is
    #   necessary in the prediction phase as the predict function will extract the information stored in this
    #   node object.

    def __init__(self):
        self.featureIndex = None
        self.valueIndex = None
        self.valueName = None
        self.gini = None
        self.leaf = False
        self.prediction = None
        self.confidence = None
        self.left = None
        self.right = None

    def display(self, indent=0, prefix="Left node -"):

        #   Display Function

        #   Recursive function called on a specific node of the tree, or the full tree by using the root node, this
        #   produces a visual representation of the tree graph in text form.

        if self.leaf:
            print(" " * indent + prefix + " Predicted Class:", self.prediction + " | Confidence:", self.confidence)
        else:
            print(" " * indent + f"Feature Index {self.featureIndex} <= {self.valueName}, Gini: {self.gini}")
            self.left.display(indent + 5, prefix="Left Node -")
            self.right.display(indent + 5, prefix="Right Node -")
