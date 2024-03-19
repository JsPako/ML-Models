class DecisionTreeRegressor:

    #   Decision Tree Regressor

    #   This regressor predicts the value of a given data point by making an acyclic graph, in each branching node of
    #   the graph, the data point feature is examined and if the feature is below a specific threshold, then the left
    #   branch is followed; otherwise the right branch is followed. When a leaf node is reached, it contains the
    #   value that data point is expected to have. The specific thresholds are determined by splitting the training
    #   data samples into two, and calculating the variance. The best variance result is used as a threshold,
    #   this process is repeated recursively until all branches lead to leaf node. Leaf nodes are saved when they are
    #   below the provided minimum split size. The value prediction is made by identifying the values in the branch
    #   split, and calculating the average which is used as the prediction for that leaf node.

    def __init__(self, minimum_subset_size):

        #   Decision Tree Constructor

        #   The class needs to be constructed with the minimum allowed subset size. The decision tree uses this value
        #   to force a leaf to be created.

        self.minSize = minimum_subset_size
        self.trainData = None
        self.trainValues = None
        self.root = None
        self.predictionResults = []

    def fit(self, training_data, training_values):

        #   Fit Function

        #   This function starts the recursive calling of the internal build function, this creates the tree graph that
        #   the model will use to follow in order to determine the value of an unseen data point.

        self.trainData = training_data
        self.trainValues = training_values
        self.root = self._build(self.trainData, self.trainValues)

    def predict(self, testing_data):

        #   Predict Function

        #   This function takes in a list of unseen data points, and returns the predicted value for every data point.
        #   This is done by following the tree graph until a leaf is reached.

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
            self.predictionResults.append(node.prediction)
        return self.predictionResults

    @staticmethod
    def _variance(left_subset_values, right_subset_values):

        #   Variance Function

        #   This function calculates the variance, provided the values in the left and right split. The variance
        #   is a necessary value because the algorithm uses this value to evaluate how good a specific split is.
        #   The lower the variance value is the better the decision threshold, and vice versa.

        left_mean = sum(left_subset_values) / len(left_subset_values)
        right_mean = sum(right_subset_values) / len(right_subset_values)

        #   Calculate the variance for each split.
        left_variance = 0
        for value in left_subset_values:
            left_variance += (value - left_mean) ** 2

        right_variance = 0
        for value in right_subset_values:
            right_variance += (value - right_mean) ** 2

        total_size = len(left_subset_values) + len(right_subset_values)

        #   Combine the two variance calculations into one and weight the result based on the split size.
        return ((len(left_subset_values) / total_size) * left_variance +
                (len(right_subset_values) / total_size) * right_variance)

    def _build(self, subset, subset_values):

        #   Internal Build Function

        #   This is the key function in the Decision Tree algorithm as this function recursively calls itself, until
        #   all the training data has been processed and all decision branches lead to a leaf.

        num_rows, num_columns = subset.shape

        #   A predefined condition to force a leaf to be created, when the minimum subset size has been met.
        #   This is necessary so that the decision tree algorithm does not overfit the training data.
        if num_rows <= self.minSize:
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True
            #   The leaf node prediction is the average value in the current split.
            target_value = sum(subset_values) / len(subset_values)
            leaf_node.prediction = target_value
            return leaf_node

        #   The variance is set to infinity as that is the worst case scenario.
        best_variance = float('inf')
        best_decision_feature_index = None
        best_decision_value_index = None
        best_decision_value = None

        for feature in range(num_columns):
            #   Sorting the subset before splitting makes the values be continuous which improves the efficiency of
            #   the algorithm.
            sorted_subset_indices = subset[:, feature].argsort()

            for sample in range(1, num_rows):
                left_split_values = subset_values[sorted_subset_indices[:sample]]
                right_split_values = subset_values[sorted_subset_indices[sample:]]

                variance = self._variance(left_split_values, right_split_values)

                #   The best available decision threshold is saved until a better split is found, this is done in the
                #   building phase of the decision tree to ensure that an optimal decision threshold is found, which
                #   optimises the tree building process.
                if variance < best_variance:
                    best_variance = variance
                    best_decision_feature_index = feature
                    best_decision_value_index = sample
                    best_decision_value = subset[sorted_subset_indices[sample], feature]

        if best_decision_value_index is None:
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True
            target_value = sum(subset_values) / len(subset_values)
            leaf_node.prediction = target_value
            return leaf_node

        #   Sort the remaining training data and recursively call the build function until all branches lead to a leaf.
        sorted_subset_indices = subset[:, best_decision_feature_index].argsort()
        left_split_data = subset[sorted_subset_indices[:best_decision_value_index]]
        left_split_values = subset_values[sorted_subset_indices[:best_decision_value_index]]
        right_split_data = subset[sorted_subset_indices[best_decision_value_index:]]
        right_split_values = subset_values[sorted_subset_indices[best_decision_value_index:]]

        left_tree = self._build(left_split_data, left_split_values)
        right_tree = self._build(right_split_data, right_split_values)

        #   This creates a decision tree decision split node with information about the threshold condition and a
        #   necessary pointer to the left and right node so that the decision tree can be traversed during the
        #   prediction phase.
        tree_node = DecisionTreeNode()
        tree_node.featureIndex = best_decision_feature_index
        tree_node.valueIndex = best_decision_value_index
        tree_node.valueName = best_decision_value
        tree_node.variance = best_variance
        tree_node.left = left_tree
        tree_node.right = right_tree

        return tree_node

    def mean_squared_error(self, testing_values):
        #   Mean Squared Error (MSE) Function

        #   This function is needed to evaluate the performance of the Decision Tree algorithm, provided the testing
        #   values this function calculates the MSE of the predicted values against the true testing values. Based on
        #   the MSE value this function returns, you can evaluate how well the model is performing, and if the model
        #   needs adjusting which is done by changing the minimum subset size.

        if not self.predictionResults or not testing_values.any():
            return -1

        total = 0
        for predicted, true in zip(self.predictionResults, testing_values):
            total += (predicted - true) ** 2

        mse = total / len(testing_values)

        return mse


class DecisionTreeNode:

    #   Decision Tree Node

    #   This class object holds all the calculated information about a decision split or a leaf prediction. This is
    #   necessary in the prediction phase as the predict function will extract the information stored in this
    #   node object.

    def __init__(self):
        self.featureIndex = None
        self.valueIndex = None
        self.valueName = None
        self.variance = None
        self.leaf = False
        self.prediction = None
        self.left = None
        self.right = None

    def display(self, indent=0, prefix="Left node -"):

        #   Display Function

        #   Recursive function called on a specific node of the tree, or the full tree by using the root node, this
        #   produces a visual representation of the tree graph in text form.

        if self.leaf:
            print(" " * indent + prefix + " Predicted Value:", self.prediction)
        else:
            print(" " * indent + f"Feature Index {self.featureIndex} <= {self.valueName}, Variance: {self.variance}")
            self.left.display(indent + 5, prefix="Left Node -")
            self.right.display(indent + 5, prefix="Right Node -")
