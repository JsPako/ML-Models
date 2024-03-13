class DecisionTreeRegressor:

    # Decision Tree Regressor Constructor
    # To initialise, the minimum allowed subset size needs to be provided,
    # The model will use this as the minimum size before forcing a leaf to be created.
    def __init__(self, minimum_subset_size):
        self.minSize = minimum_subset_size
        self.trainData = None
        self.trainValues = None
        self.root = None
        self.predictionResults = []

    # Fit function that takes the training data and the training values,
    # and sets them as variable instances inside the class,
    # then starts to build the decision tree.
    def fit(self, training_data, training_values):
        self.trainData = training_data
        self.trainValues = training_values
        self.root = self._build(self.trainData, self.trainValues)

    # Iterate through the provided testing data,
    # then travel through the decision until a leaf is reached,
    # and return the assigned prediction within that leaf.
    def predict(self, testing_data):
        for sample in testing_data:
            # Start at the root node
            node = self.root
            while not node.leaf:
                if sample[node.featureIndex] <= node.valueName:
                    node = node.left
                else:
                    node = node.right
            self.predictionResults.append(node.prediction)
        return self.predictionResults

    @staticmethod
    def _variance(left_subset_values, right_subset_values):
        # Find the mean of the left and right subsets.
        left_mean = sum(left_subset_values) / len(left_subset_values)
        right_mean = sum(right_subset_values) / len(right_subset_values)

        # Calculate the variance for each split.
        left_variance = 0
        for value in left_subset_values:
            left_variance += (value - left_mean) ** 2

        right_variance = 0
        for value in right_subset_values:
            right_variance += (value - right_mean) ** 2

        total_size = len(left_subset_values) + len(right_subset_values)

        # Combine the two variance calculations into one and weight the result based on the split size.
        return ((len(left_subset_values) / total_size) * left_variance +
                (len(right_subset_values) / total_size) * right_variance)

    def _build(self, subset, subset_values):
        # Get the number of rows and columns in the provided subset.
        num_rows, num_columns = subset.shape

        # Check if the minimum subset size has been reached.
        if num_rows <= self.minSize:
            # Initialise a decision tree node,
            # set the node to be a leaf node,
            # and calculate the average prediction.
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True

            # Take all the values inside the split,
            # average the values and return that as the prediction.
            target_value = sum(subset_values) / len(subset_values)
            leaf_node.prediction = target_value
            return leaf_node

        # Initialise default values for finding the best split.
        best_variance = float('inf')
        best_decision_feature_index = None
        best_decision_value_index = None
        best_decision_value = None

        for feature in range(num_columns):
            # Sort the subset list as to get better variance results.
            sorted_subset_indices = subset[:, feature].argsort()

            # The sample range starts at index 1 and ends at length - 1,
            # that is to ensure the left split and the right split always contain at least 1 sample in them.
            for sample in range(1, num_rows):
                left_split_values = subset_values[sorted_subset_indices[:sample]]
                right_split_values = subset_values[sorted_subset_indices[sample:]]

                variance = self._variance(left_split_values, right_split_values)

                # If the GDI value of split is better than the saved current best split,
                # save the split as the new best one.
                if variance < best_variance:
                    best_variance = variance
                    best_decision_feature_index = feature
                    best_decision_value_index = sample
                    best_decision_value = subset[sorted_subset_indices[sample], feature]

        # If no better split can be found then create a leaf.
        if best_decision_value_index is None:
            # Initialise a decision tree node,
            # set the node to be a leaf node,
            # and choose the most common prediction,
            # and calculate the confidence percentage.
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True

            # Take all the values inside the split,
            # average the values and return that as the prediction.
            target_value = sum(subset_values) / len(subset_values)

            leaf_node.prediction = target_value
            return leaf_node

        # Split the subset based on the best split found
        sorted_subset_indices = subset[:, best_decision_feature_index].argsort()
        left_split_data = subset[sorted_subset_indices[:best_decision_value_index]]
        left_split_values = subset_values[sorted_subset_indices[:best_decision_value_index]]
        right_split_data = subset[sorted_subset_indices[best_decision_value_index:]]
        right_split_values = subset_values[sorted_subset_indices[best_decision_value_index:]]

        # Call the function recursively and build left and right splits.
        left_tree = self._build(left_split_data, left_split_values)
        right_tree = self._build(right_split_data, right_split_values)

        # Initialise a decision tree node,
        # save the calculated values into the appropriate object attributes,
        # and return the tree node.
        tree_node = DecisionTreeNode()
        tree_node.featureIndex = best_decision_feature_index
        tree_node.valueIndex = best_decision_value_index
        tree_node.valueName = best_decision_value
        tree_node.variance = best_variance
        tree_node.left = left_tree
        tree_node.right = right_tree

        return tree_node

    # Simple function to quick return the mean squared error of the model.
    def mean_squared_error(self, testing_values):
        # Check to see if the predictions list or the testing values list is empty,
        # if it is empty then return -1.
        if not self.predictionResults or not testing_values.any():
            return -1

        # Try to calculate mean squared error from the predicted results and provided testing values,
        # then return the mean squared error.
        total = 0
        for predicted, true in zip(self.predictionResults, testing_values):
            total += (predicted - true) ** 2

        mse = total / len(testing_values)

        return mse


class DecisionTreeNode:

    # Decision Tree Node Constructor
    # It is an object that holds all the data required by the decision tree regressor.
    def __init__(self):
        self.featureIndex = None
        self.valueIndex = None
        self.valueName = None
        self.variance = None
        self.leaf = False
        self.prediction = None
        self.left = None
        self.right = None

    # Recursive function that prints a visual representation of the decision tree regressor.
    def display(self, indent=0, prefix="Left node -"):
        if self.leaf:
            print(" " * indent + prefix + " Predicted Value:", self.prediction)
        else:
            print(" " * indent + f"Feature Index {self.featureIndex} <= {self.valueName}, Variance: {self.variance}")
            self.left.display(indent + 5, prefix="Left Node -")
            self.right.display(indent + 5, prefix="Right Node -")
