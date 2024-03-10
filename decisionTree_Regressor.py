class DecisionTree:

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
    def _gini_diversity_index(left_subset_labels, right_subset_labels):
        # Find the sizes of each subset and the total combined size.
        left_size = len(left_subset_labels)
        right_size = len(right_subset_labels)
        total_size = left_size + right_size

        # Calculate the Gini Diversity Index for the left split.
        left_proportion = 0
        for label in set(left_subset_labels):
            count = 0
            for sample in left_subset_labels:
                if sample == label:
                    count += 1
            left_proportion = left_proportion + ((count / left_size) ** 2)

        # Calculate the Gini Diversity Index for the right split.
        right_proportion = 0
        for label in set(right_subset_labels):
            count = 0
            for sample in right_subset_labels:
                if sample == label:
                    count += 1
            right_proportion = right_proportion + ((count / right_size) ** 2)

        # Combine the two GDI calculations and weight the results based on the proportion of the split size.
        return (((left_size / total_size) * (1.0 - left_proportion)) +
                ((right_size / total_size) * (1.0 - right_proportion)))

    def _build(self, subset, subset_values):
        # Get the number of rows and columns in the provided subset,
        # turn the provided labels into a set then back into a list to get only unique labels.
        num_rows, num_columns = subset.shape
        unique_labels = set(subset_values)

        # Check if the minimum subset size has been reached,
        # and check if there's only one type of label in the subset label list.
        if num_rows <= self.minSize or len(unique_labels) == 1:
            # Initialise a decision tree node,
            # set the node to be a leaf node,
            # and choose the most common prediction,
            # and calculate the confidence percentage.
            leaf_node = DecisionTreeNode()
            leaf_node.leaf = True

            # Take all the values inside the split,
            # average the values and return that as the prediction.
            target_value = 0
            for value in subset_values:
                target_value += value
            target_value = target_value / len(subset_values)

            leaf_node.prediction = target_value
            return leaf_node

        # Initialise default values for finding the best split,
        # GDI is set to 1.1 as the max value the _gini_diversity_index function can return is 1.
        best_gini_diversity_index = 1.1
        best_decision_feature_index = None
        best_decision_value_index = None
        best_decision_value = None

        for feature in range(num_columns):
            # Sort the subset list as to get better GDI results.
            sorted_subset_indices = subset[:, feature].argsort()

            # The sample range starts at index 1 and ends at length - 1,
            # that is to ensure the left split and the right split always contain at least 1 sample in them.
            for sample in range(1, num_rows):
                left_split_labels = subset_values[sorted_subset_indices[:sample]]
                right_split_labels = subset_values[sorted_subset_indices[sample:]]

                gini = self._gini_diversity_index(left_split_labels, right_split_labels)

                # If the GDI value of split is better than the saved current best split,
                # save the split as the new best one.
                if gini < best_gini_diversity_index:
                    best_gini_diversity_index = gini
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
            target_value = 0
            for value in subset_values:
                target_value += value
            target_value = target_value / len(subset_values)

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
        tree_node.gini = best_gini_diversity_index
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
        for predicted, true in zip(self.predictionResults, self.trainValues):
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
        self.gini = None
        self.leaf = False
        self.prediction = None
        self.left = None
        self.right = None

    # Recursive function that prints a visual representation of the decision tree regressor.
    def display(self, indent=0, prefix="Left node -"):
        if self.leaf:
            print(" " * indent + prefix + " Predicted Value:", self.prediction)
        else:
            print(" " * indent + f"Feature Index {self.featureIndex} <= {self.valueName}, Gini: {self.gini}")
            self.left.display(indent + 5, prefix="Left Node -")
            self.right.display(indent + 5, prefix="Right Node -")
