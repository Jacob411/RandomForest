# Random Forest Binary Classifier

This is a Python implementation of a random forest binary classifier using only Pandas and NumPy libraries. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.

The decisionTree.py file is a fully functional decision tree classifier that can be used on its own. The forest.py file is a random forest classifier that uses the decision tree classifier to create a forest of trees. The random forest classifier can be used to train and predict on data.

## Installation

Clone the repository or download the `random_forest.py` file and include it in your project directory.

## Dependencies

- Python 3.x
- Pandas
- NumPy

Install the required dependencies using pip:

```bash
pip install pandas numpy
```
## Usage
```python

from forest import RandomForest

# Create a RandomForestClassifier object
rf_classifier = RandomForest(numTrees=10, maxDepth=10, minSamples=10, maxFeatures=6)

# Train the classifier
rf_classifier.fit(train_data, target_variable)

# Predict on test data
predictions = rf_classifier.predict(test_data)
```
### NOTE: the current main in forest.py will run the classifier on the breast cancer dataset from sklearn.

## Parameters
numTrees: The number of trees in the random forest.

maxDepth: The maximum depth of the tree.

minSamples: The minimum number of samples required to split an internal node.

maxFeatures: The number of features each tree is allowed to use.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

