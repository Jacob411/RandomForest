# Random Forest Binary Classifier

This is a Python implementation of a random forest binary classifier using only Pandas and NumPy libraries. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.

## Installation

Clone the repository or download the `random_forest.py` file and include it in your project directory.

## Dependencies

- Python 3.x
- Pandas
- NumPy

Install the required dependencies using pip:

```bash
pip install pandas numpy
Usage
python
Copy code
from random_forest import RandomForestClassifier

# Create a RandomForestClassifier object
rf_classifier = RandomForestClassifier(num_trees=10, max_depth=5, min_samples_split=2, min_samples_leaf=1)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on test data
predictions = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = rf_classifier.accuracy(y_test, predictions)
print("Accuracy:", accuracy)
Parameters
num_trees: The number of trees in the random forest.
max_depth: The maximum depth of the tree.
min_samples_split: The minimum number of samples required to split an internal node.
min_samples_leaf: The minimum number of samples required to be at a leaf node.
Data Format
The input data should be in the form of Pandas DataFrame for both features (X) and target (y).

License
This project is licensed under the MIT License - see the LICENSE file for details.

vbnet
