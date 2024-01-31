import pandas as pd
import numpy as np
from math import log2

# want to implement a random forest classifier, starting with a decision tree classifier

# Read the data

# calculate the entropy of the dataset, that has 2 classes (expected to be 0 and 1), a single column of 0s and 1
def entropy(data):
    if len(data) == 0:
        return 0
    # calculate the number of 0s and 1s
    class0 = len(data[data == 0]) / len(data)
    class1 = len(data[data == 1]) / len(data)
    # print(f'0s: {class0}, 1s: {class1}')
    if class0 == 0 or class1 == 0:
        return 0
    # calculate the entropy
    return -(class0 * log2(class0) + class1 * log2(class1))



# calculate the information gain of the dataset 
def info_gain(data,target,split_feature,split_value):


    # split the data into two parts
    left = data[data[split_feature] <= split_value]
    right = data[data[split_feature] > split_value]

    if len(left) == 0 or len(right) == 0:
        return 0

    # calculate the entropy of the left and right parts
    left_entropy = entropy(left[target])
    right_entropy = entropy(right[target])


    
    total_entropy = entropy(data[target])
    # calculate the information gain
    info_gain = total_entropy - (((len(left)/len(data))*left_entropy) + ((len(right)/len(data))*right_entropy))
    return info_gain

# exhaustive search to find the best split value for each feature
def find_best_split(data, target, feature):
    # find the best split value for the feature
    best_split_value = 0
    best_info_gain = 0
    # find the best split value for this feature
    for split_value in data[feature]:
        curr_info_gain = info_gain(data, target, feature, split_value)
        if curr_info_gain > best_info_gain:
            best_split_value = split_value
            best_info_gain = curr_info_gain

    return best_split_value, best_info_gain


def main():
    dataset = pd.read_csv('cancer.csv')
        

    for attribute in dataset.columns:
        best_split_value, best_info_gain = find_best_split(dataset, 'diagnosis(1=m, 0=b)', attribute)
        print(f'best split value: {best_split_value}, best info gain: {best_info_gain}')
        #print mean
        print(f'mean: {dataset[attribute].mean()}')
        #print minimum of that attribute
        print(f'min: {dataset[attribute].min()}')
        # values less than best split value
        print(f'split num: {len(dataset[dataset[attribute] > best_split_value])}')

    print(f'entropy of entire dataset: {entropy(dataset["diagnosis(1=m, 0=b)"])}')
if __name__ == "__main__":
    main()
