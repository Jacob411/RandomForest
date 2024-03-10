#class for our decision tree

import numpy as np
import pandas as pd
from pandas.core.frame import pandas 
from decision_tree_utils import * 
import pickle



class Node:
    '''
    Node class for our decision tree
    '''
    def __init__(self, data : pandas.DataFrame, split_feature : str, split_value : float):
        self.data = data
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = None
        self.right = None 
    
    def display(self):
        print(f'length of data: {len(self.data)}')
        print(f'split feature: {self.split_feature}')
        print(f'split value: {self.split_value}')
        print(f' is left None: {self.left == None}')
        print(f' is right None: {self.right == None}\n')




def build_tree(data : pandas.DataFrame, target, max_depth : int, min_samples_split : int, depth : int):
    '''
    Function to build the decision tree
    '''
    print(f'length of data: {len(data)}')
    print(f'entropy: {entropy(data[target])}')

    if len(data) == 0:
        return None
    elif depth >= max_depth:
        return None
    elif len(data) < min_samples_split:
        return None
    else:
        best_split, best_value = find_best_split(data, target)
        if best_split == None or best_value == None:
            return None
        else:
            split_feature, split_value = best_split, best_value
            print(f'split feature: {split_feature}')
            print(f'split value: {split_value}')
            left_data, right_data = split_data(data, split_feature, split_value)
            node = Node(data, split_feature, split_value)
            node.left = build_tree(left_data, target, max_depth, min_samples_split, depth + 1)
            node.right = build_tree(right_data,target, max_depth, min_samples_split, depth + 1)
            return node


class DecisionTree:
    '''
    Decision Tree class
    '''
    def __init__(self, max_depth : int, min_samples_split : int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.target = None
        self.tree = None

    def fit(self, data : pandas.DataFrame, target : str):
        self.tree = build_tree(data, target, self.max_depth, self.min_samples_split, 0)
    
    def predict(self, data : pandas.DataFrame):
        '''
        Function to predict the target values
        '''
    def inorder(self, node : Node):
        '''
        Function to print the tree in inorder traversal
        '''
        if node:
            self.inorder(node.left)
            node.display()
            self.inorder(node.right)
    def predict(self, data : pandas.DataFrame):
        '''
        Function to predict the target values
        '''
        current_node = self.tree
        while current_node.left != None and current_node.right != None:
            if data[current_node.split_feature] <= current_node.split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return current_node.data[self.target].mode()[0]
def main():
    dataset = pd.read_csv('cancer.csv')
    # get train and test data splits no sklearn
    shuffled_data = dataset.sample(frac=1, random_state=12)
    train_size = int(0.8 * len(shuffled_data))

    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    


    tree = build_tree(train_data, 'diagnosis(1=m, 0=b)', 10, 10, 0)
    # #save the tree
    with open('tree.pkl', 'wb') as f:
        pickle.dump(tree, f)

    #load the tree
    #get row to predict
    # with open('tree.pkl', 'rb') as f:
    #     tree = pickle.load(f)
    myTree = DecisionTree(10, 10)
    myTree.tree = tree

    myTree.target = 'diagnosis(1=m, 0=b)'

    #predict all the rows and check the accuracy
    correct = 0
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        if myTree.predict(row) == row[myTree.target]:
            correct += 1
    print(f'accuracy: {correct/len(test_data)}')
        
if __name__ == '__main__':
    main()
