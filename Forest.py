from DecisionTree import DecisionTree
import numpy as np
import pandas as pd
import random

def bootstrap(data, n):
    '''
    Create a bootstrap sample from the data
    '''
    return data.sample(n, replace=True)


class RandomForest:

    def __init__(self, numTrees, maxDepth, minSamples): 
       self.numTrees = numTrees
       self.maxDepth = maxDepth
       self.minSamples = minSamples
       self.trees = []

    
    def fit(self, data, target):
        for i in range(self.numTrees):
            tree = DecisionTree(self.maxDepth, self.minSamples)
            bootstrap_sample = bootstrap(data, len(data))
            tree.fit(bootstrap_sample, target)
            self.trees.append(tree)



def main():
    data = pd.read_csv('cancer.csv')
    bootstrap_data = bootstrap(data, len(data))
    shuffled_data = data.sample(frac=1, random_state=12)
    train_size = int(0.8 * len(shuffled_data))

    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    maxDepthIn = int(input('Enter the max depth: '))
    minSamplesIn = int(input('Enter the min samples: '))

    tree = DecisionTree(maxDepthIn, minSamplesIn)
    tree.fit(train_data, 'diagnosis(1=m, 0=b)')

    tree.target = 'diagnosis(1=m, 0=b)'


    #predict all the rows and check the accuracy
    correct = 0
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        if tree.predict(row) == row[tree.target]:
            correct += 1
    print(f'accuracy: {correct/len(test_data)}')
 

if __name__ == '__main__':
    main()
