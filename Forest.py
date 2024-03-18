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

    def __init__(self, numTrees, maxDepth, minSamples, maxFeatures=None): 
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.minSamples = minSamples
        self.maxFeatures = maxFeatures
        self.trees = []

    
    def fit(self, data, target):
        '''
        Fit the random forest to the data
        '''
        columns = data.columns
        # remove the target column
        print(columns)
        for i in range(self.numTrees):
            print(f'Fitting tree {i}')
            tree = DecisionTree(self.maxDepth, self.minSamples)
            bootstrap_sample = bootstrap(data, len(data))
            selected_sample = bootstrap_sample

            # sample the columns
            if self.maxFeatures:
                sample_columns = random.sample(list(columns), self.maxFeatures)
                print(f'Selected columns: {sample_columns}')
                # use loc to select the columns
                selected_sample = selected_sample.loc[:, sample_columns]
            # add the target back to the sample
            selected_sample[target] = bootstrap_sample[target]

            tree.fit(selected_sample, target)
            tree.target = target
            self.trees.append(tree)

    def predict(self, data): 
        '''
        Predict the target of the data point
        '''
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(data)
            predictions.append(prediction)

        # return the most common prediction
        return max(set(predictions), key=predictions.count)

def main():
    data = pd.read_csv('cancer.csv')
    #drop the id column
    data.reset_index(drop=True, inplace=True)
    print(data.head())

    shuffled_data = data.sample(frac=1, random_state=72)
    train_size = int(0.8 * len(shuffled_data))

    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    print('----------------')
    print(train_data.head())


    maxDepthIn = int(input('Enter the max depth: '))
    minSamplesIn = int(input('Enter the min samples: '))
    numTreesIn = int(input('Enter the number of trees: '))
    maxFeaturesIn = int(input('Enter the max features: '))

    forest = RandomForest(numTreesIn, maxDepthIn, minSamplesIn, maxFeaturesIn)
    
    forest.fit(train_data, 'diagnosis(1=m, 0=b)')
    #predict all the rows and check the accuracy
    correct = 0
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        if forest.predict(row) == row['diagnosis(1=m, 0=b)']:
            correct += 1
    print(f'accuracy: {correct/len(test_data)}')
 

if __name__ == '__main__':
    main()
