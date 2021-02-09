## K-Nearest-Neighbours implementation for INF1017 - Aprendizado de Maquina
## Author: Natália Gubiani Rampon (00262502)

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

POWER_OF_2 = 2
POSITIVE = 1
NEGATIVE = 0
NB_ARGUMENTS = 4
COLUMNS = 1
ROWS = 0

## Gives the Euclidean distance between two points in any dimensions
## Input: p1,p2 - vectors
## Output: Euclidean distance - integer
def euclidean_distance(p1,p2):
    return np.sqrt(np.sum(np.power(p2-p1,POWER_OF_2)))

## Returns the order of indexes that would sort an array
## Input: vector to be sorted - vector
## Output: indexes of elements in sorted order - vector
def get_sorted_indexes(vector):
    return np.argsort(vector)

## Gives the K nearest neighbours for a given point
## Input:   point - the test point
##          data - the training data
##          k - hyperparameter k
## Output:  indexes of k nearest neighbours - vector
def get_k_nearest_neighbours(point, data, k):
    distances = np.zeros(data.shape[ROWS])
    for i in range(len(distances)):
        distance = euclidean_distance(point,data.iloc[i])
        distances[i] = distance
    return get_sorted_indexes(distances)[:k]

## Evaluates the test point applying the knn algorithm using provided data
## Input:   point - the test point
##          data - the training data
##          outcomes - the outcome labeling for the training data
##          k - hyperparameter k
## Output:  predicted value for the test point
def evaluate_test_point(point,data,outcomes,k):
    count_positive = 0
    k_indexes = get_k_nearest_neighbours(point,data,k)
    for outcome in outcomes[k_indexes]:
        if (outcome == POSITIVE):
            count_positive += 1
    return POSITIVE if count_positive > k/2 else NEGATIVE

## Returns partition of dataset into attributes and target outcomes
## Input:   file - file name containing dataset - string
## Output:  data attributes and target outcomes
def get_partition_of_dataset(file):
    df = pd.read_csv(file)
    data = df.iloc[: , :df.shape[COLUMNS]-1]
    outcomes = df.iloc[: , df.shape[COLUMNS]-1]
    return data, outcomes

## Returns the max interval between all attributes
## Input:   data - the data frame to be analysed
## Output:  double - max interval between any attribute
def calculate_max_interval(data):
    interval = []
    for column in data.columns:
        interval.append(data[column].max() - data[column].min())
        print(data[column].max())
        print(column)
    print(data['28'].to_string())
    return max(interval)

## Produce results for the model (print accuracy values and saved graph)
## Input:   k_list : list with k values
##          accuracy_list : list with accuracy values
##          file_name : file name for the graph picture
## Output:  none
produce_results(k_list, accuracy_list, file_name = 'result.png'):
    print("K values: {}".format(k_list))
    print("Accuracy values: {}".format(accuracy_list))
    
    fig = plt.figure()
    fig.suptitle('KNN Evaluation')
    plt.plot(k_list, accuracy_list)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    fig.savefig(file_name)

if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        k_list = list(map(int, sys.argv[3].strip('[]').split(',')))
        print("Executing K-Nearest-Neighbours algorithm\nTrain file: {}\nTest file: {}\nk = {}".format(train_file, test_file, k_list))
        
        train_data,train_outcomes = get_partition_of_dataset(train_file)
        test_data,test_outcomes = get_partition_of_dataset(test_file)

        print("Max interval train data: {}".format(calculate_max_interval(train_data)))
        exit()
        print("Max interval test data: {}".format(calculate_max_interval(test_data)))

        accuracy_list = []
        for k in k_list:
            hit_count = 0

            for i, test_datum in test_data.iterrows():
                prediction = evaluate_test_point(test_datum, train_data, train_outcomes, k)
                if prediction == test_outcomes.iloc[i]:
                    hit_count += 1
            accuracy = hit_count/test_outcomes.shape[ROWS]
            accuracy_list.append(accuracy)
        
        produce_results(k_list, accuracy_list)
    else:
        print("Wrong number of arguments ({}). Please use script as python knn.py <train_file_name> <test_file_name> <k-list>. E.g.: knn.py train.csv test.csv [1,3,5,7]".format(len(sys.argv)))