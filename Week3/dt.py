## Decision Trees implementation for INF1017 - Aprendizado de Maquina
## Author: Natália Gubiani Rampon (00262502)

import numpy as np
import pandas as pd
import sys
from sklearn import tree
from sklearn.model_selection import train_test_split
from graphviz import Source
from sklearn import tree

# For numpy shapes method
COLUMNS = 1
ROWS = 0

# For sklearn train_test_split, means the random holdout will be the same across executions
NO_RANDOM = 0

NB_ARGUMENTS = 2

## Experiment parameters
TRAIN_DATA_PERCENTAGE = 0.8
GINI = "gini"
ENTROPY = "entropy"
CRITERION = ENTROPY
MAX_DEPTH_TREE = None
MIN_SAMPLES_LEAF = 1 #2#3
PRUNING_COMPLEXITY = 0 #0.01 #0.05 #0.1

## Returns partition of dataset into attributes and target outcomes
## Input:   df - data frame containing dataset
## Output:  data attributes and target outcomes
def get_partition_of_dataset(df):
    data = df.iloc[: , :df.shape[COLUMNS]-1]
    outcomes = df.iloc[: , df.shape[COLUMNS]-1]
    return data, outcomes

## Returns accuracy (percentage of correct predictions)
## Input:   test_outcomes - the correct test outcomes from the dataset
##          predict_outcomes - the corresponding predicted outcomes
## Output:  accuracy - double
def get_accuracy(test_outcomes, predict_outcomes):
    hit_count = 0
    for i in range(len(test_outcomes)):
        if test_outcomes.iloc[i] == predict_outcomes[i]:
            hit_count += 1
    return hit_count/len(test_outcomes)

if __name__ == "__main__":
    if len(sys.argv) == NB_ARGUMENTS:
        data_file = sys.argv[1]
        df = pd.read_csv(data_file, sep='\t')

        X,y = get_partition_of_dataset(df)
        train_data, test_data, train_outcomes, test_outcomes = train_test_split(X,y,
                                                                                train_size = TRAIN_DATA_PERCENTAGE,
                                                                                random_state = NO_RANDOM,
                                                                                stratify = y)
        
        decision_tree = tree.DecisionTreeClassifier(criterion = CRITERION,
                                                    max_depth = MAX_DEPTH_TREE,
                                                    min_samples_leaf = MIN_SAMPLES_LEAF,
                                                    ccp_alpha = PRUNING_COMPLEXITY,
                                                    random_state = NO_RANDOM)
        decision_tree = decision_tree.fit(train_data, train_outcomes)
        predict_outcomes = decision_tree.predict(test_data)

        accuracy = get_accuracy(test_outcomes, predict_outcomes)

        print("Accuracy for Decision Tree with\n \
        Train data percentage: {}\n \
        Criterion: {}\n \
        Tree max depth: {}\n \
        Minimum samples in leaf: {}\n \
        Pruning complexity: {}\n \
        Accuray:{}".format(TRAIN_DATA_PERCENTAGE,CRITERION,MAX_DEPTH_TREE,MIN_SAMPLES_LEAF,PRUNING_COMPLEXITY,accuracy))

        graph = Source( tree.export_graphviz(decision_tree, out_file=None, feature_names=X.columns,filled=True))
        graph.format = 'png'
        graph.render('Results/tree_c{}_md{}_msl{}_p{}'.format(CRITERION,MAX_DEPTH_TREE,MIN_SAMPLES_LEAF,PRUNING_COMPLEXITY),view=False)
    else:
        print("Wrong number of arguments ({}). Please use script as python dt.py <data_file_name>. E.g.: dt.py vote.tsv".format(len(sys.argv)))

        

