"""
This file includes utility functions for HW1. 

Do not modify code in this file. 
"""
import csv
import os
import random
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 
from typing import List, Set, Optional, Union, Dict 


class Dataset:
    """
    Represents a set of training/testing data. self.train is a list of
    Examples, as is self.dev and self.test.
    """

    def __init__(self, name: str = None, include_test: bool = False):
        self.name = name
        self.splits = OrderedDict()
        self.train = []
        self.dev = []

        self.splits['train'] = self.train
        self.splits['dev'] = self.dev

        if include_test:
            self.test = []
            self.splits["test"] = self.test

    def shuffle(self) -> None:
        for split_name in self.splits:
            random.shuffle(self.splits[split_name])


class Example:
    """
    Represents a document (list of words) with a corresponding label.
    """

    def __init__(self, words: List[str], label: Optional[int]):
        self.words = words
        self.label = label

def segment_words(s: str) -> List[str]:
    """
    Splits lines into words on whitespace.

    Args:
        s (str): The line to segment.

    Returns:
        list[str]: List of segmented words in the line.
    """
    return s.split()


def read_file(file_name: str,
              encoding: str = "utf8", mode: str = "word") -> List[str]:
    """
    Reads lines or words from a file.

    Args:
        file_name (str): The filename to read from.
        encoding (str): The encoding to use for reading the file.
        mode (str): How to extract the file contents. If "word", outputs a list
        of words (as strings). If "line", outputs a list of lines (as strings).

    Returns:
        list[str]: List of segmented words or lines in the file.
    """
    outputs = []
    with open(file_name, encoding=encoding) as f:
        for line in f:
            if mode == "word":
                outputs.extend(segment_words(line))
            elif mode == "line":
                outputs.append(line)
            else:
                raise ValueError("Invalid mode: {}".format(mode))
    return outputs


def calculate_accuracy(data: List[Example], classifier) -> float:
    """
    Calculates the classifier's accuracy on a provided dataset.

    Args:
        data (list[Example]): List of examples to evaluate on.
        classifier (Classifier): Classifier to compute accuracy for.

    Returns:
        float: Accuracy of the given classifier on the given dataset.
    """

    if len(data) == 0:
        return 0.0

    predictions = classifier.predict(data)

    correct = 0.0
    for prediction, example in zip(predictions, data):
        if example.label == prediction:
            correct += 1.0
    return correct / len(data)


def load_data(data_dir: str,
              include_test: bool = False,
              dataset_name: str = None) -> Dataset:
    """
    Loads data into a Dataset object.

    Args:
        data_dir (str): Path to the directory containing the data.
        include_test (bool): Whether to load test data (this will only be
                    available to the teaching staff in the autograder).
        dataset_name (str): Optional, name to give to the created dataset.

    Returns:
        Dataset: Dataset containing the loaded data.
    """
    dataset = Dataset(name=dataset_name, include_test=include_test)
    for split_name in dataset.splits:
        with open(os.path.join(data_dir, split_name + ".csv"),
                  newline='', mode="r", encoding="utf8") as infile:
            reader = csv.DictReader(infile, delimiter="|")
            for row in reader:
                text = row["Text"]
                label = int(row["Label"])
                example = Example(segment_words(text.rstrip('\n')),
                                  label)
                dataset.splits[split_name].append(example)

    dataset.shuffle()
    return dataset


def evaluate(classifier, dataset: Dataset,
             limit_training_set: bool = False) -> None:
    """
    Evaluate a classifier on the training and dev sets, printing the accuracy.

    Args:
        classifier (Classifier): Classifier to evaluate on the dataset.
        dataset (Dataset): Dataset to train and evaluate on.
        limit_training_set (bool): If true, truncate training set to
        only 25% of its full size.
    """
    training_set = dataset.train[:int(0.25 * len(dataset.train))] \
        if limit_training_set else dataset.train

    classifier.train(training_set)

    for split_name in dataset.splits:
        accuracy = calculate_accuracy(training_set
                                      if split_name == "train"
                                      else dataset.splits[split_name],
                                      classifier)
        print('Accuracy ({}): {}'.format(split_name, accuracy))

def examples_to_lists(examples): 
    x_list = []
    Y = []
    for example in examples:
        x_list.append(" ".join(example.words))
        Y.append(example.label)
    return x_list, Y

def transform_examples_to_arrays(train_examples: List[Example], 
                                 dev_examples: List[Example], 
                                 test_examples=None) -> Dict: 
    """
    Takes as input a list of examples

    Returns X, a numpy array 
    with rows as the number of examples and columns and counts of words 

    If test_examples is not None: also converts the test data 
    """
    data = {}
    
    # covert datatypes for convenience 
    x_train_list, Y_train = examples_to_lists(train_examples)
    x_dev_list, Y_dev = examples_to_lists(dev_examples)

    if test_examples is not None: 
        x_test_list, Y_test = examples_to_lists(test_examples)
    
    #use the training data to create the vocab 
    vectorizer = CountVectorizer(min_df=20, #only look at words that occur in at least 20 docs
                                stop_words='english', # remove english stop words
                                max_features=1000, #only select the top 1000 features 
                                ) 
    data['X_train'] = vectorizer.fit_transform(x_train_list).toarray()
    data['Y_train'] = np.array(Y_train)
    
    data['x_columns_as_words'] = vectorizer.get_feature_names_out()
    
    # use the same vectorizer for dev sets 
    data['X_dev'] = vectorizer.transform(x_dev_list).toarray()
    data['Y_dev'] = np.array(Y_dev)
    
    assert data['X_train'].shape[0] == data['Y_train'].shape[0]
    assert data['X_dev'].shape[0] == data['Y_dev'].shape[0]
    assert data['X_train'].shape[1] == data['X_dev'].shape[1] == data['x_columns_as_words'].shape[0]
        
    if test_examples is not None: 
        data['X_test'] = vectorizer.transform(x_test_list).toarray()
        data['Y_test'] = np.array(Y_test)
        
        assert data['X_test'].shape[0] == data['Y_test'].shape[0]
        assert data['X_train'].shape[1] == data['X_test'].shape[1] == data['x_columns_as_words'].shape[0]

    return data 
    