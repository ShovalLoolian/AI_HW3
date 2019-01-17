import hw3_utils
import pickle
import math
import numpy as np
from sklearn import tree, linear_model

FOLD_PREFIX = "ecg_fold_"
FOLD_SUFFIX = ".data"

def euclidean_distance(feature_set1, feature_set2):
    return sum(list(map(lambda tup: (tup[0] - tup[1]) ** 2, zip(feature_set1, feature_set2)))) ** 0.5

def split_crosscheck_groups(dataset, num_folds):
    features_indexes_true = set([i for i in range(len(dataset[0])) if dataset[1][i] == True])
    features_indexes_false = set([i for i in range(len(dataset[0])) if dataset[1][i] == False])
    size_true = math.ceil(len(features_indexes_true) / float(num_folds))
    size_false = math.ceil(len(features_indexes_false) / float(num_folds))
    for i in range(num_folds):
        size_true, size_false = min(size_true, len(features_indexes_true)), min(size_false, len(features_indexes_false))
        dataset_true_i = np.random.choice(list(features_indexes_true), size_true, replace=False)
        features_indexes_true = features_indexes_true - set(dataset_true_i)

        dataset_false_i = np.random.choice(list(features_indexes_false), size_false, replace=False)
        features_indexes_false = features_indexes_false - set(dataset_false_i)

        dataset_to_write = ([dataset[0][i] for i in dataset_true_i] + [dataset[0][i] for i in dataset_false_i],
                            [dataset[1][i] for i in dataset_true_i] + [dataset[1][i] for i in dataset_false_i])
        print("size of trues " + str(i) + " is " + str(len(dataset_true_i)))
        print("size of falses " + str(i) + " is " + str(len(dataset_false_i)))
        print("total size " + str(i) + " is " + str(len(dataset_to_write[0])) + "\n")
        pickle.dump(dataset_to_write, open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX, 'wb'))

def evaluate(classifier_factory, k):
    folds = [pickle.load(open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX,'rb')) for i in range(k)]
    hits = 0
    N = sum(list(map(lambda x: len(x[0]), folds)))
    for test in folds:
        features, labels = [], []
        for data in folds:
            if data != test:
                features += data[0]
                labels += data[1]
        classifier = classifier_factory.train(features, labels)
        hits += sum([1 if classifier.classify(sample[0]) == sample[1] else 0 for sample in zip(test[0], test[1])])
    return hits / N, 1 - (hits / N)



class knn_classifier(hw3_utils.abstract_classifier):

    def __init__(self, data, labels, k):
        self.data = data
        self.labels = labels
        self.k = k

    def classify(self, features):
        distances = [(euclidean_distance(self.data[i], features), i) for i in range(len(self.data))]
        dataset = list(map(lambda x: x[1], sorted(distances, key=lambda x: x[0])))[:self.k]
        trues = sum(list(map(lambda x: 1 if self.labels[x] == True else 0, dataset)))
        return trues >= self.k / 2


class knn_factory(hw3_utils.abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):

        return knn_classifier(data, labels, self.k)


class ID3Classifier(hw3_utils.abstract_classifier):

    def __init__(self, tree_classifier):
        self.tree_classifier = tree_classifier

    def classify(self, features):
        return self.tree_classifier.predict([features])


class ID3Factory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):

        classifier = tree.DecisionTreeClassifier()
        classifier.fit(data, labels)
        return ID3Classifier(classifier)


class PerceptronClassifier(hw3_utils.abstract_classifier):

    def __init__(self, linear_model):
        self.linear_model = linear_model

    def classify(self, features):
        return self.linear_model.predict([features])

class PerceptronFactory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):

        classifier = linear_model.Perceptron()
        classifier.fit(data, labels)
        return PerceptronClassifier(classifier)


class contestClassifier(hw3_utils.abstract_classifier):

    def __init__(self, classifier_a, classifier_b, classifier_c):
        self.a = classifier_a
        self.b = classifier_b
        self.c = classifier_c

    def classify(self, features):
        return sum([1 if self.a.classify(features) else 0,
                    1 if self.b.classify(features) else 0,
                    1 if self.c.classify(features) else 0]) > 1

class contestFactory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):

        a_factory = knn_factory(1)
        b_factory = ID3Factory()
        c_factory = PerceptronFactory()

        return contestClassifier(a_factory.train(data, labels),
                                 b_factory.train(data, labels), c_factory.train(data, labels))