import hw3_utils
import pickle
import math
from sklearn import tree, linear_model, ensemble
import numpy as np

FOLD_PREFIX = "ecg_fold_"
FOLD_SUFFIX = ".data"

def euclidean_distance(feature_set1, feature_set2):
    return sum(list(map(lambda tup: (tup[0] - tup[1]) ** 2, zip(feature_set1, feature_set2)))) ** 0.5

def split_crosscheck_groups(dataset, num_folds):
    features_indexes_true = np.random.permutation([i for i in range(len(dataset[0])) if dataset[1][i] == True])
    features_indexes_false = np.random.permutation([i for i in range(len(dataset[0])) if dataset[1][i] == False])
    factor_true = math.ceil(len(features_indexes_true) / float(num_folds))
    factor_false = math.ceil(len(features_indexes_false) / float(num_folds))

    for i in range(num_folds):
        dataset_true_i = features_indexes_true[i*factor_true : (i+1)*factor_true]
        dataset_false_i = features_indexes_false[i*factor_false : (i+1)*factor_false]
        dataset_to_write = ([dataset[0][i] for i in dataset_true_i] + [dataset[0][i] for i in dataset_false_i],
                            [dataset[1][i] for i in dataset_true_i] + [dataset[1][i] for i in dataset_false_i])
        pickle.dump(dataset_to_write, open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX, 'wb'))

def evaluate(classifier_factory, k):
    folds = [(i, pickle.load(open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX,'rb'))) for i in range(k)]
    hits = 0
    N = sum(list(map(lambda x: len(x[1][0]), folds)))
    for test in folds:
        features, labels = [], []
        for data in folds:
            if data[0] != test[0]:
                features += data[1][0]
                labels += data[1][1]
        classifier = classifier_factory.train(features, labels)
        for sample in zip(test[1][0], test[1][1]):
            classified = classifier.classify(sample[0])
            hit = classified == sample[1]
            hits += hit
        # hits += sum([1 if classifier.classify(sample[0]) == sample[1] else 0 for sample in zip(test[1][0], test[1][1])])
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
    def __init__(self, min_samples_split = 2):
        self.min_samples_split = min_samples_split

    def train(self, data, labels):

        classifier = tree.DecisionTreeClassifier(min_samples_split=self.min_samples_split)
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


class PolynomialClassifier(hw3_utils.abstract_classifier):
    def __init__(self, classifier):
        self.classifier = classifier

    def classify(self, features):
        modified_data = np.polynomial.polynomial.polyfit(list(range(125)), features[:125], 4)
        return self.classifier.classify(modified_data)


class PolynomialFactory(hw3_utils.abstract_classifier_factory):

    def train(self, data, labels):

        modified_data = [np.polynomial.polynomial.polyfit(list(range(125)), data[i][:125], 4) for i in range(len(data))]
        factory = knn_factory(1)

        return PolynomialClassifier(factory.train(modified_data, labels))


class BestKClassifier(hw3_utils.abstract_classifier):
    def __init__(self, classifier, selector):
        self.classifier = classifier
        self.selector = selector

    def classify(self, features):
        modified_data = self.selector.transform([features])
        return self.classifier.classify(modified_data[0])


class BestKFactory(hw3_utils.abstract_classifier_factory):
    def __init__(self, selector):
        self.selector = selector

    def train(self, data, labels):

        modified_data = self.selector.transform(data)
        factory = ID3Factory()

        return BestKClassifier(factory.train(modified_data, labels), self.selector)

class RangedClassifier(hw3_utils.abstract_classifier):
    def __init__(self, classifier, feature_range):
        self.classifier = classifier
        self.feature_range = feature_range

    def classify(self, features):
        modified_data = features[self.feature_range[0]:self.feature_range[1]]
        return self.classifier.classify(modified_data)


class RangedFactory(hw3_utils.abstract_classifier_factory):
    def __init__(self, feature_range):
        self.feature_range = feature_range

    def train(self, data, labels):

        modified_data = [sample[self.feature_range[0]:self.feature_range[1]] for sample in data]
        factory = knn_factory(3)

        return RangedClassifier(factory.train(modified_data, labels), self.feature_range)


class RandomForestClassifier(hw3_utils.abstract_classifier):

    def __init__(self, tree_classifier):
        self.tree_classifier = tree_classifier

    def classify(self, features):
        return self.tree_classifier.predict([features])

class RandomForestFactory(hw3_utils.abstract_classifier_factory):
    def __init__(self, n_estimators = 10, min_samples_split = 2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split

    def train(self, data, labels):

        classifier = ensemble.RandomForestClassifier(n_estimators=self.n_estimators,
                                                     min_samples_split=self.min_samples_split)
        classifier.fit(data, labels)
        return RandomForestClassifier(classifier)


class contestClassifier(hw3_utils.abstract_classifier):

    def __init__(self, classifiers, weights):
        self.classifiers = classifiers
        self.weights = weights

    def classify(self, features):
        return sum(list(map(lambda x: x[0] * (1 if x[1].classify(features) else 0),
                            zip(self.weights, self.classifiers)))) > 0.5

class contestFactory(hw3_utils.abstract_classifier_factory):

    def __init__(self, factories, selectors, weights):
        self.factories = factories
        self.selectors = selectors
        self.weights = weights

    def train(self, data, labels):

        return contestClassifier([BestKClassifier(self.factories[i].train(self.selectors[i].transform(data), labels),
                                                  self.selectors[i]) for i in range(len(self.factories))], self.weights)