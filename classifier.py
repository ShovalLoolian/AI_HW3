import hw3_utils
import pickle
import math

FOLD_PREFIX = "ecg_fold_"
FOLD_SUFFIX = ".data"

def euclidean_distance(feature_set1, feature_set2):
    return sum(list(map(lambda tup: (tup[0] - tup[1]) ** 2, zip(feature_set1, feature_set2)))) ** 0.5

def split_crosscheck_groups(dataset, num_folds):
    features_indexes_true = [i for i in range(len(dataset[0])) if dataset[1][i] == True]
    features_indexes_false = [i for i in range(len(dataset[0])) if dataset[1][i] == False]
    factor_true = math.ceil(len(features_indexes_true) / float(num_folds))
    factor_false = math.ceil(len(features_indexes_false) / float(num_folds))

    for i in range(num_folds):
        dataset_true_i = features_indexes_true[i*factor_true : (i+1)*factor_true]
        dataset_false_i = features_indexes_false[i*factor_false : (i+1)*factor_false]
        dataset_to_write = ([dataset[0][i] for i in dataset_true_i] + [dataset[0][i] for i in dataset_false_i],
                            [dataset[1][i] for i in dataset_true_i] + [dataset[1][i] for i in dataset_false_i])
        pickle.dump(dataset_to_write, open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX, 'wb'))

def evaluate(classifier_factory, k):
    folds = [pickle.load(open(FOLD_PREFIX + str(i+1) + FOLD_SUFFIX,'rb')) for i in range(k)]
    hits = 0
    N = sum(list(map(lambda x: len(x[0]), folds)))
    for test in folds:
        features, labels = [], []
        for data in folds:
            if data != test:
                features += [data[0]]
                labels += [data[1]]
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
        dataset = list(map(lambda x: x[1], sorted(distances, key=lambda x: x[0])[:self.k]))
        trues = sum(list(map(lambda x: 1 if self.labels[x] == True else 0, dataset)))
        return trues >= self.k / 2


class knn_factory(hw3_utils.abstract_classifier_factory):

    def __init__(self, k):
        self.k = k

    def train(self, data, labels):

        return knn_classifier(data, labels, self.k)
