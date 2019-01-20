import hw3_utils, classifier
import os
import pickle
import csv, sklearn, math
import numpy

from matplotlib import pyplot as plt



# TODO: remove
def load_data_try(path=r'data/data.pickle'):
    '''
    return the dataset that will be used in HW 3
    prameters:
    :param path: the path of the csv data file (default value is data/ecg_examples.data)

    :returns: a tuple train_features, train_labels ,test_features
    features - a numpy matrix where  the ith raw is the feature vector of patient i.
    '''
    with open(path,'rb') as f:
        train_features, train_labels = pickle.load(f)
    return train_features, train_labels


def main():
    # train_features, train_labels, test_features = hw3_utils.load_data()
    # classifier.split_crosscheck_groups((train_features, train_labels), 2)

    # q. 5

    # with open("experiments6.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for k in [1,3,5,7,13]:
    #         factory = classifier.knn_factory(k)
    #         result = classifier.evaluate(factory, 2)
    #         writer.writerow([k, result[0], result[1]])

    # q. 7

    # with open("experiments12.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     factory = classifier.ID3Factory()
    #     result = classifier.evaluate(factory, 2)
    #     writer.writerow([1, result[0], result[1]])
    #
    #     factory = classifier.PerceptronFactory()
    #     result = classifier.evaluate(factory, 2)
    #     writer.writerow([2, result[0], result[1]])

    # part c

    # plt.figure()
    #
    # sick = [train_features[i] for i in range(22) if train_labels[i] == False]
    # healthy = [train_features[i] for i in range(4) if train_labels[i] == True]
    # for sample in sick:
    #     plt.plot(list(range(187)), sample, '-k')
    #
    # for sample in healthy:
    #     plt.plot(list(range(187)), sample, '-r')
    #
    # plt.show()

    # ---polynomial---

    # with open("experiments1.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     factory = classifier.PolynomialFactory()
    #     result = classifier.evaluate(factory, 2)
    #     writer.writerow([1, result[0], result[1]])

    # ---feature selection---

    # with open("experiments3.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in [5,10,20,100,150,160]:
    #         selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, n).fit(train_features, train_labels) # sklearn.feature_selection.mutual_info_classif,
    #         factory = classifier.BestKFactory(selector)
    #         result = classifier.evaluate(factory, 2)
    #         writer.writerow([n, result[0], result[1]])


    # ---remove duplications---
    # duplicates = set()
    # for i in range(len(train_features)):
    #     dup_for_i = set()
    #     for j in range(i+1, len(train_features)):
    #         if all(list(map(lambda x: x[0] == x[1],zip(list(train_features[i]), list(train_features[j]))))):
    #             dup_for_i.add(j)
    #     if len(dup_for_i) > 0:
    #         if not all(list(map(lambda x: train_labels[i] == train_labels[x], dup_for_i))):
    #             dup_for_i.add(i)
    #         duplicates = duplicates | dup_for_i
    # modified_data = [train_features[i] for i in range(len(train_features)) if i not in duplicates]
    # modified_labels = [train_labels[i] for i in range(len(train_features)) if i not in duplicates]

    # ---min_samples_split---

    # with open("experiments3.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in range(2,30):
    #         result = 0
    #         factory = classifier.ID3Factory(n)
    #         for i in range(10):
    #             result += classifier.evaluate(factory, 2)[0]
    #         writer.writerow([n, result / 10, 1 - (result / 10)])

    # ---random forest---

    with open("experimentsRandomforest.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for n in range(10,101,10):
            result = 0
            factory = classifier.RandomForestFactory(n)
            for i in range(5):
                result += classifier.evaluate(factory, 2)[0]
            writer.writerow([n, result / 5, 1 - (result / 5)])




if __name__ == '__main__':
    main()