import hw3_utils, classifier
import os
import pickle
import csv, sklearn, math
import numpy

from matplotlib import pyplot as plt


def removeDuplications(train_features, train_labels):
    duplicates = set()
    for i in range(len(train_features)):
        dup_for_i = set()
        for j in range(i + 1, len(train_features)):
            if all(list(map(lambda x: x[0] == x[1], zip(list(train_features[i]), list(train_features[j]))))):
                dup_for_i.add(j)
        if len(dup_for_i) > 0:
            if not all(list(map(lambda x: train_labels[i] == train_labels[x], dup_for_i))):
                dup_for_i.add(i)
            duplicates = duplicates | dup_for_i
    modified_data = [train_features[i] for i in range(len(train_features)) if i not in duplicates]
    modified_labels = [train_labels[i] for i in range(len(train_features)) if i not in duplicates]
    return modified_data, modified_labels


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

    # with open("ID3WIDE.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in range(10,181, 20): #range(154,167,2)
    #         result = 0
    #         selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, n).fit(train_features, train_labels) # sklearn.feature_selection.mutual_info_classif,
    #         factory = classifier.BestKFactory(selector)
    #         for i in range(10):
    #             result += classifier.evaluate(factory, 2)[0]
    #         writer.writerow([n, result / 10, 1 - (result / 10)])

    # with open("experimentsKNN3ranged.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in range(40, 181, 40):
    #         factory = classifier.RangedFactory((n - 40, n))
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

    # with open("experimentsRandomforest.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in range(10,101,10):
    #         result = 0
    #         factory = classifier.RandomForestFactory(n)
    #         for i in range(5):
    #             result += classifier.evaluate(factory, 2)[0]
    #         writer.writerow([n, result / 5, 1 - (result / 5)])

    # contest ideas

    # knn_1_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 158).fit(train_features, train_labels)
    # knn_3_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 106).fit(train_features, train_labels)
    # # ID3_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 156).fit(train_features, train_labels)
    # Random_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 156).fit(train_features, train_labels)
    # # default_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 187).fit(train_features, train_labels)
    #
    # knn_1_factory = classifier.knn_factory(1)
    # knn_3_factory = classifier.knn_factory(3)
    # # ID3_factory = classifier.ID3Factory(7)
    # random_forest_factory_7 = classifier.RandomForestFactory(40, 7)
    # random_forest_factory_2 = classifier.RandomForestFactory(40, 2)
    #
    # # three best classifiers
    # factory_a = classifier.contestFactory([knn_1_factory, knn_3_factory, random_forest_factory_2],
    #                                       [knn_1_selector, knn_3_selector, Random_selector],
    #                                       [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    # # avoid overfitting- not using knn with k=1 or random forest with min split 2
    # # more weight for random forest
    # factory_b = classifier.contestFactory([knn_3_factory, random_forest_factory_7, random_forest_factory_7],
    #                                       [knn_3_selector, Random_selector, Random_selector],
    #                                       [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    #
    # # avoid overfitting- k =3 has more weight then k = 1
    # # more weight for knn
    # factory_c = classifier.contestFactory([knn_3_factory, knn_1_factory, random_forest_factory_7, random_forest_factory_7],
    #                                       [knn_3_selector, knn_1_selector, Random_selector, Random_selector],
    #                                       [0.4, 0.2, 0.2, 0.2])
    #
    #
    # with open("experimentsContest222.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #
    #     result = classifier.evaluate(factory_a, 2)
    #     writer.writerow([1, result[0], result[1]])
    #
    #     result = classifier.evaluate(factory_b, 2)
    #     writer.writerow([2, result[0], result[1]])
    #
    #     result = classifier.evaluate(factory_c, 2)
    #     writer.writerow([6, result[0], result[1]])


    # submission

    train_features, train_labels, test_features = hw3_utils.load_data()

    modified_data, modified_labels = removeDuplications(train_features, train_labels)

    knn_1_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 158).fit(train_features, train_labels)
    knn_3_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 106).fit(train_features, train_labels)
    Random_selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, 156).fit(train_features, train_labels)

    knn_1_factory = classifier.knn_factory(1)
    knn_3_factory = classifier.knn_factory(3)
    random_forest_factory_7 = classifier.RandomForestFactory(40, 7)

    final_factory = classifier.contestFactory(
        [knn_3_factory, knn_1_factory, random_forest_factory_7, random_forest_factory_7],
        [knn_3_selector, knn_1_selector, Random_selector, Random_selector],
        [0.4, 0.2, 0.2, 0.2])

    final_classifier = final_factory.train(modified_data, modified_labels)

    our_labels = [1 if final_classifier.classify(features) else 0 for features in test_features]

    hw3_utils.write_prediction(our_labels)


if __name__ == '__main__':
    main()