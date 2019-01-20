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

    # ---create the polynomial
    # polynomial_features = []
    # for i in range(len(train_features)):
    # polynomial_features += [numpy.polynomial.polynomial.polyfit(list(range(187)), train_features[i], 4)]
    # classifier.split_crosscheck_groups((polynomial_features, train_labels), 2)

    # ---knn
    # with open("experiments1.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for k in [1,3]:
    #         factory = classifier.knn_factory(k)
    #         result = classifier.evaluate(factory, 2)
    #         writer.writerow([k, result[0], result[1]])

    # ---ID3 and Perceptron
    # with open("experiments2.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     factory = classifier.ID3Factory()
    #     result = classifier.evaluate(factory, 2)
    #     writer.writerow([1, result[0], result[1]])
    #
    #     factory = classifier.PerceptronFactory()
    #     result = classifier.evaluate(factory, 2)
    #     writer.writerow([2, result[0], result[1]])

    # ---feature selection---

    # with open("experiments3.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for n in [5,10,20,100,150,160]:
    #         selector = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.f_classif, n).fit(train_features, train_labels) # sklearn.feature_selection.mutual_info_classif,
    #         factory = classifier.BestKFactory(selector)
    #         result = classifier.evaluate(factory, 2)
    #         writer.writerow([n, result[0], result[1]])


    # test for duplications:
    train_features = [[1,2,3],[1,2,3],[4,5,6],[1,2,3]]
    train_labels = [True,False,True, True]




    # remove duplications
    duplicates = set()
    for i in range(len(train_features)):
        dup_for_i = set()
        for j in range(i+1, len(train_features)):
            if all(list(map(lambda x: x[0] == x[1],zip(list(train_features[i]), list(train_features[j]))))):
                dup_for_i.add(j)
        if len(dup_for_i) > 0:
            if not all(list(map(lambda x: train_labels[i] == train_labels[x], dup_for_i))):
                dup_for_i.add(i)
            duplicates = duplicates | dup_for_i
    modified_data = [train_features[i] for i in range(len(train_features)) if i not in duplicates]
    modified_labels = [train_labels[i] for i in range(len(train_features)) if i not in duplicates]
    print("done")





# train_featurs_final = train_features[200:]
    # train_labels_final = train_labels[200:]
    # test_final = train_features[:200]
    # test_labels_final = train_labels[:200]
    # sick = [train_features[i] for i in range(50) if train_labels[i] == False]
    # healthy = [train_features[i] for i in range(50) if train_labels[i] == True]

    # features_indexes_true = [i for i in range(len(train_features)) if train_labels[i] == True]
    # features_indexes_false = [i for i in range(len(train_features)) if train_labels[i] == False]
    # factor_true = math.ceil(len(features_indexes_true) / float(5))
    # factor_false = math.ceil(len(features_indexes_false) / float(5))
    #
    # train_true = features_indexes_true[: 4 * factor_true]
    # train_false = features_indexes_false[: 4 * factor_false]
    # test_true = features_indexes_true[4 * factor_true:]
    # test_false = features_indexes_false[4 * factor_false:]
    #
    # train_featurs_final = [train_features[i] for i in train_true] + [train_features[i] for i in train_false]
    # train_labels_final = [train_labels[i] for i in train_true] + [train_labels[i] for i in train_false]
    #
    # test_featurs_final = [train_features[i] for i in test_true] + [train_features[i] for i in test_false]
    # test_labels_final = [train_labels[i] for i in test_true] + [train_labels[i] for i in test_false]



    # plt.figure(1)
    #
    # # plt.plot([1], list(map(lambda x: x / 10, vec_org[1:2])),'o', label="OriginalReflexAgent")
    # for sample in sick:
    #     plt.plot(list(range(187)), sample, '-k')
    #
    # # plt.plot([1], list(map(lambda x: x / 10, vec_org[1:2])),'o', label="OriginalReflexAgent")
    # for sample in healthy:
    #     plt.plot(list(range(187)), sample, '-r')
    #
    # plt.show()

    # knn = classifier.knn_factory(3).train(train_features, train_labels)
    # print(knn.classify(test_features[59]))

# create ecg_fold files:
#     classifier.split_crosscheck_groups((train_featurs_final, train_labels_final), 8)
    # train_features, train_labels = load_data_try('ecg_fold_1.data')
    # train_features2, train_labels2 = load_data_try('ecg_fold_2.data')
    # # train_features3, train_labels3 = load_data_try('ecg_fold_3.data')
    # print(sum([1 for label in train_labels if label]))
    # print(sum([1 for label in train_labels2 if label]))
    # print(sum([1 for label in train_labels if not label]))
    # print(sum([1 for label in train_labels2 if not label]))


    # data = [[1,1,1],[2,2,2],[100,3,3]]
    # lables = [True, False, False]
    # features = [4,4,4]
    # distances = [(classifier.euclidean_distance(data[i], features), i) for i in range(len(data))]
    # dataset = list(map(lambda x: x[1], sorted(distances, key=lambda x: x[0])[:2]))
    # trues = sum(list(map(lambda x: 1 if lables[x] == True else 0, dataset)))
    # print(list(map(lambda x: 1 if lables[x] == True else 0, dataset)))
    # print(trues >= 2 / 2)


    # q. 5


    #
    #         classifies = [ID3_classifier.classify(sample) for sample in test_final]
    #
    #         false_positive = [i if classifies[i] != test_labels_final[i] and test_labels_final[i] == True else 0 for i in range(len(test_final))]
    #         false_negative = [i if classifies[i] != test_labels_final[i] and test_labels_final[i] == False else 0 for i in range(len(test_final))]
    #
    #         true_positive = [i if classifies[i] == test_labels_final[i] and test_labels_final[i] == True else 0 for i in range(len(test_final))]
    #         true_negative = [i if classifies[i] == test_labels_final[i] and test_labels_final[i] == False else 0 for i in range(len(test_final))]
# with open("experiments6.csv", 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file)
#     for k in [1]:#,3,5,7,9,11,13]:
#         factory = classifier.ID3Factory()
#         ID3_classifier = factory.train(train_featurs_final, train_labels_final)
    #
    #
    #         # result = (res/len(test_final), 1 - (res/len(test_final)))
    #         # writer.writerow([k, result[0], result[1]])
    #
    #         plt.figure(1)
    #
    #         for i in false_positive:
    #             plt.plot(list(range(187)), test_final[i], '-k')
    #
    #         for i in false_negative:
    #             plt.plot(list(range(187)), test_final[i], '-r')
    #
    #         for i in true_positive[:5]:
    #             plt.plot(list(range(187)), test_final[i], '-b')
    #
    #         for i in true_negative[:5]:
    #             plt.plot(list(range(187)), test_final[i], '-g')
    #         plt.legend()
    #         plt.show()

    # q. 7

    # with open("experiments12.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     factory = classifier.contestFactory()
    #     contest_classifier = factory.train(train_featurs_final, train_labels_final)
    #     res = sum([1 if contest_classifier.classify(test_featurs_final[i]) == test_labels_final[i] else 0 for i in
    #                range(len(test_featurs_final))])
    #     result = (res/len(test_featurs_final), 1 - (res/len(test_featurs_final)))
    #     writer.writerow([1, result[0], result[1]])

    #     Perceptron_factory = classifier.PerceptronFactory()
    #     Perceptron_classifier = Perceptron_factory.train(train_featurs_final, train_labels_final)
    #     res = sum([1 if Perceptron_classifier.classify(test_final[i]) == test_labels_final[i] else 0 for i in
    #                range(len(test_final))])
    #     result = (res / len(test_final), 1 - (res / len(test_final)))
    #     writer.writerow([2, result[0], result[1]])
    # #
        # Perceptron_factory = classifier.PerceptronFactory()
        # result = classifier.evaluate(Perceptron_factory, 2)
        # writer.writerow([2, result[0], result[1]])




if __name__ == '__main__':
    main()