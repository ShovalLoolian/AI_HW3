import hw3_utils, classifier
import os
import pickle
import csv, sklearn, math

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
    #q b.3
    train_features, train_labels, test_features = hw3_utils.load_data()
    classifier.split_crosscheck_groups((train_features, train_labels), 8)

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

    # with open("experiments6.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for k in [1]:#,3,5,7,9,11,13]:
    #         factory = classifier.ID3Factory()
    #         ID3_classifier = factory.train(train_featurs_final, train_labels_final)
    #
    #         classifies = [ID3_classifier.classify(sample) for sample in test_final]
    #
    #         false_positive = [i if classifies[i] != test_labels_final[i] and test_labels_final[i] == True else 0 for i in range(len(test_final))]
    #         false_negative = [i if classifies[i] != test_labels_final[i] and test_labels_final[i] == False else 0 for i in range(len(test_final))]
    #
    #         true_positive = [i if classifies[i] == test_labels_final[i] and test_labels_final[i] == True else 0 for i in range(len(test_final))]
    #         true_negative = [i if classifies[i] == test_labels_final[i] and test_labels_final[i] == False else 0 for i in range(len(test_final))]
    #
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