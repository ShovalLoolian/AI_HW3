import hw3_utils, classifier
import os
import pickle


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
    train_features, train_labels, test_features = hw3_utils.load_data()
    knn = classifier.knn_factory(3).train(train_features, train_labels)
    print(knn.classify(test_features[59]))

    classifier.split_crosscheck_groups((train_features, train_labels), 2)
    train_features, train_labels = load_data_try('ecg_fold_1.data')
    train_features2, train_labels2 = load_data_try('ecg_fold_2.data')
    print(sum([1 for label in train_labels if label]))
    print(sum([1 for label in train_labels2 if label]))
    print(sum([1 for label in train_labels if not label]))
    print(sum([1 for label in train_labels2 if not label]))


    # data = [[1,1,1],[2,2,2],[100,3,3]]
    # lables = [True, False, False]
    # features = [4,4,4]
    # distances = [(classifier.euclidean_distance(data[i], features), i) for i in range(len(data))]
    # dataset = list(map(lambda x: x[1], sorted(distances, key=lambda x: x[0])[:2]))
    # trues = sum(list(map(lambda x: 1 if lables[x] == True else 0, dataset)))
    # print(list(map(lambda x: 1 if lables[x] == True else 0, dataset)))
    # print(trues >= 2 / 2)



if __name__ == '__main__':
    main()