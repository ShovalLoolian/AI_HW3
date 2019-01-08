import hw3_utils, classifier
import os



def main():
    train_features, train_labels, test_features = hw3_utils.load_data()
    knn = classifier.knn_factory(3).train(train_features, train_labels)
    print(knn.classify(test_features[59]))


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