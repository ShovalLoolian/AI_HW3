import hw3_utils
import os

def main():
    train_features, train_labels, test_features = hw3_utils.load_data(os.path.join("data", "data.pickle"))



if __name__ == '__main__':
    main()