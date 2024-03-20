"""
This script trains and compares the performance of a Deep Neural Network (DNN) with and without pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Creates a list of training sizes for the DNN.
5. Trains a DNN with pretraining for each training size and records the training and testing mistake_rates.
6. Trains a DNN without pretraining for each training layer size and records the training and testing mistake_rates.
7. Plots the mistake_rates for comparison.
"""

from utils.dnn import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    """
    Main function that trains and compares the performance of a Deep Neural Network (DNN) with and without pretraining on the MNIST dataset.
    """
    # Load the MNIST dataset for the specified digit classes
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, X_test, Y_train, Y_test = lire_MNIST(nums)

    # Define the parameters for the DNN and the training process
    p = X_train.shape[1]
    q = 200
    learning_rate = 0.075
    batch_size = 125
    nb_iter = 100
    n_epochs = 51
    layers = [p, q, q, len(nums)]

    # Define the number of training datas to train the DNN
    data_train_range = [1000, 3000, 7000, 10000, 30000, 60000]

    # Initialize lists to store mistake_rates
    pretrained_train_mistake_rates = []
    pretrained_test_mistake_rates = []
    non_pretrained_train_mistake_rates = []
    non_pretrained_test_mistake_rates = []

    # Train a DNN with pretraining and without pretraining for each number of training datas
    for train_size in data_train_range:
        
        # Split the dataset into training and testing sets
        X_train_split, _, Y_train_split, _ = train_test_split(X_train, Y_train, train_size=train_size, random_state=42)
        
        # Train a DNN with pretraining
        dnn = DNN(layers)
        dnn.pretrain_DNN(X_train_split, learning_rate, batch_size, nb_iter)
        dnn.retropropagation(X_train_split, Y_train_split, learning_rate, n_epochs, batch_size)
        pretrained_train_mistake_rate = dnn.test_DNN(X_train_split, Y_train_split)
        pretrained_test_mistake_rate = dnn.test_DNN(X_test, Y_test)
        pretrained_train_mistake_rates.append(pretrained_train_mistake_rate)
        pretrained_test_mistake_rates.append(pretrained_test_mistake_rate)

        # Train a DNN without pretraining
        dnn_without_pretraining = DNN(layers)
        dnn_without_pretraining.retropropagation(X_train_split, Y_train_split, learning_rate, n_epochs, batch_size)
        non_pretrained_train_mistake_rate = dnn_without_pretraining.test_DNN(X_train_split, Y_train_split)
        non_pretrained_test_mistake_rate = dnn_without_pretraining.test_DNN(X_test, Y_test)
        non_pretrained_train_mistake_rates.append(non_pretrained_train_mistake_rate)
        non_pretrained_test_mistake_rates.append(non_pretrained_test_mistake_rate)
        
        print('Finished training for', train_size, 'training datas')

    # Plot the mistake_rates
    plt.plot(data_train_range, pretrained_train_mistake_rates, 'o-', linewidth=1, label='With pretraining - Train', marker='o')
    plt.plot(data_train_range, pretrained_test_mistake_rates, 'o-', linewidth=1, label='With pretraining - Test', marker='o')
    plt.plot(data_train_range, non_pretrained_train_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Train', marker='o')
    plt.plot(data_train_range, non_pretrained_test_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Test', marker='o')
    plt.xlabel('Training size')
    plt.ylabel('Mistake rate')
    plt.legend()
    plt.savefig('plots/Figure_3_train_size.png')
    plt.show()

if __name__ == "__main__":
    main()
