"""
This script trains and compares the performance of a Deep Neural Network (DNN) with and without pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Creates a list of number of neurons of the hidden layers for the DNN.
5. Trains a DNN with pretraining for each number of neurons of the hidden layers and records the training and testing mistake_rates.
6. Trains a DNN without pretraining for each number of neurons of the hidden layers and records the training and testing mistake_rates.
7. Plots the mistake_rates for comparison.
"""

from utils.dnn import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    # Load the MNIST dataset for the specified digit classes
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X, Y = lire_MNIST(nums)

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/7, random_state=42)

    # Define the parameters for the DNN and the training process
    p = X.shape[1]
    learning_rate = 0.075
    batch_size = 125
    nb_iter = 100
    n_epochs = 51

    # Define the number of neurons for hidden layers
    neurons_range = [100, 300, 500, 700, 900]

    # Initialize lists to store mistake_rates
    pretrained_train_mistake_rates = []
    pretrained_test_mistake_rates = []
    non_pretrained_train_mistake_rates = []
    non_pretrained_test_mistake_rates = []

    # Train a DNN with pretraining and without pretraining for each number of neurons of the hidden layers
    for neurons in neurons_range:
        # Create a list representing the layers of the network
        layers = [p, neurons, neurons, len(nums)]

        # Train a DNN with pretraining
        dnn = DNN(layers)
        dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)
        dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)
        pretrained_train_mistake_rate = dnn.test_DNN(X_train, Y_train)
        pretrained_test_mistake_rate = dnn.test_DNN(X_test, Y_test)
        pretrained_train_mistake_rates.append(pretrained_train_mistake_rate)
        pretrained_test_mistake_rates.append(pretrained_test_mistake_rate)

        # Train a DNN without pretraining
        dnn_without_pretraining = DNN(layers)
        dnn_without_pretraining.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)
        non_pretrained_train_mistake_rate = dnn_without_pretraining.test_DNN(X_train, Y_train)
        non_pretrained_test_mistake_rate = dnn_without_pretraining.test_DNN(X_test, Y_test)
        non_pretrained_train_mistake_rates.append(non_pretrained_train_mistake_rate)
        non_pretrained_test_mistake_rates.append(non_pretrained_test_mistake_rate)

    # Plot the mistake_rates
    plt.plot(neurons_range, pretrained_train_mistake_rates, 'o-', linewidth=1, label='With pretraining - Train')
    plt.plot(neurons_range, pretrained_test_mistake_rates, 'o-', linewidth=1, label='With pretraining - Test')
    plt.plot(neurons_range, non_pretrained_train_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Train')
    plt.plot(neurons_range, non_pretrained_test_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Test')
    plt.xlabel('Number of neurons in hidden layers')
    plt.ylabel('Mistake rate')
    plt.legend()
    plt.savefig('plots/Figure_2_nb_neurons.png')
    plt.show()

if __name__ == "__main__":
    main()