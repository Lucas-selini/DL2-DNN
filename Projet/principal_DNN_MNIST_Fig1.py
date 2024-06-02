"""
This script trains and compares the performance of a Deep Neural Network (DNN) with and without pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Creates a list of hidden layer sizes for the DNN.
5. Trains a DNN with pretraining for each hidden layer size and records the training and testing mistake_rates.
6. Trains a DNN without pretraining for each hidden layer size and records the training and testing mistake_rates.
7. Plots the mistake_rates for comparison.
"""

from utils.dnn import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    # Load the MNIST dataset for the specified digit classes
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, X_test, Y_train, Y_test = lire_MNIST(nums)
    
    # read X_train , X_test, Y_train, Y_test :
    # X_train = np.load('data/X_train_MNIST.npy')
    # X_test = np.load('data/X_test_MNIST.npy')
    # Y_train = np.load('data/Y_train_MNIST.npy')
    # Y_test = np.load('data/Y_test_MNIST.npy')
    X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size=30000, random_state=46)
    
    # Define the parameters for the DNN and the training process
    p = X_train.shape[1]
    q = 100
    n_classes = len(nums)
    learning_rate = 0.1
    batch_size = 64
    nb_iter = 76
    n_epochs = 126

    # Define the number of hidden layers
    num_hidden_layers = range(1, 6)

    # Initialize lists to store mistake_rates
    pretrained_train_mistake_rates = []
    pretrained_test_mistake_rates = []
    non_pretrained_train_mistake_rates = []
    non_pretrained_test_mistake_rates = []

    # Train a DNN with pretraining and without pretraining for each number of hidden layers
    for num_layers in num_hidden_layers:
        # Create a list representing the layers of the network
        layers = [p] + [q] * num_layers

        # Train a DNN with pretraining
        dnn = DNN(layers, n_classes)
        dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter, verbose=False, plot=False)
        dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size, verbose=True, plot=True)
        pretrained_train_mistake_rate = dnn.test_DNN(X_train, Y_train)
        pretrained_test_mistake_rate = dnn.test_DNN(X_test, Y_test)
        pretrained_train_mistake_rates.append(pretrained_train_mistake_rate)
        pretrained_test_mistake_rates.append(pretrained_test_mistake_rate)

        # Train a DNN without pretraining
        dnn_without_pretraining = DNN(layers, n_classes)
        dnn_without_pretraining.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size, verbose=True, plot=True)
        non_pretrained_train_mistake_rate = dnn_without_pretraining.test_DNN(X_train, Y_train)
        non_pretrained_test_mistake_rate = dnn_without_pretraining.test_DNN(X_test, Y_test)
        non_pretrained_train_mistake_rates.append(non_pretrained_train_mistake_rate)
        non_pretrained_test_mistake_rates.append(non_pretrained_test_mistake_rate)
        
        print('Finished training for', num_layers, 'hidden layers')

    # Plot the mistake_rates
    plt.plot(num_hidden_layers, pretrained_train_mistake_rates, 'o-', linewidth=1, label='With pretraining - Train')
    plt.plot(num_hidden_layers, pretrained_test_mistake_rates, 'o-', linewidth=1, label='With pretraining - Test')
    plt.plot(num_hidden_layers, non_pretrained_train_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Train')
    plt.plot(num_hidden_layers, non_pretrained_test_mistake_rates, 'o-', linewidth=1, label='Without pretraining - Test')
    plt.xlabel('Number of hidden layers')
    plt.ylabel('Mistake rate')
    plt.legend()
    plt.savefig('plots/Figure_1_nb_layers.png')
    plt.show()
    
if __name__ == "__main__":
    main()