"""
This script trains and compares the performance of a Deep Neural Network (DNN) with and without pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Creates a list of hidden layer sizes for the DNN.
5. Trains a DNN with pretraining for each hidden layer size and records the training and testing accuracies.
6. Trains a DNN without pretraining for each hidden layer size and records the training and testing accuracies.
7. Plots the accuracies for comparison.
"""

from utils.dnn import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the MNIST dataset for the specified digit classes
nums = [0, 1, 2]
X, Y = lire_MNIST(nums)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the parameters for the DNN and the training process
p = X.shape[1]
q = 40
learning_rate = 0.075
batch_size = 125
nb_iter = 100
n_epochs = 21
nb_gibbs_iteration = 8
nb_image_generate = 1
layers = [p, q, len(nums)]

# Define the number of neurons for hidden layers
neurons_range = [100, 300, 500, 700]

# Initialize lists to store accuracies
pretrained_train_accuracies = []
pretrained_test_accuracies = []
non_pretrained_train_accuracies = []
non_pretrained_test_accuracies = []

# Train a DNN with pretraining and without pretraining for each hidden layer size
for neurons in neurons_range:
    # Create a list representing the layers of the network
    layers = [p, neurons, neurons, len(nums)]

    # Train a DNN with pretraining
    dnn = DNN(layers)
    dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)
    dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)
    pretrained_train_accuracy = dnn.test_DNN(X_train, Y_train)
    pretrained_test_accuracy = dnn.test_DNN(X_test, Y_test)
    pretrained_train_accuracies.append(pretrained_train_accuracy)
    pretrained_test_accuracies.append(pretrained_test_accuracy)

    # Train a DNN without pretraining
    dnn_without_pretraining = DNN(layers)
    dnn_without_pretraining.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)
    non_pretrained_train_accuracy = dnn_without_pretraining.test_DNN(X_train, Y_train)
    non_pretrained_test_accuracy = dnn_without_pretraining.test_DNN(X_test, Y_test)
    non_pretrained_train_accuracies.append(non_pretrained_train_accuracy)
    non_pretrained_test_accuracies.append(non_pretrained_test_accuracy)

# Plot the accuracies
plt.plot(neurons_range, pretrained_train_accuracies, 'o-', linewidth=1, label='With pretraining - Train')
plt.plot(neurons_range, pretrained_test_accuracies, 'o-', linewidth=1, label='With pretraining - Test')
plt.plot(neurons_range, non_pretrained_train_accuracies, 'o-', linewidth=1, label='Without pretraining - Train')
plt.plot(neurons_range, non_pretrained_test_accuracies, 'o-', linewidth=1, label='Without pretraining - Test')
plt.xlabel('Number of neurons in hidden layers')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
