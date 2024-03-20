import numpy as np
from utils.dnn import DNN
from utils.utils import *
import matplotlib.pyplot as plt
# import time
# import copy

# # Chargement des données
nums = [0,1,2]
X, Y = lire_MNIST(nums)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Paramètres liés au réseau et à l'apprentissage
p = X.shape[1]
q = 40
learning_rate = 0.075
batch_size = 125
nb_iter = 100
n_epochs = 21
nb_gibbs_iteration = 8
nb_image_generate = 1
layers = [p,q,len(nums)]

## Analyse : Création des courbes

# Define the number of neurons per layer
neurons_per_layer = 200

# Define the range of layers to test
layers_range = range(2, 6)

# Initialize lists to store accuracies
pretrained_train_accuracies = []
pretrained_test_accuracies = []
non_pretrained_train_accuracies = []
non_pretrained_test_accuracies = []

for n_layers in layers_range:
    # Create a list representing the layers of the network
    layers = [p] + [neurons_per_layer]*(n_layers) + [len(nums)]

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
plt.plot(layers_range, pretrained_train_accuracies, label='With pretraining - Train')
plt.plot(layers_range, pretrained_test_accuracies, label='With pretraining - Test')
plt.plot(layers_range, non_pretrained_train_accuracies, label='Without pretraining - Train')
plt.plot(layers_range, non_pretrained_test_accuracies, label='Without pretraining - Test')
plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# # # Entrainement du DNN avec pretraining
# dnn = DNN(layers)
# dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)
# dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)

# # # Comparaison avec un DNN sans pretraining
# dnn_without_pretraining = DNN(layers)
# dnn_without_pretraining.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)

# print('\n')
# print('Pre-trained DNN - Train accuracy:', dnn.test_DNN(X_train, Y_train))
# print('Pre-trained DNN - Test accuracy:', dnn.test_DNN(X_test, Y_test))
# print('\n')
# print('Not-pre-trained DNN - Train accuracy:', dnn_without_pretraining.test_DNN(X_train, Y_train))
# print('Not-pre-trained DNN - Test accuracy:', dnn_without_pretraining.test_DNN(X_test, Y_test))