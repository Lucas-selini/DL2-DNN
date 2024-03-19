import numpy as np
from utils.dnn import DNN
from utils.utils import *
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

# # Entrainement du DNN
dnn = DNN(layers)
dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)
dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)

# # Comparaison avec un DNN uniquement pre-trained
dnn_pretrained = DNN(layers)
dnn_pretrained.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)

print('\n')
print('Trained DNN - Train accuracy:', dnn.test_DNN(X_train, Y_train))
print('Trained DNN - Test accuracy:', dnn.test_DNN(X_test, Y_test))
print('\n')
print('Pre-trained DNN - Train accuracy:', dnn_pretrained.test_DNN(X_train, Y_train))
print('Pre-trained DNN - Test accuracy:', dnn_pretrained.test_DNN(X_test, Y_test))

