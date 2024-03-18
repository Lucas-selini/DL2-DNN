import numpy as np
from utils.dnn import DNN
from utils.utils import *
import time
import copy

# Paramètres liés au réseau et à l'apprentissage
p = 784
q = 50
learning_rate = 0.2
batch_size = 5
nb_iter = 100
n_epochs = 200
num = [0, 1]

nb_gibbs_iteration = 20
nb_image_generate = 1

# Load the MNIST dataset and split it into training and test sets
X_train, X_test, y_train, y_test = lire_MNIST_v2(num)

# Entrainement du DNN
dnn = DNN([p,q,p+10,q+10,len(num)])
#dnn.pretrain_DNN(X_train[:6000], nb_iter, learning_rate, batch_size)
dnn.retropropagation(X_train, y_train, learning_rate, n_epochs, batch_size)