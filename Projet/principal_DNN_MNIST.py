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
q = 50
learning_rate = 0.05
batch_size = 125
nb_iter = 100
n_epochs = 30
nb_gibbs_iteration = 10
nb_image_generate = 1

# # Entrainement du DNN
dnn = DNN([p,q,p,len(nums)])
dnn.pretrain_DNN(X_train, learning_rate, batch_size, nb_iter)
dnn.retropropagation(X_train, Y_train, learning_rate, n_epochs, batch_size)
print(dnn.test_DNN(X_test, Y_test))