"""
This script trains a Restricted Boltzmann Machine (RBM) model on a dataset of alpha digits and generates images using the trained model.
It also studies the influence of different hyperparameters on the model's performance.

Parameters:
- p (int): Number of visible units in the RBM.
- q (int): Number of hidden units in the RBM.
- learning_rate (float): Learning rate for training the RBM.
- batch_size (int): Number of samples per batch for training the RBM.
- nb_iter (int): Number of training iterations.
- data_sample (list): List of integers representing the alpha digits to be used as training data.
- nb_gibbs_iteration (int): Number of Gibbs sampling iterations for generating images.
- nb_image_generate (int): Number of images to generate.

Returns:
- None
"""

from utils.rbm import RBM
from utils.utils import *

# Paramètres liés au réseau et à l'apprentissage
p = 320
q = 50
learning_rate = 0.01
batch_size = 5
nb_iter = 100
data_sample = [10] # 10 = A, B = 11, ..., Z = 35

nb_gibbs_iteration = 10
nb_image_generate = 1

# Chargement des données
data = lire_alpha_digit(data_sample)

# # Entrainement du DBN
# rbm_model = RBM(p,q)
# _ = rbm_model.train(data, learning_rate, batch_size, nb_iter)

# # Génération d'images
# display_image(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)

# Etude de l'influence des différents hyperparamètres
q_list = [1,5,10,20,40,50,75,100,200]
learning_rates = [0.0001,0.001,0.01,0.05,0.1,0.3,0.8]
images = []

for q in q_list:
    rbm_model = RBM(p,q)
    _ = rbm_model.train(data, learning_rate, batch_size, nb_iter)
    images.append(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate))

display_image(images,20,16,save=True)