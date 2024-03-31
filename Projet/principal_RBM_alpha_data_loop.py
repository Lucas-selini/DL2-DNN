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
q = 130
learning_rate = 0.01
batch_size = 5
nb_iter = 200
data_sample = [10] # 10 = A, B = 11, ..., Z = 35
data_samples = [[10],[10,11],[10,11,12],[10,11,12,13],[10,11,12,13,14,15,16,17],[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]
nb_gibbs_iteration = 40
nb_image_generate = 1
images = []
for data_sample in data_samples:
    # Chargement des données
    data = lire_alpha_digit(data_sample)
    rbm_model = RBM(p,q)
    _ = rbm_model.train(data, learning_rate, batch_size, nb_iter)
    images.append(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate))

display_image(images,20,16,save=True)