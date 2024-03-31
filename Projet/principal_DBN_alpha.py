"""
This script trains a Deep Belief Network (DBN) on a dataset of alpha digits.
It then generates images using the trained DBN and displays them.

Parameters:
- data_sample: list of integers representing the alpha digits to be loaded
- p: integer, the number of input units
- q: integer, the number of hidden units
- learning_rate: float, the learning rate for training the DBN
- batch_size: integer, the batch size for training the DBN
- nb_iter: integer, the number of training iterations
- nb_gibbs_iteration: integer, the number of Gibbs sampling iterations for generating images
- nb_image_generate: integer, the number of images to generate

Returns:
- None
"""

from utils.dbn import DBN
from utils.utils import *

# Chargement des données
data_sample = [11] # 10 = A, B = 11, ..., Z = 35
data = lire_alpha_digit(data_sample)

# Paramètres liés au réseau et à l'apprentissage
p = data.shape[1] # 20*16 = 320
q = 10
learning_rate = 0.075
batch_size = 5
nb_iter = 5

nb_gibbs_iteration = 20
nb_image_generate = 1

# # Entrainement du DBN
# dbn_model = DBN([p,q,p+10,q+10,p+20,q+20,p+30])
# _ = dbn_model.train(data, nb_iter,learning_rate, batch_size)

# # Génération d'images
# display_image(dbn_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)

# Etude de l'influence des différents hyperparamètres
learning_rates = [0.1, 0.2]
images = []

for learning_rate in learning_rates:
    dbn_model = DBN([p,q,q])
    _ = dbn_model.train(data,learning_rate, batch_size, nb_iter)
    images.append(dbn_model.generer_image(nb_gibbs_iteration,nb_image_generate))

display_image(images,20,16,save=True)
