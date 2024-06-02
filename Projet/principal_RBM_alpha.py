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

def main():
    # Chargement des données
    data_sample = [10, 11, 12] # 10 = A, B = 11, ..., Z = 35
    data = lire_alpha_digit(data_sample)
    p = data.shape[1]

    # Paramètres liés au réseau et à l'apprentissage
    q = 200
    learning_rate = 0.05
    batch_size = 10
    nb_iter = 1001
    nb_gibbs_iteration = 200
    nb_image_generate = 15

    # Entrainement du DBN
    rbm_model = RBM(p,q)
    _ = rbm_model.train(data, learning_rate, batch_size, nb_iter, verbose=True, plot=True)

    # Génération d'images
    display_image(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)
    
if __name__ == "__main__":
    main()