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

from utils.rbmv2 import RBM
from utils.utils import *

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_train, _, _, _ = lire_MNIST(nums)


# Chargement des données
# data_sample = [10,11,12,13,14,15] # 10 = A, B = 11, ..., Z = 35
# data = lire_alpha_digit(data_sample)
# p = data.shape[1]

# Paramètres liés au réseau et à l'apprentissage
q = 200
learning_rate = 0.05
batch_size = 10
nb_iter = 1000
nb_gibbs_iteration = 200
nb_image_generate = 15


###### Affichage des données
# permutation = np.random.permutation(data.shape[0])
# data_shuffled = data[permutation]

# fig, axes = plt.subplots(1, 15, figsize=(20, 2))
# for i in range(15):
#     axes[i].imshow(data_shuffled[i].reshape(20, 16), cmap='gray')
#     axes[i].axis('off')

# # plt.imshow(data[19].reshape(20, 16), cmap='gray')
# plt.show()

# data = np.delete(data, 19, axis=0)



###### Entrainement du RBM
# Entrainement du DBN
# rbm_model = RBM(p,q)
# _ = rbm_model.train(data, learning_rate, batch_size, nb_iter, verbose=True, plot=False)

# # Génération d'images
# display_image(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)