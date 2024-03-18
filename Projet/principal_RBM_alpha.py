from utils.rbm import RBM
from utils.utils import *

# Paramètres liés au réseau et à l'apprentissage
p = 320
q = 50
learning_rate = 0.25
batch_size = 5
nb_iter = 1000
data_sample = [10] # 10 = A, B = 11, ..., Z = 35

nb_gibbs_iteration = 20
nb_image_generate = 5

# Chargement des données
data = lire_alpha_digit(data_sample)

# Entrainement du DBN
rbm_model = RBM(p,q)
_ = rbm_model.train(data, learning_rate, batch_size, nb_iter)

# Génération d'images
display_image(rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)