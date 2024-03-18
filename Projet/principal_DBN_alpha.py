from utils.dbn import DBN
from utils.utils import *

# Paramètres liés au réseau et à l'apprentissage
p = 320
q = 50
learning_rate = 0.2
batch_size = 5
nb_iter = 100
data_sample = [10] # 10 = A, B = 11, ..., Z = 35

nb_gibbs_iteration = 20
nb_image_generate = 1

# Chargement des données
data = lire_alpha_digit(data_sample)

# # Entrainement du DBN
# dbn_model = DBN([p,q,p+10,q+10,p+20,q+20,p+30])
# _ = dbn_model.train(data, nb_iter,learning_rate, batch_size)

# # Génération d'images
# display_image(dbn_model.generer_image(nb_gibbs_iteration,nb_image_generate),20,16,save=True)

# Etude de l'influence des différents hyperparamètres
learning_rates = [0.001,0.01,0.1,0.2,0.3,0.4,0.5]
images = []

for learning_rate in learning_rates:
    dbn_model = DBN([p,q,p+10,q+10,p+20,q+20,p+30])
    _ = dbn_model.train(data, nb_iter,learning_rate, batch_size)
    images.append(dbn_model.generer_image(nb_gibbs_iteration,nb_image_generate))
    
display_image(images,20,16,save=True)
