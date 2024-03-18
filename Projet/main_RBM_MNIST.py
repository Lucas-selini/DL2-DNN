from utils.rbm import RBM
from utils.dbn import DBN
from utils.dnn import DNN
from utils.utils import*
import time


p = 784
q = 50
learning_rate = 0.2
batch_size = 5
nb_iter = 100
num = 3

nb_gibbs_iteration = 20
nb_image_generate = 1

generated_images = []
errors = []
training_time = []

data = lire_MNIST(num)
display_image([data[500]],28,28,save=True)

for iter in range(1) :

    print(iter)

    rbm_model = RBM(p,q)

    start_time = time.time()
    training_errors = rbm_model.train(data, learning_rate, batch_size, nb_iter)
    end_time = time.time()

    errors.append(training_errors)
    training_time.append(end_time-start_time)

    generated_image = rbm_model.generer_image(nb_gibbs_iteration,nb_image_generate)
    generated_images.append(generated_image)

display_image(generated_images,28,28,save=True)
