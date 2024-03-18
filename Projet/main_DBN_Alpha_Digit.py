from utils.dbn import DBN
# from utils.dnn import DNN
from utils.utils import *
import time


p = 320
q = 50
learning_rate = 0.1
batch_size = 5
nb_iter = 100
data_sample = [10]

nb_gibbs_iteration = 20
nb_image_generate = 1

generated_images = []
errors = []
training_time = []

#learning_rate = np.arange(0, 0.51, 0.01).tolist()
#q_list = np.arange(10, 150, 1).tolist()
#iter_list = np.arange(10,500,10).tolist()

data = lire_alpha_digit(data_sample)

for iter in range(1) :

    print(iter)

    dbn_model = DBN([p,q,p+10,q+10,p+20,q+20,p+30])

    start_time = time.time()
    training_errors = dbn_model.train(data, nb_iter,learning_rate, batch_size)
    end_time = time.time()

    errors.append(training_errors)
    training_time.append(end_time-start_time)

    generated_image = dbn_model.generer_image(nb_gibbs_iteration,nb_image_generate)
    generated_images.append(generated_image)

display_image(generated_images,20,16,save=True)
#display_error(errors,learning_rate,save=True)
#display_final_error(errors,iter_list,training_time,save=True)