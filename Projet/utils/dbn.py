from utils.rbm import RBM
import numpy as np

class DBN():
    def __init__(self, couche):
        """
        Args:
            couche (list): list of number of neurons for each layer
        """
        self.n_layers = len(couche)
        self.rbms = [RBM(couche[i], couche[i+1]) for i in range(self.n_layers-1)]

    def train(self, X, learning_rate, batch_size, nb_iter, is_DNN = False):
        """
        Args:
            X (np.array): size n*p
            nb_iter (int): number of iterations
            learning_rate (float): learning rate
            batch_size (int): batch size
        """
        X_copy= X.copy()
        if not is_DNN:
            for i in range(self.n_layers-1):
                self.rbms[i].train(X_copy, learning_rate, batch_size, nb_iter)
                X_copy = self.rbms[i].entree_sortie(X_copy)
        else:
            for i in range(self.n_layers-2):
                self.rbms[i].train(X_copy, learning_rate, batch_size, nb_iter)
                X_copy = self.rbms[i].entree_sortie(X_copy)

    
    def generer_image(self, n_gibbs, n_images):
        """
        Args:
            n_gibbs (int): number of gibbs iteration
            n_images (int): number of images to generate
        Return:
            (np.array) array of size n_images*p
        """
        v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) ).astype(int)
        for j in range(n_gibbs):
            h = (np.random.rand(n_images, self.rbms[self.n_layers-2].b.shape[0]) < self.rbms[self.n_layers-2].entree_sortie(v)).astype(int)
            v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < self.rbms[self.n_layers-2].sortie_entree(h)).astype(int)
        for i in range(self.n_layers-3, -1, -1):
            v = self.rbms[i].sortie_entree(v)
        return np.array(v)