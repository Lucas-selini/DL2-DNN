from utils.rbm import RBM
import numpy as np

class DBN():
    def __init__(self, couche):
        """
        Deep Belief Network (DBN) class.

        Args:
            couche (list): List of number of neurons for each layer.
        """
        self.n_layers = len(couche)
        self.rbms = [RBM(couche[i], couche[i+1]) for i in range(self.n_layers-1)]

    def train(self, X, learning_rate, batch_size, nb_iter):
        """
        Trains the DBN model.

        Args:
            X (np.array): Input data of size n*p.
            nb_iter (int): Number of iterations.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.

        Returns:
            None
        """
        X_copy = X.copy()
        for i in range(self.n_layers-2):
            self.rbms[i].train(X_copy, learning_rate, batch_size, nb_iter)
            X_copy = self.rbms[i].entree_sortie(X_copy)

    
    def generer_image(self, n_gibbs, n_images):
        """
        Generates images using the trained DBN model.

        Args:
            n_gibbs (int): Number of Gibbs iterations.
            n_images (int): Number of images to generate.

        Returns:
            np.array: Array of generated images of size n_images*p.
        """
        v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0])).astype(int)
        for j in range(n_gibbs):
            h = (np.random.rand(n_images, self.rbms[self.n_layers-2].b.shape[0]) < self.rbms[self.n_layers-2].entree_sortie(v)).astype(int)
            v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < self.rbms[self.n_layers-2].sortie_entree(h)).astype(int)
        for i in range(self.n_layers-3, -1, -1):
            v = self.rbms[i].sortie_entree(v)
        return np.array(v)