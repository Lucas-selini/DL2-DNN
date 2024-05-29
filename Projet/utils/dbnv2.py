from utils.rbmv2 import RBM
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

    def train(self, X, learning_rate, batch_size, nb_iter, verbose=False, plot=False):
        """
        Trains the DBN model.

        Args:
            X (np.array): Input data of size n*p.
            nb_iter (int): Number of iterations.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            verbose (bool): If True, prints the loss at each iteration.
            plot (bool): If True, plots the loss at each iteration.

        Returns:
            None
        """
        X_copy = X.copy()
        for i in range(self.n_layers-1):
            self.rbms[i].train(X_copy, learning_rate, batch_size, nb_iter, verbose, plot)
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
        generated_images = []
        for _ in range(n_images):
            v = np.random.rand(1, self.rbms[0].p)

            for _ in range(n_gibbs):
                for i in range(self.n_layers-1):
                    prob_h = self.rbms[i].entree_sortie(v)
                    v = (np.random.rand(1, self.rbms[i].q) < prob_h).astype(int)

                for i in range(self.n_layers-2, -1, -1):
                    prob_v = self.rbms[i].sortie_entree(v)
                    v = (np.random.rand(1, self.rbms[i].p) < prob_v).astype(int)

            generated_images.append(np.round(v))

        return np.array(generated_images)
    
    def count_parameters(self):
        return sum([rbm.count_parameters() for rbm in self.rbms])