from rbm import RBM
import numpy as np

class DBN:
    def __init__(self, couche):
        """initie la classe DBN

        Args:
            couche (list): Liste des couches du réseau 
        """
        self.n_layers = len(couche)
        self.rbms = [RBM(couche[i], couche[i+1]) for i in range(self.n_layers-1)]

    def train(self, X, n_epochs, learning_rate, batch_size,is_DNN = False):
        X_copy= X.copy()
        if not is_DNN:
            for i in range(self.n_layers-1):
                self.rbms[i].train_RBM(X_copy, n_epochs, learning_rate, batch_size)
                X_copy = self.rbms[i].entree_sortie(X_copy)
        else:
            for i in range(self.n_layers-2):
                self.rbms[i].train_RBM(X_copy, n_epochs, learning_rate, batch_size)
                X_copy = self.rbms[i].entree_sortie(X_copy)

    
    def generer_image(self, n_gibbs, n_images):
        """Génération d'images

        Args:
            n_iter (int): nombre d'itérations
            n_images (int): nombre d'images à générer

        Returns:
            np_array: vecteur d'images générées
        """
        v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) ).astype(int)
        for j in range(n_gibbs):
            h = (np.random.rand(n_images, self.rbms[self.n_layers-2].b.shape[0]) < self.rbms[self.n_layers-2].entree_sortie(v)).astype(int)
            v = (np.random.rand(n_images, self.rbms[self.n_layers-2].a.shape[0]) < self.rbms[self.n_layers-2].sortie_entree(h)).astype(int)
        for i in range(self.n_layers-3, -1, -1):
            v = self.rbms[i].sortie_entree(v)
        return np.array(v)