import numpy as np
import copy
from dbn import DBN

class DNN :
    def __init__(self, couche):
        """initie la classe DNN

        Args:
            couche (list): Liste des couches du réseau 
        """ 
        self.dbn = DBN(couche)
        
    
    def train_DNN(self, X, n_epochs, learning_rate, batch_size):
        self.dbn.train_DBN(X, n_epochs, learning_rate, batch_size)
    
    def calcul_softmax(self, X):
        """Calcul de la fonction softmax

        Args:
            X (np_array): données d'entrée

        Returns:
            np_array: vecteur de sortie
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def entree_sortie_reseau(self,X):
        """Calcul de la sortie du réseau

        Args:
            X (np_array): données d'entrée

        Returns:
            np_array: vecteur de sortie
        """
        L = [X]
        for rbm in self.dbn.rbms:
            X = rbm.entree_sortie(X)
            L.append(X)
        return L, self.calcul_softmax(L[-1])
    
    def retropropagation(self, X, Y, learning_rate,n_epochs,batch_size):
        """Rétropropagation

        Args:
            X (np_array): données d'entrée
            Y (np_array): données de sortie
            learning_rate (float): taux d'apprentissage
            n_epochs (int): nombre d'epochs
            batch_size (int): taille des batchs
        """
        X_copy= X.copy()
        for epoch in range(n_epochs):
            for j in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[j:min(j+batch_size, X_copy.shape[0])]
                Y_batch = Y[j:min(j+batch_size, X_copy.shape[0])]
                tb = X_batch.shape[0]
                L, Y_hat = self.entree_sortie_reseau(X_batch)
                delta = Y_hat - Y_batch
                dbm_copy = copy.deepcopy(self.dbn)
                for i in range(self.dbn.n_layers-2, -1, -1):
                    dbm_copy.rbms[i].b -= learning_rate * np.mean(delta, axis=0)
                    dbm_copy.rbms[i].W -= learning_rate * np.dot(L[i].T, delta) / tb
                    delta = np.dot(delta, self.dbn.rbms[i].W.T) * L[i] * (1 - L[i])
                self.dbn = dbm_copy
            L, Y_hat = self.entree_sortie_reseau(X_copy)
            loss = -np.mean(Y * np.log(Y_hat))
            print(f"Loss at epoch {epoch} : {loss}")

    def test_DNN(self, X, Y):
        """Test du réseau

        Args:
            X (np_array): données d'entrée
            Y (np_array): données de sortie

        Returns:
            float: taux de réussite
        """
        _, Y_hat = self.entree_sortie_reseau(X)
        return np.mean(np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1))