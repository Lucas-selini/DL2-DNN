import numpy as np
import matplotlib.pyplot as plt
from utils.utils import sigmoid
class RBM():
    def __init__(self, p, q):
        """
        Initialize a Restricted Boltzmann Machine (RBM) model.

        Args:
            p (int): input size (number of visible variables)
            q (int): output size (number of latent variables)
        """
        self.p = p
        self.q = q
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        # Modification de l'initialisation des poids pour une meilleure convergence
        self.W = np.random.normal(0, 0.01, (self.p, self.q))

    def entree_sortie(self, X):
        """
        Compute the output of the RBM given the input.

        Args:
            X (np.array): array of size n*p

        Return:
            (np.array) array of size n*q
        """
        prob_q = X @ self.W + self.b
        sortie = sigmoid(prob_q)
        return sortie
    
    def sortie_entree(self, H):
        """
        Compute the input of the RBM given the output.

        Args:
            H (np.array): array of size n*q

        Return:
            (np.array) array of size n*p
        """
        prob_p = H @ self.W.T + self.a
        entree = sigmoid(prob_p)
        return entree

    def train(self, X, lr, batch_size, nb_iter, verbose=False, plot=True):
        """
        Train the RBM model using Contrastive Divergence algorithm.

        Args:
            X (np.array): size n*p
            lr (float): learning_rate
            batch_size (int): batch_size
            nb_iter (int): number of iterations
        """
        errors = []
        for k in range(nb_iter):
            np.random.shuffle(X)

            for l in range(0, X.shape[0], batch_size):
                X_batch = X[l:min(X.shape[0], l + batch_size)]
                tb = X_batch.shape[0]

                # Forward
                v_0 = X_batch  # size tb*p
                p_h_v_0 = self.entree_sortie(v_0)  # size tb*q
                h_0 = (np.random.rand(tb, self.q) < p_h_v_0) * 1  # size tb*q
                # Backward
                p_v_h_0 = self.sortie_entree(h_0)  # size tb*p
                v_1 = (np.random.rand(tb, self.p) < p_v_h_0) * 1  # size tb*p
                p_h_v_1 = self.entree_sortie(v_1)

                # Compute gradients
                grad_a = np.sum(v_0 - v_1, axis=0)  # size p
                grad_b = np.sum(p_h_v_0 - p_h_v_1, axis=0)  # size q
                grad_W = v_0.T @ p_h_v_0 - v_1.T @ p_h_v_1  # size p*q

                # Update parameters
                self.W = self.W + (lr / tb) * grad_W
                self.a = self.a + (lr / tb) * grad_a
                self.b = self.b + (lr / tb) * grad_b


            # Calcul de la reconstruction pour évaluer la perte
            H = self.entree_sortie(X)
            X_rec = self.sortie_entree(H)
            loss = np.mean((X - X_rec)**2)  # Utilisation de l'erreur quadratique moyenne
            errors.append(loss)
            
            # Affichage de la perte toutes les 10 itérations si verbose est activé
            if verbose and k % 10 == 0:
                print(f"Mean Square Error at iteration {k}: {loss}")
            
            # Affichage du graphique de la perte si plot est activé
        if plot:
            plt.plot(errors)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

        return errors
        

    def generer_image(self, nb_iter_gibbs, nb_image):
        """
        Generate images using the RBM model.

        Args:
            nb_iter_gibbs (int): number of gibbs iteration
            nb_image (int): number of images to generate

        Return:
            (np.array) array of size nb_image*p
        """
        p = self.a.size
        q = self.b.size
        generated_images = []

        for _ in range(nb_image):
            v = (np.random.rand(p) < 0.5) * 1  # visible state
            for _ in range(nb_iter_gibbs):
                h = (np.random.rand(q) < self.entree_sortie(v)) * 1  # hidden state
                v = (np.random.rand(p) < self.sortie_entree(h)) * 1

            generated_images.append(v)

        return generated_images

    def count_parameters(self):
        return self.W.size + self.a.size + self.b.size