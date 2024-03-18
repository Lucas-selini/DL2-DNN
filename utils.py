from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def lire_alpha_digit(char) :
    """
    :param char: List of char we want to learn
    :return: A matrix with in line the data and in column the pixels
    """
    mat = loadmat("../data/binaryalphadigs.mat")
    data = mat['dat']

    output = []

    for i in char :
        for j in range(38) :
            out_inter = []
            for k in range(20) :
                for l in range(16) :
                    out_inter += [data[i][j][k][l]]
            output += [out_inter]
    return np.array(output)

def lire_MNIST(num) :
    mat = loadmat("../data/mnist_all.mat")
    char = 'train' + str(num)
    data = mat[char]

    # On passe d'un code 8-bit à un code binaire
    output = []
    seuil = 128

    for ligne in data :
        ligne_output = [1 if pixel >= seuil else 0 for pixel in ligne]
        output.append(ligne_output)

    return np.array(output)

def display_error(errors,parameters,window_size=5,save=False) :
    """
    :param errors: A matrix of subsequent error during the training
    :return: a gaph
    """

    plt.figure(figsize=(10, 6))

    handles = []  # Liste pour stocker les poignées des légendes
    labels = []  # Liste pour stocker les étiquettes des légendes

    for i, errors in enumerate(errors):
        # Couleur pour la courbe principale et légèrement plus foncée pour la courbe de tendance
        color = plt.cm.get_cmap('tab10')(i / len(parameters))
        darker_color = tuple(c * 0.8 for c in color)

        # Tracer la courbe principale avec légende
        handle, = plt.plot(range(len(errors)), errors, label=f"LR: {parameters[i]}", color=color,
                           alpha=0.7)  # Ajustement de l'opacité
        handles.append(handle)
        labels.append(f"LR: {parameters[i]}")

        # Calcul de la moyenne mobile (courbe de tendance) sans légende
        smoothed_errors = gaussian_filter1d(errors, sigma=window_size)
        plt.plot(range(len(smoothed_errors)), smoothed_errors, linestyle='--', color=darker_color,
                 alpha=1.0)  # Ajustement de l'opacité

    plt.xlabel('Iterations')
    plt.ylabel('Erreur de reconstruction')

    # Afficher la légende pour les courbes principales
    plt.legend(handles, labels, loc='best')

    plt.grid(True)
    if save :
        plt.savefig("errors.png")
    plt.show()

def display_final_error(errors,parameters,training_time,window_size=5,save=False) :
    final_errors = [errors[-1] for errors in errors]

    # Lisser les erreurs finales avec gaussian_filter1d
    smoothed_errors = gaussian_filter1d(final_errors, sigma=window_size)
    smoothed_time = gaussian_filter1d(training_time, sigma=window_size)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Tracer l'erreur sur l'axe y1
    ax1.plot(parameters, final_errors, linestyle='-', color='skyblue', label="Erreur à la fin de l'entraînement")
    ax1.plot(parameters, smoothed_errors, color='blue', linestyle='--', label='Smoothed Trend')
    ax1.set_xlabel("Nombre d'itérations")
    ax1.set_ylabel("Erreur à la fin de l'entraînement", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.legend(loc='upper left')

    # Créer un deuxième axe y2 partageant le même axe x
    ax2 = ax1.twinx()
    ax2.set_ylabel("Temps d'entraînement (en sec.)", color=(0.6, 0.8, 0.6))
    ax2.plot(parameters, training_time, linestyle='-', color=(0.6, 0.8, 0.6), label="Temps d'entraînement")
    ax2.plot(parameters, smoothed_time, color='green', linestyle='--', label='Smoothed Trend')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    plt.grid(True)
    if save :
        plt.savefig("erreur_time.png")
    plt.show()


def display_image(images,hauteur,largeur,save=False) :
    """
    :param image: a numpy array
    :return: a representation of the image
    """
    images = np.array(images)
    nb_images = images.shape[0]
    generated_images_array = images.reshape(nb_images, hauteur, largeur)

    # Afficher les images générées
    rows = 1  # Nombre de lignes dans la grille d'affichage
    cols = 1  # Nombre de colonnes dans la grille d'affichage

    plt.figure(figsize=(10, 6))

    for i in range(nb_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images_array[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    if save :
        plt.savefig("generated_image.png")
    plt.show()




