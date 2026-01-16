from bayesian_classifier import BayesianClassifier
from logistic_regression import *


def generate_data(mu_a, mu_b, p_a, p_b, sigma_matrix, n_samples):
    """
    Genera dati casuali per un problema di classificazione binaria utilizzando distribuzioni normali.
    - mu_a (numpy.ndarray): Vettore che rappresenta la media della distribuzione per la classe positiva (+1).
    - mu_b (numpy.ndarray): Vettore che rappresenta la media della distribuzione per la classe negativa (-1).
    - p_a (float): Probabilità a priori della classe positiva.
    - p_b (float): Probabilità a priori della classe negativa.
    - sigma_matrix (numpy.ndarray): Matrice di covarianza per le distribuzioni delle due classi.
    - n_samples (int): Numero di campioni da generare per il dataset.

     Ritorna:
    - X (numpy.ndarray): Matrice con i dati generati, ogni colonna rappresenta un campione.
    - y (numpy.ndarray): Vettore contenente le etichette corrispondenti per ogni campione.
    """

    y = np.random.choice([-1, 1], size=n_samples, p=[p_b, p_a])
    x = np.zeros((2, n_samples))  # i dati li ho per colonna
    for i in range(n_samples):
        x[:, i] = np.random.multivariate_normal(mu_a, sigma_matrix) if y[i] == 1 else np.random.multivariate_normal(
            mu_b, sigma_matrix)
    return x, y


# Impostiamo il seme per la riproducibilità
np.random.seed(2024)

# Definizione delle distribuzioni corrette secondo la traccia
sigma1_squared = 1  # sigma_1^2 = 1
sigma2_squared = 2  # sigma_2^2 = 2

# Matrice di covarianza
sigma_matrix = np.array([[sigma1_squared, 0], [0, sigma2_squared]])

# Media delle due classi (centri delle distribuzioni)
mu_a = np.array([0.5, 0.5])  # Classe Y = a => label(1)
mu_b = np.array([-0.5, -0.5])  # Classe Y = b => label(-1)

# Probabilità a priori uniformi
p_a = 0.5
p_b = 0.5

# Numero di campioni per i test Monte Carlo
n_samples_test = 2000
n_samples_train = 8000
n_montecarlo = 10

# **CASO 1: Classificatore Bayesiano**
classifier_bayesian = BayesianClassifier(mu_a, mu_b, sigma_matrix)

# ** CASO 2: Classificatore Logistico**
classifier_logistic = Classifier_Logistic(n_features=2)
classifier_logistic.fit(n_samples_train, mu_a, mu_b, sigma_matrix, p_a, p_b, n_montecarlo)

# Classificatore logistico allenato con lr variabile
classifier_logistic_lr = Classifier_Logistic(n_features=2, constant_step_size=False, lr=1)
classifier_logistic_lr.fit(n_samples_train, mu_a, mu_b, sigma_matrix, p_a, p_b, n_montecarlo)

# Inizializzazione degli array per memorizzare l'accuratezza dei classificatori
accuracy_bayesian = np.zeros(n_montecarlo)
accuracy_sgd = np.zeros(n_montecarlo)  # test con learning rate fisso
accuracy_sgd_lr = np.zeros(n_montecarlo)  # test con learning rate variabile

# Simulazione di Monte Carlo per stimare l'accuratezza media
for mc in range(n_montecarlo):
    # Generazione test set
    x_test, y_test = generate_data(mu_a, mu_b, p_a, p_b, sigma_matrix, n_samples_test)

    # Classificatore Bayesiano
    _, accuracy_bayesian[mc] = classifier_bayesian.predict(x_test, y_test)

    # Aggiunge il termine 1 per ogni campione (colonna) per includere l'intercetta
    X_test = np.vstack((x_test, np.ones((1, x_test.shape[1]))))

    # Classificatore Logistico con SGD a step costante
    _, accuracy_sgd[mc] = classifier_logistic.predict(X_test, y_test)

    # Classificatore Logistico con SGD a step variabile
    _, accuracy_sgd_lr[mc] = classifier_logistic_lr.predict(X_test, y_test)

# **Stampa dei risultati finali**

print(f"Accuracy Bayesiano: {np.mean(accuracy_bayesian):.4f}")
print(f"Errore Bayesiano: {1 - np.mean(accuracy_bayesian):.4f}")

print(f"Accuracy Regressione Logistica (LR Costante): {np.mean(accuracy_sgd):.4f}")
print(f"Errore Regressione Logistica (LR Costante): {1 - np.mean(accuracy_sgd):.4f}")

print(f"Accuracy Regressione Logistica (LR Variabile): {np.mean(accuracy_sgd_lr):.4f}")
print(f"Errore Regressione Logistica (LR Variabile): {1 - np.mean(accuracy_sgd_lr):.4f}")
