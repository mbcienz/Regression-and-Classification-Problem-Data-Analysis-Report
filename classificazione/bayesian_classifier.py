import numpy as np


# Definizione del classificatore Bayesiano
class BayesianClassifier:
    def __init__(self, mu_a, mu_b, sigma_matrix):
        """
        - mu_a (numpy.ndarray): Media della distribuzione per la classe positiva (+1).
        - mu_b (numpy.ndarray): Media della distribuzione per la classe negativa (-1).
        - sigma_matrix (numpy.ndarray): Matrice di covarianza condivisa tra le classi.
        """
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_matrix = sigma_matrix
        self.sigma_inv = np.linalg.inv(sigma_matrix)  # Inversa della matrice di covarianza

    def predict(self, X, y_test=None):
        """
        Esegue le predizioni sui dati di test.
        - X (numpy.ndarray): Dati di test da classificare (ogni colonna è un campione).
        - y_test (numpy.ndarray, opzionale): Etichette reali dei dati di test (per calcolare l'accuratezza).

        Ritorna:
        - predictions (numpy.ndarray): Etichette predette per i campioni di test.
        - accuracy (float, opzionale): Accuratezza del modello (se sono fornite le etichette reali).
        """
        predictions = []
        accuracy = None
        sigma1_sq = self.sigma_matrix[0, 0]  # sigma_1^2
        sigma2_sq = self.sigma_matrix[1, 1]  # sigma_2^2
        for x in X.T:
            decision_value = (x[0] / sigma1_sq) + (x[1] / sigma2_sq)
            predictions.append(1 if decision_value > 0 else -1 if decision_value < 0 else np.random.choice(
                [1, -1]))  # Assegna la classe a con label 1, b con label -1

        if y_test is not None:
            accuracy = self.calculate_accuracy(y_test, predictions)

        return np.array(predictions), accuracy

    # Funzione per calcolare l'accuratezza
    def calculate_accuracy(self, y_true, y_pred):
        """
        Calcola l'accuratezza del modello.
        Parametri:
        - y_true (numpy.ndarray): Etichette reali dei dati.
        - y_pred (numpy.ndarray): Etichette predette dal modello.

        Ritorna:
        - accuracy (float): Percentuale di predizioni corrette.
        """
        return np.mean(y_true == y_pred)
