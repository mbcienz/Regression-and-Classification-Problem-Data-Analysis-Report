import numpy as np
from matplotlib import pyplot as plt


def generate_data(mu_a, mu_b, p_a, p_b, sigma_matrix, n_samples):
    """
    Genera un dataset casuale per un problema di classificazione binaria utilizzando distribuzioni normali.
    Parametri:
    - mu_a (numpy.ndarray): Media della distribuzione per la classe positiva (+1).
    - mu_b (numpy.ndarray): Media della distribuzione per la classe negativa (-1).
    - p_a (float): Probabilità a priori della classe positiva.
    - p_b (float): Probabilità a priori della classe negativa.
    - sigma_matrix (numpy.ndarray): Matrice di covarianza per le distribuzioni delle due classi.
    - n_samples (int): Numero di campioni da generare.

    Ritorna:
    - X (numpy.ndarray): Matrice contenente i dati generati (aggiunge una riga con tutti 1 per includere l'intercetta).
    - y (numpy.ndarray): Vettore contenente le etichette (classi) per ogni campione.
    """

    y = np.random.choice([-1, 1], size=n_samples, p=[p_b, p_a])
    x = np.zeros((2, n_samples))  # i dati li ho per colonna
    for i in range(n_samples):
        x[:, i] = np.random.multivariate_normal(mu_a, sigma_matrix) if y[i] == 1 else np.random.multivariate_normal(
            mu_b, sigma_matrix)

    X = np.vstack((x, np.ones((1, n_samples))))  # Aggiunge il termine 1 ad ogni campione

    return X, y


def compute_gradient_linreg(weights, X, y):
    """
    Calcola il gradiente della funzione di costo per la regressione lineare (Errore Quadratico Medio).
    Parametri:
    - weights (numpy.ndarray): Vettore dei pesi attuali.
    - X (numpy.ndarray): Campione di input (un esempio del dataset).
    - y (float): Etichetta associata al campione.

    Ritorna:
    - grad (numpy.ndarray): Gradiente della funzione di costo MSE rispetto ai pesi.
    """
    return 2 * (X * np.transpose(X) * weights - X * y)


def compute_gradient_logreg(w, X, y):
    """
    Calcola il gradiente della funzione di costo per la regressione logistica (Log-Loss).
    Parametri:
    - w (numpy.ndarray): Vettore dei pesi attuali.
    - X (numpy.ndarray): Campione di input (un esempio del dataset).
    - y (float): Etichetta associata al campione.

    Ritorna:
    - grad (numpy.ndarray): Gradiente della funzione di costo logistica rispetto ai pesi.
    """
    return -y * X / (1 + np.exp(y * np.dot(np.transpose(w), X)))


class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, constant_step_size=True):
        """
        Inizializza l'algoritmo di Stochastic Gradient Descent (SGD).
        Parametri:
        - learning_rate (float): Tasso di apprendimento iniziale.
        - constant_step_size (bool): Se True, il tasso di apprendimento resta costante; altrimenti decresce nel tempo.
        - true_weights (numpy.ndarray, opzionale): Pesi reali per confronti (se disponibili).
        """
        self.lr = learning_rate
        self.is_constant = constant_step_size
        self.weights = None

    def update_lr(self, iter):
        """
        Aggiorna il tasso di apprendimento in base all'iterazione corrente.
        Parametri:
        - iter (int): Numero dell'iterazione corrente.
        Ritorna:
        - Nuovo valore del tasso di apprendimento.
        """
        if not self.is_constant:
            updated_lr = self.lr / iter  # Decadimento del learning rate
        else:
            updated_lr = self.lr  # Mantiene il valore costante
        return updated_lr

    def run_sgd(self, X, y, cost_funct):
        """
        Esegue l'algoritmo SGD sui dati forniti.
        Parametri:
        - X (numpy.ndarray): Matrice dei dati di input.
        - y (numpy.ndarray): Vettore delle etichette.
        - cost_funct (str): Funzione di costo da ottimizzare ('mse' per regressione lineare, 'logloss' per regressione logistica).
        Ritorna:
        - weights (numpy.ndarray): Matrice con l'evoluzione dei pesi durante le iterazioni.
        """
        n_features, n_iters = X.shape  # Numero di feature e numero di iterazioni (campioni)
        self.weights = np.zeros((n_features, n_iters))  # Inizializza i pesi a zero

        for i in range(1, n_iters):
            if cost_funct == 'mse':  # Gradiente per regressione lineare
                grad = compute_gradient_linreg(self.weights[:, i - 1], X[:, i], y[i])
            elif cost_funct == 'logloss':  # Gradiente per regressione logistica
                grad = compute_gradient_logreg(self.weights[:, i - 1], X[:, i], y[i])
            else:
                print("Funzione di costo non supportata! Usa 'mse' o 'logloss'.")
                return None

            updated_lr = self.update_lr(i)  # Aggiorna il learning rate
            self.weights[:, i] = self.weights[:, i - 1] - updated_lr * grad  # Aggiorna i pesi

        return self.weights


class Classifier_Logistic:
    def __init__(self, n_features, constant_step_size=True, lr=0.01):
        """
        Inizializza il classificatore basato su regressione logistica con SGD.
        Parametri:
        - n_features (int): Numero di feature del dataset.
        - constant_step_size (bool): Se True, il learning rate è costante; altrimenti decresce nel tempo.
        - lr (float): Learning rate iniziale.
        """
        self.n_features = n_features
        self.sgd_logreg = StochasticGradientDescent(learning_rate=lr, constant_step_size=constant_step_size)
        self.weights = None

    def fit(self, n_samples, mu_a, mu_b, sigma_matrix, p_pos, p_neg, mc_iteration):
        """
        Allena il modello utilizzando SGD e regressione logistica.
        Parametri:
        - n_samples (int): Numero di campioni da generare per ogni iterazione.
        - mu_pos (float): Media della distribuzione per la classe positiva.
        - mu_neg (float): Media della distribuzione per la classe negativa.
        - sigma (float): Deviazione standard delle distribuzioni.
        - p_pos (float): Probabilità a priori della classe positiva.
        - p_neg (float): Probabilità a priori della classe negativa.
        - mc_iteration (int): Numero di iterazioni Monte Carlo per stabilizzare i pesi.
        """
        weights_logreg_mc = np.zeros((self.n_features + 1, n_samples))  # Inizializza la matrice dei pesi (+1 per il B0)

        # Calcolo dei pesi (usando Monte Carlo)
        for _ in range(mc_iteration):
            X, y = generate_data(mu_a, mu_b, p_pos, p_neg, sigma_matrix, n_samples)
            weights_logreg = self.sgd_logreg.run_sgd(X, y, 'logloss')  # Training con SGD
            weights_logreg_mc += weights_logreg  # Accumula i pesi ottenuti
        weights_logreg_mc /= mc_iteration  # Media dei pesi sulle iterazioni Monte Carlo

        # Plot dell'evoluzione dei pesi
        plt.figure()
        plt.plot(weights_logreg_mc[0, :], label=r"$\beta_1$")
        plt.plot(weights_logreg_mc[1, :], label=r"$\beta_2$")
        plt.plot(weights_logreg_mc[2, :], label=r"$\beta_0$")
        plt.xlabel("Iterazioni")
        plt.ylabel("Evoluzione dei pesi")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.weights = weights_logreg_mc[:, -1].reshape(-1, 1)  # Salva gli ultimi pesi ottenuti

    def predict(self, x_test, y_test=None):
        """
        Effettua le predizioni sui dati di test utilizzando la funzione sigmoide.
        Parametri:
        - x_test (numpy.ndarray): Dati di test.
        - y_test (numpy.ndarray, opzionale): Etichette corrispondenti dai dati di test (per il calcolo dell'accuratezza).

        Ritorna:
        - predictions (numpy.ndarray): Etichette predette.
        - accuracy (float, opzionale): Accuratezza del modello.
        """
        accuracy = None
        # Calcolo del prodotto scalare tra i pesi e i dati di test
        z = np.dot(np.transpose(self.weights), x_test)  # Trasponi x_test se ha forma (3, 2000)

        # Calcolo della probabilità 
        probabilities = 1 / (1 + np.exp(-z))

        # Assegnazione della classe +1 se la probabilità è > 0.5, -1 se la probabilità è < 0.5, altrimenti sceglie a caso
        predictions = np.where(probabilities > 0.5, 1,np.where(probabilities < 0.5, -1, np.random.choice([1, -1])))

        if y_test is not None:
            accuracy = np.mean(predictions == y_test)

        return np.array(predictions), accuracy