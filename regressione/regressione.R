#### Lettura e divisione del dataset ####
# Lettura del dataset
dataset = read.csv("RegressionExam250218.csv",header=T,na.strings ="?") # non legge l'intestazione
dataset = na.omit(dataset) # cancella le righe che hanno dei valori mancanti
n = dim(dataset) # dim dataset (senza le righe con valori mancanti)
X = dataset$X
y = dataset$Y # variabile dipendente

# Creazione della cartella per le immagini
if (!dir.exists("img")) {
  dir.create("img")
}


#### PUNTO 1 ####

# Creazione di una matrice con le potenze di X fino al grado 10
x <- sapply(1:10, function(p) X^p)
colnames(x) <- paste0("X", 1:10) # aggiunge i nomi alle colonne

# Creazione di un nuovo dataset con le nuove X e la variabile Y
dataset <- data.frame(Y=y, x)

# Divisione del dataset in train e test
set.seed(2025)
train = sample(1:nrow(x), 0.75*nrow(x))  # indici per la divisione del dataset
test = (-train)
y.test = y[test]
d_train_complete = dataset[train,] # train: 75% dei dati
d_test_complete = dataset[test,] # test: 25% dei dati
attach(d_train_complete)

# Inizializza un vettore per memorizzare gli errori di predizione sul test set
test_errors <- numeric(10)

# Itera sui polinomi di grado da 1 a 10
for (p in 1:10) {
  # Costruisce la formula della regressione polinomiale includendo tutte le potenze fino a X^p
  formula_poly <- as.formula(paste("Y ~", paste(colnames(x)[1:p], collapse = " + ")))
  
  # Addestra il modello di regressione polinomiale sui dati di training
  model <- lm(formula_poly, data = d_train_complete)
  
  # Effettua le previsioni sui dati di test
  predictions <- predict(model, newdata = d_test_complete)
  
  # Calcola l'errore quadratico medio (MSE) tra le predizioni e i valori reali del test set
  test_errors[p] <- mean((y.test - predictions)^2)
}

# Genera un grafico che mostra l'errore quadratico medio in funzione del grado del polinomio
library(ggplot2)
dev.new()
ggplot(data.frame(Grado = 1:10, MSE = test_errors), aes(x = Grado, y = MSE)) +
  geom_line(color = "blue") +  
  geom_point(color = "red", size = 3) +
  labs(title = "Errore di predizione in funzione del grado del polinomio", x = "Grado del polinomio", y = "MSE (Errore Quadratico Medio)") +
  theme_minimal()
dev.print(device = jpeg, "img/error_plot.jpg", width = 800, height = 800)

# Identifica il grado del polinomio che minimizza l'errore di predizione
best_p <- which.min(test_errors)
print(paste("Il miglior grado del polinomio è:", best_p))
# Il miglior grado del polinomio è: 4


#### PUNTO 2 ####

# Ridimensionamento del dataset
d_train = dataset[train, 1:5]  # contiene solo le prime 5 colonne di d_train (fino a X4)
d_test = dataset[test, 1:5]    # contiene solo le prime 5 colonne di d_test (fino a X4)
attach(d_train)

## Backward Stepwise Selection ##
startmod=lm(Y~.,data=d_train) # start: modello con tutti i regressori (.) 
scopmod=lm(Y~1,data=d_train) # scop: modello costante (1)

# Costruisce il modello ottimo basato su BIC con la direzione specificata
optmodBIC <- step(startmod,direction = "backward", scope=formula(scopmod), k=log(n[1])) # bisogna specificare il k

# Restituisce il numero di parametri del modello selezionato (inclusa l'intercetta) e il valore del BIC del modello finale.
extractAIC(optmodBIC, k=log(n[1])) 
summary(optmodBIC)

# Estrazione dei coefficienti del modello (optmodBIC)
coefficients = coef(optmodBIC)

#### PUNTO 3 ####

# plot dei dati
dev.new()
plot(X, y, pch = 19, col = "black", main = "Regressione Polinomiale", xlab = "X", ylab = "Y") 

# calcolo e plot della funzione di regressione
x_seq <- seq(min(X), max(X), along.with = X)
y_pred <- coefficients[1] + coefficients[2] * x_seq^2 + coefficients[3] * x_seq^4
lines(x_seq, y_pred, col ="red",lwd=2)
dev.print(device=pdf, "img/regression_function.pdf")

# calcolo e plot dell'intervallo di confidenza e dell'intervallo di predizione
xx <- data.frame(X2 = x_seq^2, X4 = x_seq^4)
ci_lin <- predict(optmodBIC,xx,se.fit = T,interval = "confidence")
matplot(x_seq,ci_lin$fit[,2],lty=2,lwd=2,col="red", type="l", add=T)
matplot(x_seq,ci_lin$fit[,3],lty=2,lwd=2,col="red", type="l", add=T)
pi_lin <- predict(optmodBIC,xx,se.fit = T,interval = "prediction")
matplot(x_seq,pi_lin$fit[,2],lty=2,lwd=2,col="green", type="l", add=T)
matplot(x_seq,pi_lin$fit[,3],lty=2,lwd=2,col="green", type="l", add=T)
legend('topright', c('data','regr. line','0.95 conf. bound',NA,'0.95 pred. bound',NA), lty=c(NA, 1,2,NA,2,NA), col=c('black','red','red','red','green','green'), pch=c(1,NA,NA,NA,NA,NA), lwd=c(NA,4,2,2,2,2), cex=.9)
dev.print(device=pdf, "img/conf_pred_int.pdf")

#### PUNTO 4 ####
dev.new()
par(mfrow=c(2,2))
plot(optmodBIC)