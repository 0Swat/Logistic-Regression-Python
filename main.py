import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_log_likelihood(X, Y, B):
    log_likelihood = 0
    for i in range(len(Y)):
        logit = np.dot(X[i], B.T)
        log_likelihood += Y[i] * np.log(sigmoid(logit)) + (1 - Y[i]) * np.log(1 - sigmoid(logit))
    return log_likelihood

def logistic_regression(X, Y, alfa, num_iterations=100):
    B = np.zeros((1, X.shape[1]))

    for iteration in range(num_iterations):
        for i in range(len(Y)):
            logit = np.dot(X[i], B.T)
            error = sigmoid(logit) - Y[i]
            gradient = X[i] * error 
            B -= alfa * gradient

        print(f"Iteracja {iteration+1}, Wartości B: {B}, Log likelihood: {compute_log_likelihood(X, Y, B)}")

    return B

data = pd.read_csv("haberman.csv", header=0)  
X = data.iloc[:, :-1].astype(float).values  
Y = data.iloc[:, -1].values

Y = np.where(Y == 2, 0, Y).astype(float)  
alfa = 0.01

B = logistic_regression(X, Y, alfa)

print("Finalne wartości B:", B)



data = pd.read_csv('haberman.csv')

X = data.iloc[:, [0, 1, 2]] 
y = data.iloc[:, 3]         
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)

print(X_test)

model = LogisticRegression(C=30, max_iter=150)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


cm = confusion_matrix(y_test, predictions)

print(cm)

plt.figure(figsize = (5,4))
sn.heatmap(cm, annot=True)
plt.xlabel('Przewidziana wartość')
plt.ylabel('Rzeczywista wartość')
plt.show()
