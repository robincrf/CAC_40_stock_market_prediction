import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from helpers import csv_file as cf

df = cf.loadcsv()

df["return"] = df["AdjClose"].pct_change() # rendement
df["MA5"] = df["AdjClose"].rolling(5).mean() # moyenne 5 derniers jours
df["MA10"] = df["AdjClose"].rolling(10).mean() # moyenne 10 derniers jours
df["Vol5"] = df["return"].rolling(5).std() # écart type 5 jours
df["Vol10"] = df["return"].rolling(10).std() # écart type 10 jours
df["DeltaVol"] = df["Volume"].pct_change() # variation du volume d'un jour à l'autre
df["RatioPriceMA10"] = df["AdjClose"] / df["MA10"]


df["target"] = df["AdjClose"].shift(-1)

df = df.dropna()
X = df[["return","MA5","MA10","Vol5","Vol10","DeltaVol","RatioPriceMA10"]]
Y = df["target"]

train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

model = KMeans(n_clusters=8, init="k-means++" ,max_iter=300)
model.fit(X_train,Y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(Y_test, pred)
r2 = r2_score(Y_test, pred)
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")



def display_rf():
    plt.figure(figsize=(10,5))
    plt.plot(Y_test.index, Y_test, label="Réel", color="blue")
    plt.plot(Y_test.index, pred, label="Prédit", color="orange")
    plt.title("Prédiction du prix (KMEANS) ")
    plt.legend()
    plt.show()

display_rf()




