import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Charger ton CSV ---
from helpers import csv_file as cf
df = cf.loadcsv()

# --- Préparation des features enrichies ---
# Rendements
df["return"] = df["AdjClose"].pct_change()
df["return_2d"] = df["AdjClose"].pct_change(2)
df["return_5d"] = df["AdjClose"].pct_change(5)

# Moyennes mobiles
df["MA5"] = df["AdjClose"].rolling(5).mean()
df["MA10"] = df["AdjClose"].rolling(10).mean()
df["MA20"] = df["AdjClose"].rolling(20).mean()

# Volatilité
df["Vol5"] = df["return"].rolling(5).std()
df["Vol10"] = df["return"].rolling(10).std()
df["Vol20"] = df["return"].rolling(20).std()

# Volume
df["DeltaVol"] = df["Volume"].pct_change()
df["VolMA5"] = df["Volume"].rolling(5).mean()
df["RatioVolMA5"] = df["Volume"] / df["VolMA5"]

# Ratios de prix
df["RatioPriceMA10"] = df["AdjClose"] / df["MA10"]
df["RatioPriceMA20"] = df["AdjClose"] / df["MA20"]

# RSI (Relative Strength Index)
delta = df["AdjClose"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
exp12 = df["AdjClose"].ewm(span=12).mean()
exp26 = df["AdjClose"].ewm(span=26).mean()
df["MACD"] = exp12 - exp26
df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

# Momentum
df["Momentum"] = df["AdjClose"] - df["AdjClose"].shift(10)

# --- Cible : prix du lendemain ---
df["target"] = df["AdjClose"].shift(-1)

df = df.dropna()

# Sélection des features
features = ["return", "return_2d", "return_5d", 
            "MA5", "MA10", "MA20",
            "Vol5", "Vol10", "Vol20",
            "DeltaVol", "RatioVolMA5",
            "RatioPriceMA10", "RatioPriceMA20",
            "RSI", "MACD", "MACD_signal", "Momentum"]

X = df[features]
y = df["target"]

# --- Split temporel ---
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Random Forest optimisé ---
model = RandomForestRegressor(
    n_estimators=400,      # Plus d'arbres
    max_depth=12,          # Plus de profondeur
    min_samples_split=5,   # Contrôle du surapprentissage
    min_samples_leaf=2,
    max_features='sqrt',   # Diversité des arbres
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# --- Évaluation ---
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# --- Importance des features ---
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportance des variables :")
print(importances)

# --- Graphique ---
def display_rf():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Graphique des prédictions
    ax1.plot(y_test.index, y_test.values, label="Réel", color="blue", linewidth=2)
    ax1.plot(y_test.index, pred, label="Prédit", color="orange", linewidth=2, alpha=0.7)
    ax1.set_title("Prédiction du prix (Random Forest)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Prix (€)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique de l'erreur
    error = y_test.values - pred
    ax2.plot(y_test.index, error, label="Erreur", color="red", linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_title("Erreur de prédiction", fontsize=12)
    ax2.set_ylabel("Erreur (€)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

display_rf()