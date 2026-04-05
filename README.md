# Simulation et Analyse de Cours d’Actions — Projet FinanceJR

## Objectif du projet

Ce projet a pour but de simuler l’évolution du cours d’une action réelle (par exemple une société du CAC 40) à partir de ses données historiques, puis de comparer les résultats simulés à la trajectoire réelle.

L’objectif est de comprendre comment modéliser un prix d’action à l’aide d’outils statistiques simples issus de la finance quantitative, et d’évaluer la qualité des simulations.

---

## Concepts clés

- Rendements logarithmiques : mesure de la variation relative du prix d’un jour à l’autre.  
- Drift (tendance moyenne) : moyenne des rendements journaliers.  
- Volatilité : écart-type des rendements, reflétant la variabilité du marché.  
- Mouvement brownien géométrique (GBM) : modèle simple utilisé pour simuler l’évolution des prix.  
- Comparaison simulation / réalité : via des indicateurs d’erreur (RMSE, MAPE, couverture de bande).

---

## Méthodologie

1. **Téléchargement des données**  
   Utilisation de la bibliothèque `yfinance` pour récupérer les cours historiques (prix de clôture ajustés).

2. **Préparation**  
   Calcul des rendements logarithmiques à partir des prix.  
   Estimation du drift (moyenne) et de la volatilité (écart-type).

3. **Simulation**  
   Génération de plusieurs trajectoires futures du cours avec le modèle de Mouvement Brownien Géométrique :  

4. **Comparaison**  
   Comparaison des cours simulés et des cours réels sur la même période.  
   Calcul de métriques :
   - RMSE (erreur quadratique moyenne)
   - MAPE (erreur moyenne en pourcentage)
   - Coverage (taux de présence du réel dans la bande de simulation)

---

## Résultats attendus

- Visualisation : courbe du prix réel vs médiane des simulations + bandes de confiance (10–90 %).  
- Tableau de performance des prévisions :

  | Indicateur | Description | Interprétation |
  |-------------|-------------|----------------|
  | RMSE | Erreur moyenne entre prix réel et simulé | Plus petit = meilleur |
  | MAPE | Erreur moyenne en % | Plus petit = meilleur |
  | Coverage | % du temps où le réel tombe dans la bande simulée | Proche de 80–90 % = bon calibrage |

---

## Technologies utilisées

- Python 3.11+
- Bibliothèques principales :
  - pandas — manipulation de données
  - numpy — calculs numériques
  - matplotlib — visualisation
  - yfinance — récupération des données de marché
  - scipy — outils statistiques (si besoin)



