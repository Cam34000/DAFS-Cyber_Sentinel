import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

#calculates ratio of normal to anomaly as to give more weight to anomalies
#Calculer le poids des anomalies selon la proportion des deux classes.
def _scale_pos_weight(y):
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos


def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Select the number of trees using an internal validation set,
    then refit a complete pipeline on all training data.
    """
    # 1. Reserve validation data within the training set.
    # Réserver 20 % des données d'entraînement pour choisir le nombre d'arbres.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=seed,
        stratify=y_train,
    )

    # 2. Learn preprocessing only on the fitting subset.
    # Apprendre les transformations sur l'apprentissage uniquement,
    # puis appliquer les mêmes règles à la validation.
    validation_preproc = clone(preproc)
    X_fit_transformed = validation_preproc.fit_transform(X_fit, y_fit)
    X_valid_transformed = validation_preproc.transform(X_valid)

    params = {
        "max_depth": 4,         #Limite la profondeur de chaque arbre
        "learning_rate": 0.08,  #Limite la contribution de chaque nouvel arbre
        "subsample": 0.9,       #Utilise une fraction des lignes pour chaque étape
        "colsample_bytree": 0.9,#Utilise une fraction des variables pour chaque arbre
        "reg_lambda": 1.0,      #Pénalisent la complexité du modèle
        "reg_alpha": 0.0,       
        "random_state": seed,   #Fixe les choix aléatoires
        "n_jobs": -1,           #Utilise tous les threads disponibles
        "eval_metric": "logloss",#Évalue la qualité des probabilités ; plus bas est meilleur
        "tree_method": "hist",  #Utilise une méthode de construction par histogrammes
        "verbosity": 0,         #Réduit les messages affichés
    }

    # 3. Stop after 20 rounds without validation improvement.
    # Autoriser jusqu'à 300 arbres et arrêter après 20 étapes
    validation_model = XGBClassifier(
        **params,
        n_estimators=300,
        early_stopping_rounds=20,
        scale_pos_weight=_scale_pos_weight(y_fit),
    )

    # Entraîner sur l'apprentissage et surveiller le score sur la validation.
    validation_model.fit(
        X_fit_transformed,
        y_fit,
        eval_set=[(X_valid_transformed, y_valid)],
        verbose=False,
    )

    # Iterations start at zero, so add one for the tree count.
    # Retenir le nombre d'arbres associé au meilleur score de validation.
    best_n_estimators = validation_model.best_iteration + 1
    print(f"Number of trees selected: {best_n_estimators}")

    # 4. Refit on all training data with the selected tree count.
    # Créer le modèle final avec le nombre d'arbres choisi par validation.
    final_model = XGBClassifier(
        **params,
        n_estimators=best_n_estimators,
        scale_pos_weight=_scale_pos_weight(y_train),
    )
    # Réentraîner le préprocesseur et le modèle sur toutes les données
    # d'entraînement, puis renvoyer le pipeline prêt à prédire.
    pipe = Pipeline([
        ("preproc", clone(preproc)),
        ("model", final_model),
    ])
    pipe.fit(X_train, y_train)

    return pipe