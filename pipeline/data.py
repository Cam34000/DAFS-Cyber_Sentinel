import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
   Charge le jeu de données d'intrusion et le prépare pour le pipeline.
   Cible: colonne 'class' convertie en 0 (normal) / 1 (anormaly).
   
   Etapes:
   1. Lecture du CSV avec types imposés (dtype hint) sur les colonnes texte.
   2. Politique NA: On compte les cases vides; s'il y en a, on supprime les lignes concernées et on prévient 
      avec un warning.
   3. Validation du schéma: Colonnes attendues présentes, 42 colonnes au total, et toutes 
      les autres colonnes doivent être numériques.
   4. Normalisation de la cible en 0/1, puis retour du DataFrame.
   
   Chaque étape écrit un message de log pour suivre ce qui se passe.

   """
    dtype_hints = {col: "string" for col in EXPECTED_CATEGORICAL}
    logger.info("Types imposés: %s", dtype_hints)
    df = pd.read_csv(path, dtype=dtype_hints)

    logger.info("Fichier chargé: %d lignes, %d colonnes", df.shape[0], df.shape[1])
    n_na = int(df.isna().sum().sum())
    logger.info("Cases vides (NA) trouvées : %d", n_na)
    if n_na > 0:
        avant = len(df)
        df = df.dropna()
        logger.warning("NA: %d lignes supprimées", avant -len(df))
    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    nb_attendu = 42
    if df.shape[1] != nb_attendu:
        logger.warning("Schéma: %d colonnes trouvées, %d attendues", df.shape[1], nb_attendu)

    cols_num = [c for c in df.columns if c not in EXPECTED_CATEGORICAL and c != TARGET_COL]
    non_num = [c for c in cols_num if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise ValueError(f"CSV: colonnes non numériques inattendues: {non_num}")
    logger.info("Schéma validé: %d colonnes numériques", len(cols_num))
    # Normalize target → binary 0/1
    mapping = {"anomaly": 1, "normal": 0}
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map(mapping)
    # If already numeric, assume 0/1
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")

    return df
