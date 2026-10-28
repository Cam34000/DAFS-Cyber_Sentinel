import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
)

def evaluate(model, X_test, y_test) -> dict:
    """
    Calcule les métriques de classification binaire (anomaly = classe positive).

    Ajouts: 
    1. Métriques par classe: Précision/recall/f1 pour normal et pour anomaly.
    2. Calibration: Score de Brier (0 = probabilités parfaites).
    3. Recherche de seuil: Seuil qui maximise le F1, et seuil le plus élevé qui
       garde au moins 99% de recall.

    Retourne au dictionnaire (sauvegardé dans outputs/metrics.json).
    """
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some estimators only have decision_function
        if hasattr(model.named_steps["model"], "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "positives_test": int(np.sum(y_test == 1)),
        "negatives_test": int(np.sum(y_test == 0)),
    }
    p_cls = precision_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
    r_cls = recall_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
    f_cls = f1_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)

    metrics["per_class"] = {
        "normal": {"precision": float(p_cls[0]), "recall": float(r_cls[0]), "f1": float(f_cls[0])},
        "anomaly": {"precision": float(p_cls[1]), "recall": float(r_cls[1]), "f1": float(f_cls[1])}
    }
    print("Per_class: ", metrics["per_class"])

    metrics["brier_score"] = float(brier_score_loss(y_test, y_proba))
    print("Brier score: ", metrics["brier_score"])

    prec, rec, seuils = precision_recall_curve(y_test, y_proba)
    f1_seuils = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    i_best = int(np.argmax(f1_seuils))
    metrics["best_thresold_f1"] = {
        "threshold": float(seuils[i_best]),
        "f1": float(f1_seuils[i_best])
    }
    ok = np.where(rec[:-1] >= 0.99)[0]
    if len(ok) > 0:
        j = int(ok[-1])
        metrics["threshold_recall_99"] = {
            "threshold": float(seuils[j]),
            "precision": float(prec[j]),
            "recall": float(rec[j])
        }
    else:
        metrics["threshold_recall_99"] = None
    print("Best F1 seuil: ", metrics["best_thresold_f1"])
    print("Recall seuil >= 99%: ", metrics["threshold_recall_99"])
    return metrics