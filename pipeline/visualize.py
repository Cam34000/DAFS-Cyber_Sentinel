import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
)

def _ensure(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def make_plots(model, X_test, y_test, df_full, outdir: str):
    """
    Save PNGs: confusion matrix, ROC, PR curve, top feature importances (if available).
    TODO (Student E): add per-flag/per-service breakdowns, error analysis, and threshold-sweep plots.
    """
    _ensure(outdir)

    # Predictions / scores
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal (0)", "anomaly (1)"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Precision–Recall curve
    try:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = np.trapz(prec[::-1], rec[::-1])  # rough area; AP is in metrics.json
        plt.figure()
        plt.plot(rec, prec, label=f"Area ≈ {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "pr_curve.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Feature importances (best-effort): try XGB feature names after preprocessing
    try:
        import numpy as np
        model_step = model.named_steps["model"]
        booster = model_step.get_booster()
        scores = booster.get_fscore()  # dict of {'f0': count, 'f1': count, ...}

        # Map fN to transformed feature indices
        # NOTE: ColumnTransformer + OHE expands columns; getting exact names is advanced.
        # As a simple proxy, plot top indices by importance:
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
        if items:
            names = [k for k,_ in items]
            vals  = [v for _,v in items]
            plt.figure(figsize=(6, max(3, len(items)*0.25)))
            sns.barplot(x=vals, y=names, orient="h")
            plt.title("Top XGBoost Split Importances (by feature index)")
            plt.xlabel("Split Count")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "xgb_feature_importance.png"), dpi=150)
            plt.close()
    except Exception:
        pass

 # Threshold sweep
    thresholds = np.arange(0.05, 1.00, 0.05)

    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_threshold = (y_proba >= threshold).astype(int)

        precisions.append(
            precision_score(y_test, y_threshold, zero_division=0)
        )
        recalls.append(
            recall_score(y_test, y_threshold, zero_division=0)
        )
        f1_scores.append(
            f1_score(y_test, y_threshold, zero_division=0)
        )

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1_scores, label="F1-score")

    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Performance by classification threshold")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "threshold_sweep.png"),
        dpi=150
    )
    plt.close()