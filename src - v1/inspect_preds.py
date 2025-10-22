# src/inspect_preds.py v2
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False

from sklearn.metrics import confusion_matrix, classification_report

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
PRED_PATH = os.path.join(ARTIFACTS_DIR, "walkforward_predictions_balanced.csv")


def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    if _HAVE_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="k")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(
            f"Prediction file not found: {PRED_PATH}\n"
            f"Run walkforward_eval_balanced.py first."
        )

    df = pd.read_csv(PRED_PATH)

    print("=== HEAD (first 25 rows) ===")
    print(df.head(25).to_string())

    # --- Trend evaluation ---
    print("\n=== Trend Value Counts ===")
    print("True Trend Labels:")
    print(df["true_trend"].value_counts().to_string())
    print("Predicted Trend Labels:")
    print(df["pred_trend"].value_counts().to_string())

    print("\n=== Trend Confusion Matrix ===")
    print(pd.crosstab(df["true_trend"], df["pred_trend"]).to_string())
    try:
        plot_confusion(df["true_trend"], df["pred_trend"], labels=[-1, 0, 1],
                       title="Confusion Matrix (Trend)")
    except Exception as e:
        print("Skipped plotting trend confusion (plotting error):", e)

    print("\n=== Classification Report (Trend) ===")
    print(classification_report(df["true_trend"], df["pred_trend"], digits=4))

    # --- Strength evaluation (3 bins) ---
    print("\n=== Strength Value Counts (3 bins: 0=Weak, 1=Medium, 2=Strong) ===")
    print("True Strength Labels:")
    print(df["true_strength_bin"].value_counts().to_string())
    print("Predicted Strength Labels:")
    print(df["pred_strength_bin"].value_counts().to_string())

    print("\n=== Strength Confusion Matrix (3 bins) ===")
    print(pd.crosstab(df["true_strength_bin"], df["pred_strength_bin"]).to_string())
    try:
        plot_confusion(df["true_strength_bin"], df["pred_strength_bin"],
                       labels=[0, 1, 2],
                       title="Confusion Matrix (Strength: Weak / Medium / Strong)")
    except Exception as e:
        print("Skipped plotting strength confusion (plotting error):", e)

    print("\n=== Classification Report (Strength, 3 bins) ===")
    print(classification_report(df["true_strength_bin"], df["pred_strength_bin"], digits=4))

    # --- Blended categories ---
    print("\n=== Blended Category Value Counts ===")
    print(df["blended"].value_counts().to_string())


if __name__ == "__main__":
    main()
