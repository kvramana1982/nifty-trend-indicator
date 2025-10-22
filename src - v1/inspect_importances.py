# src/inspect_importances.py  (v2)
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# === Paths ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TREND_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_trend.joblib")
STRENGTH_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_strength.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")

FI_TREND_PATH = os.path.join(ARTIFACTS_DIR, "feature_importances_trend_single.csv")
FI_STRENGTH_PATH = os.path.join(ARTIFACTS_DIR, "feature_importances_strength_3bin_single.csv")


def plot_importances(df, title, top_n=20):
    """Plot top-n feature importances as horizontal bar chart."""
    df.head(top_n).plot(kind="barh", x="feature", y="importance",
                        legend=False, figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def save_and_display_importances(model, feature_cols, csv_path, title):
    """Save feature importances to CSV and print top 30."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"Model {title} has no feature_importances_ attribute")

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importances.to_csv(csv_path, index=False)
    print(f"\n=== Top 30 {title} Feature Importances ===")
    print(importances.head(30).to_string(index=False))

    try:
        plot_importances(importances, f"Top 20 {title} Feature Importances")
    except Exception as e:
        print(f"Skipped plotting {title} importances:", e)


def main():
    if not (os.path.exists(TREND_MODEL_PATH) and os.path.exists(STRENGTH_MODEL_PATH) and os.path.exists(FEATURES_PATH)):
        raise FileNotFoundError(
            "Missing required models or feature list. "
            "Run train.py or predict.py first to generate models."
        )

    # Load models and feature list
    clf_trend = joblib.load(TREND_MODEL_PATH)
    clf_strength = joblib.load(STRENGTH_MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    # Save + show importances
    save_and_display_importances(clf_trend, feature_cols, FI_TREND_PATH, "Trend")
    save_and_display_importances(clf_strength, feature_cols, FI_STRENGTH_PATH, "Strength (3-bin)")


if __name__ == "__main__":
    main()
