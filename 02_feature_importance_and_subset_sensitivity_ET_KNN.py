# Supplementary Material 1 - Part 2
# Feature importance and feature subset sensitivity analysis for the ET + KNN hierarchical model

import os
import re
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


SEED = 42
TEST_SIZE = 0.25

DATA_PATH = r"<PATH_TO_TRAINING_DATA>/training_samples.csv"
OUTPUT_DIR = r"<PATH_TO_OUTPUT_DIRECTORY>/part2_feature_sensitivity"

LABEL_COLUMN = "type"
SELECTED_K = 28

ET_PARAMS = {}
KNN_PARAMS = {}

FOREST_TYPES = {
    "Evergreen broadleaved forest",
    "Deciduous broadleaved forest",
    "Evergreen needleaved forest",
    "Deciduous needleaved forest",
    "Mixed forest",
}

CLASS_ORDER_6 = [
    "Evergreen broadleaved forest",
    "Deciduous broadleaved forest",
    "Evergreen needleaved forest",
    "Deciduous needleaved forest",
    "Mixed forest",
    "other",
]

FOREST_ORDER_5 = [
    "Evergreen broadleaved forest",
    "Deciduous broadleaved forest",
    "Evergreen needleaved forest",
    "Deciduous needleaved forest",
    "Mixed forest",
]


def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)


def map_to_6class(label):
    return label if label in FOREST_TYPES else "other"


def get_feature_columns(df):
    feature_cols = [col for col in df.columns if re.fullmatch(r"A\d{2}", str(col))]
    if len(feature_cols) == 0:
        raise ValueError("No AEF feature columns were found. Expected columns named A00-A63.")
    return sorted(feature_cols)


def load_training_data(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    feature_cols = get_feature_columns(df)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label column '{LABEL_COLUMN}' was not found.")

    data = df[feature_cols + [LABEL_COLUMN]].copy()
    data[LABEL_COLUMN] = data[LABEL_COLUMN].astype(str).str.strip()
    data = data.dropna(axis=0).reset_index(drop=True)

    data["label_6class"] = data[LABEL_COLUMN].apply(map_to_6class)
    data["label_binary"] = data[LABEL_COLUMN].apply(lambda x: 1 if x in FOREST_TYPES else 0)
    data["label_forest5"] = data[LABEL_COLUMN].where(data[LABEL_COLUMN].isin(FOREST_TYPES), np.nan)

    return data, feature_cols


def prepare_split(data):
    encoder_5 = LabelEncoder()
    encoder_5.fit(FOREST_ORDER_5)

    encoder_6 = LabelEncoder()
    encoder_6.fit(CLASS_ORDER_6)

    train_idx, val_idx = train_test_split(
        np.arange(len(data)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=data["label_6class"].astype(str).values,
    )

    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)

    train_forest_df = train_df[train_df[LABEL_COLUMN].isin(FOREST_TYPES)].copy()
    val_forest_df = val_df[val_df[LABEL_COLUMN].isin(FOREST_TYPES)].copy()

    return train_df, val_df, train_forest_df, val_forest_df, encoder_5, encoder_6


def rank_features_by_et(X_train, y_train, feature_cols):
    model = ExtraTreesClassifier(random_state=SEED, **ET_PARAMS)
    model.fit(X_train, y_train)

    ranking = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return ranking, model


def compute_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_OA": accuracy_score(y_true, y_pred),
        f"{prefix}_Kappa": cohen_kappa_score(y_true, y_pred),
        f"{prefix}_Macro_F1": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
    }


def fit_et_classifier(X_train, y_train):
    model = ExtraTreesClassifier(random_state=SEED, **ET_PARAMS)
    model.fit(X_train, y_train)
    return model


def fit_knn_classifier(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(**KNN_PARAMS)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def run_subset_sensitivity(
    train_df,
    val_df,
    train_forest_df,
    val_forest_df,
    feature_cols,
    encoder_5,
    encoder_6,
    step1_ranked_features,
    step2_ranked_features,
):
    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_cols)}

    X_train_step1_all = train_df[feature_cols].values
    X_val_step1_all = val_df[feature_cols].values
    y_train_step1 = train_df["label_binary"].values
    y_val_step1 = val_df["label_binary"].values

    X_train_step2_all = train_forest_df[feature_cols].values
    X_val_step2_all = val_forest_df[feature_cols].values
    y_train_step2 = encoder_5.transform(train_forest_df[LABEL_COLUMN].astype(str).values)
    y_val_step2 = encoder_5.transform(val_forest_df[LABEL_COLUMN].astype(str).values)

    y_true_final = encoder_6.transform(val_df["label_6class"].astype(str).values)

    results = []

    for k in range(1, len(feature_cols) + 1):
        step1_features = step1_ranked_features[:k]
        step2_features = step2_ranked_features[:k]

        step1_indices = [feature_to_idx[feature] for feature in step1_features]
        step2_indices = [feature_to_idx[feature] for feature in step2_features]

        X_train_step1 = X_train_step1_all[:, step1_indices]
        X_val_step1 = X_val_step1_all[:, step1_indices]

        step1_model = fit_et_classifier(X_train_step1, y_train_step1)
        y_pred_step1 = step1_model.predict(X_val_step1)

        step1_metrics = {
            "Step1_OA": accuracy_score(y_val_step1, y_pred_step1),
            "Step1_Kappa": cohen_kappa_score(y_val_step1, y_pred_step1),
            "Step1_F1": f1_score(y_val_step1, y_pred_step1, average="binary"),
        }

        X_train_step2 = X_train_step2_all[:, step2_indices]
        X_val_step2 = X_val_step2_all[:, step2_indices]

        step2_model, step2_scaler = fit_knn_classifier(X_train_step2, y_train_step2)
        y_pred_step2 = step2_model.predict(step2_scaler.transform(X_val_step2))

        step2_metrics = compute_metrics(y_val_step2, y_pred_step2, "Step2")

        y_pred_final_raw = np.array(["other"] * len(val_df), dtype=object)
        forest_mask = y_pred_step1 == 1

        if forest_mask.sum() > 0:
            X_val_for_step2 = val_df.loc[forest_mask, feature_cols].values[:, step2_indices]
            X_val_for_step2_scaled = step2_scaler.transform(X_val_for_step2)
            y_pred_step2_final = step2_model.predict(X_val_for_step2_scaled)
            y_pred_final_raw[forest_mask] = encoder_5.inverse_transform(y_pred_step2_final)

        y_pred_final = encoder_6.transform(y_pred_final_raw)
        final_metrics = compute_metrics(y_true_final, y_pred_final, "Final")

        results.append({
            "k": k,
            **step1_metrics,
            **step2_metrics,
            **final_metrics,
            "Step1_TopK_Features": ",".join(step1_features),
            "Step2_TopK_Features": ",".join(step2_features),
        })

        print(f"k={k:02d}, Final_OA={final_metrics['Final_OA']:.4f}")

    return pd.DataFrame(results)


def train_selected_models(
    train_df,
    feature_cols,
    encoder_5,
    encoder_6,
    step1_ranked_features,
    step2_ranked_features,
    selected_k=SELECTED_K,
):
    step1_features = step1_ranked_features[:selected_k]
    step2_features = step2_ranked_features[:selected_k]

    train_forest_df = train_df[train_df[LABEL_COLUMN].isin(FOREST_TYPES)].copy()

    X_train_step1 = train_df[step1_features].values
    y_train_step1 = train_df["label_binary"].values

    X_train_step2 = train_forest_df[step2_features].values
    y_train_step2 = encoder_5.transform(train_forest_df[LABEL_COLUMN].astype(str).values)

    step1_model = fit_et_classifier(X_train_step1, y_train_step1)
    step2_model, step2_scaler = fit_knn_classifier(X_train_step2, y_train_step2)

    return {
        "selected_k": selected_k,
        "step1_model": step1_model,
        "step2_model": step2_model,
        "step2_scaler": step2_scaler,
        "step1_features": step1_features,
        "step2_features": step2_features,
        "encoder_5": encoder_5,
        "encoder_6": encoder_6,
        "forest_types": sorted(list(FOREST_TYPES)),
        "class_order_6": CLASS_ORDER_6,
        "forest_order_5": FOREST_ORDER_5,
    }


def plot_sensitivity(results_df, output_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(results_df["k"], results_df["Final_OA"], marker="o", linewidth=1.4, label="Final OA")
    plt.plot(results_df["k"], results_df["Final_Macro_F1"], marker="s", linewidth=1.4, label="Final Macro-F1")
    plt.xlabel("Number of top-k features")
    plt.ylabel("Metric value")
    plt.title("ET + KNN feature subset sensitivity analysis")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ET_KNN_feature_subset_sensitivity_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(results_df["k"], results_df["Step1_OA"], marker="o", linewidth=1.4, label="Step1 OA")
    plt.plot(results_df["k"], results_df["Step2_OA"], marker="s", linewidth=1.4, label="Step2 OA")
    plt.xlabel("Number of top-k features")
    plt.ylabel("Overall accuracy")
    plt.title("ET + KNN step-wise accuracy by feature subset size")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ET_KNN_stepwise_accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data, feature_cols = load_training_data(DATA_PATH)
    train_df, val_df, train_forest_df, val_forest_df, encoder_5, encoder_6 = prepare_split(data)

    X_train_step1 = train_df[feature_cols].values
    y_train_step1 = train_df["label_binary"].values

    X_train_step2 = train_forest_df[feature_cols].values
    y_train_step2 = encoder_5.transform(train_forest_df[LABEL_COLUMN].astype(str).values)

    step1_ranking, _ = rank_features_by_et(X_train_step1, y_train_step1, feature_cols)
    step2_ranking, _ = rank_features_by_et(X_train_step2, y_train_step2, feature_cols)

    step1_ranking.to_csv(
        os.path.join(OUTPUT_DIR, "ET_feature_ranking_step1_binary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    step2_ranking.to_csv(
        os.path.join(OUTPUT_DIR, "ET_surrogate_feature_ranking_step2_forest5_for_KNN.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    step1_ranked_features = step1_ranking["feature"].tolist()
    step2_ranked_features = step2_ranking["feature"].tolist()

    results_df = run_subset_sensitivity(
        train_df,
        val_df,
        train_forest_df,
        val_forest_df,
        feature_cols,
        encoder_5,
        encoder_6,
        step1_ranked_features,
        step2_ranked_features,
    )

    sensitivity_path = os.path.join(OUTPUT_DIR, "ET_KNN_feature_subset_sensitivity_analysis.csv")
    results_df.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")

    results_df.sort_values("Final_OA", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "ET_KNN_feature_subset_ranked_by_Final_OA.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    results_df.sort_values("Final_Macro_F1", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "ET_KNN_feature_subset_ranked_by_Final_Macro_F1.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    selected_models = train_selected_models(
        train_df,
        feature_cols,
        encoder_5,
        encoder_6,
        step1_ranked_features,
        step2_ranked_features,
        selected_k=SELECTED_K,
    )

    joblib.dump(
        selected_models,
        os.path.join(OUTPUT_DIR, f"ET_KNN_top{SELECTED_K}_models.joblib"),
    )

    selected_features_df = pd.DataFrame({
        "rank": np.arange(1, SELECTED_K + 1),
        "step1_et_features": selected_models["step1_features"],
        "step2_knn_features": selected_models["step2_features"],
    })
    selected_features_df.to_csv(
        os.path.join(OUTPUT_DIR, f"ET_KNN_top{SELECTED_K}_selected_features.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_sensitivity(results_df, OUTPUT_DIR)

    best_row = results_df.loc[results_df["Final_OA"].idxmax()]
    print("\nBest feature subset by Final OA:")
    print(best_row[["k", "Final_OA", "Final_Kappa", "Final_Macro_F1", "Final_Weighted_F1"]])
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
