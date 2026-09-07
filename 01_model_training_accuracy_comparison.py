# Supplementary Material 1 - Part 1
# Model training and accuracy comparison under different classification strategies

import os
import re
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


SEED = 42
TEST_SIZE = 0.25

DATA_PATH = r"<PATH_TO_TRAINING_DATA>/training_samples.csv"
OUTPUT_DIR = r"<PATH_TO_OUTPUT_DIRECTORY>/part1_accuracy_comparison"

LABEL_COLUMN = "type"
MODEL_NAMES = ["RF", "ET", "KNN", "MLP", "SVM", "XGB", "LGBM"]

MODEL_PARAMS = {
    "RF": {},
    "ET": {},
    "KNN": {},
    "MLP": {},
    "SVM": {},
    "XGB": {},
    "LGBM": {},
}

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
    return feature_cols


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


def build_model(model_name, task="multiclass", num_classes=None):
    model_name = model_name.upper()

    if model_name == "RF":
        return RandomForestClassifier(random_state=SEED, **MODEL_PARAMS["RF"])

    if model_name == "ET":
        return ExtraTreesClassifier(random_state=SEED, **MODEL_PARAMS["ET"])

    if model_name == "KNN":
        return KNeighborsClassifier(**MODEL_PARAMS["KNN"])

    if model_name == "MLP":
        return MLPClassifier(random_state=SEED, **MODEL_PARAMS["MLP"])

    if model_name == "SVM":
        return SVC(**MODEL_PARAMS["SVM"])

    if model_name == "XGB":
        from xgboost import XGBClassifier
        params = MODEL_PARAMS["XGB"].copy()
        if task == "binary":
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=SEED, **params)
        return XGBClassifier(objective="multi:softmax", num_class=num_classes, eval_metric="mlogloss", random_state=SEED, **params)

    if model_name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, **MODEL_PARAMS["LGBM"])

    raise ValueError(f"Unsupported model: {model_name}")


def use_scaled_features(model_name):
    return model_name.upper() in {"KNN", "MLP", "SVM"}


def metric_dict(y_true, y_pred):
    return {
        "OA": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
    }


def fit_and_predict(model_name, X_train, y_train, X_val, task="multiclass", num_classes=None):
    scaler = None
    model = build_model(model_name, task=task, num_classes=num_classes)

    if use_scaled_features(model_name):
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_val_used = scaler.transform(X_val)
    else:
        X_train_used = X_train
        X_val_used = X_val

    model.fit(X_train_used, y_train)
    y_pred = model.predict(X_val_used)

    return model, scaler, y_pred


def predict_with_fitted_model(model, scaler, X):
    X_used = scaler.transform(X) if scaler is not None else X
    return model.predict(X_used)


def run_s1_hierarchical_classification(data, feature_cols):
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

    X_train_step1 = train_df[feature_cols].values
    X_val_step1 = val_df[feature_cols].values
    y_train_step1 = train_df["label_binary"].values
    y_val_step1 = val_df["label_binary"].values

    train_forest_df = train_df[train_df[LABEL_COLUMN].isin(FOREST_TYPES)].copy()
    val_forest_df = val_df[val_df[LABEL_COLUMN].isin(FOREST_TYPES)].copy()

    X_train_step2 = train_forest_df[feature_cols].values
    X_val_step2 = val_forest_df[feature_cols].values
    y_train_step2 = encoder_5.transform(train_forest_df[LABEL_COLUMN].astype(str).values)
    y_val_step2 = encoder_5.transform(val_forest_df[LABEL_COLUMN].astype(str).values)

    y_true_final = encoder_6.transform(val_df["label_6class"].astype(str).values)

    step1_cache = {}
    for model_name in MODEL_NAMES:
        model, scaler, y_pred = fit_and_predict(
            model_name,
            X_train_step1,
            y_train_step1,
            X_val_step1,
            task="binary",
            num_classes=2,
        )
        step1_metrics = {
            "Step1_OA": accuracy_score(y_val_step1, y_pred),
            "Step1_Kappa": cohen_kappa_score(y_val_step1, y_pred),
            "Step1_F1": f1_score(y_val_step1, y_pred, average="binary"),
        }
        step1_cache[model_name] = {
            "model": model,
            "scaler": scaler,
            "val_prediction": y_pred,
            "metrics": step1_metrics,
        }

    step2_cache = {}
    for model_name in MODEL_NAMES:
        model, scaler, y_pred = fit_and_predict(
            model_name,
            X_train_step2,
            y_train_step2,
            X_val_step2,
            task="multiclass",
            num_classes=5,
        )
        step2_metrics = {
            "Step2_OA": accuracy_score(y_val_step2, y_pred),
            "Step2_Kappa": cohen_kappa_score(y_val_step2, y_pred),
            "Step2_Macro_F1": f1_score(y_val_step2, y_pred, average="macro"),
            "Step2_Weighted_F1": f1_score(y_val_step2, y_pred, average="weighted"),
        }
        step2_cache[model_name] = {
            "model": model,
            "scaler": scaler,
            "metrics": step2_metrics,
        }

    results = []
    for step1_name in MODEL_NAMES:
        for step2_name in MODEL_NAMES:
            y_pred_step1 = step1_cache[step1_name]["val_prediction"]
            forest_mask = y_pred_step1 == 1

            y_pred_final_raw = np.array(["other"] * len(val_df), dtype=object)

            if forest_mask.sum() > 0:
                X_val_forest_pred = val_df.loc[forest_mask, feature_cols].values
                step2_info = step2_cache[step2_name]
                y_pred_step2 = predict_with_fitted_model(
                    step2_info["model"],
                    step2_info["scaler"],
                    X_val_forest_pred,
                )
                y_pred_final_raw[forest_mask] = encoder_5.inverse_transform(y_pred_step2)

            y_pred_final = encoder_6.transform(y_pred_final_raw)
            final_metrics = metric_dict(y_true_final, y_pred_final)

            row = {
                "Strategy": "S1",
                "Step1_Model": step1_name,
                "Step2_Model": step2_name,
                **step1_cache[step1_name]["metrics"],
                **step2_cache[step2_name]["metrics"],
                "Final_OA": final_metrics["OA"],
                "Final_Kappa": final_metrics["Kappa"],
                "Final_Macro_F1": final_metrics["Macro_F1"],
                "Final_Weighted_F1": final_metrics["Weighted_F1"],
            }
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["Final_OA", "Final_Macro_F1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results_df.insert(0, "id", range(1, len(results_df) + 1))

    return results_df


def run_s2_direct_6class(data, feature_cols):
    encoder_6 = LabelEncoder()
    encoder_6.fit(CLASS_ORDER_6)

    X = data[feature_cols].values
    y = encoder_6.transform(data["label_6class"].astype(str).values)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    results = []
    for model_name in MODEL_NAMES:
        _, _, y_pred = fit_and_predict(
            model_name,
            X_train,
            y_train,
            X_val,
            task="multiclass",
            num_classes=len(encoder_6.classes_),
        )
        metrics = metric_dict(y_val, y_pred)
        results.append({
            "Strategy": "S2",
            "Model": model_name,
            **metrics,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["OA", "Macro_F1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results_df.insert(0, "id", range(1, len(results_df) + 1))

    return results_df


def run_s3_fine_class_training_with_6class_evaluation(data, feature_cols):
    encoder_fine = LabelEncoder()
    y_fine = encoder_fine.fit_transform(data[LABEL_COLUMN].astype(str).values)

    encoder_6 = LabelEncoder()
    encoder_6.fit(CLASS_ORDER_6)
    y_eval_6 = encoder_6.transform(data["label_6class"].astype(str).values)

    X = data[feature_cols].values

    X_train, X_val, y_train_fine, y_val_fine, y_train_eval_6, y_val_eval_6 = train_test_split(
        X,
        y_fine,
        y_eval_6,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_fine,
    )

    fine_to_6 = {
        fine_code: encoder_6.transform([map_to_6class(fine_label)])[0]
        for fine_code, fine_label in enumerate(encoder_fine.classes_)
    }

    results = []
    for model_name in MODEL_NAMES:
        _, _, y_pred_fine = fit_and_predict(
            model_name,
            X_train,
            y_train_fine,
            X_val,
            task="multiclass",
            num_classes=len(encoder_fine.classes_),
        )
        y_pred_6 = np.array([fine_to_6[int(v)] for v in y_pred_fine])
        metrics = metric_dict(y_val_eval_6, y_pred_6)

        results.append({
            "Strategy": "S3",
            "Model": model_name,
            **metrics,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["OA", "Macro_F1"],
        ascending=[False, False],
    ).reset_index(drop=True)
    results_df.insert(0, "id", range(1, len(results_df) + 1))

    return results_df


def build_summary_table(s1_results, s2_results, s3_results, top_n_s1=10, top_n_s2=10, top_n_s3=10):
    s1_summary = s1_results.head(top_n_s1).copy()
    s1_summary["Model_or_Combination"] = (
        s1_summary["Step1_Model"].astype(str) + " + " + s1_summary["Step2_Model"].astype(str)
    )
    s1_summary = s1_summary.rename(columns={
        "Final_OA": "OA",
        "Final_Kappa": "Kappa",
        "Final_Macro_F1": "Macro_F1",
        "Final_Weighted_F1": "Weighted_F1",
    })
    s1_summary = s1_summary[["Strategy", "Model_or_Combination", "OA", "Kappa", "Macro_F1", "Weighted_F1"]]

    s2_summary = s2_results.head(top_n_s2).copy()
    s2_summary["Model_or_Combination"] = s2_summary["Model"]
    s2_summary = s2_summary[["Strategy", "Model_or_Combination", "OA", "Kappa", "Macro_F1", "Weighted_F1"]]

    s3_summary = s3_results.head(top_n_s3).copy()
    s3_summary["Model_or_Combination"] = s3_summary["Model"]
    s3_summary = s3_summary[["Strategy", "Model_or_Combination", "OA", "Kappa", "Macro_F1", "Weighted_F1"]]

    summary = pd.concat([s1_summary, s2_summary, s3_summary], ignore_index=True)
    summary.insert(0, "id", range(1, len(summary) + 1))

    return summary


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data, feature_cols = load_training_data(DATA_PATH)

    s1_results = run_s1_hierarchical_classification(data, feature_cols)
    s2_results = run_s2_direct_6class(data, feature_cols)
    s3_results = run_s3_fine_class_training_with_6class_evaluation(data, feature_cols)

    summary = build_summary_table(s1_results, s2_results, s3_results)

    s1_path = os.path.join(OUTPUT_DIR, "S1_hierarchical_model_combinations.csv")
    s2_path = os.path.join(OUTPUT_DIR, "S2_direct_6class_models.csv")
    s3_path = os.path.join(OUTPUT_DIR, "S3_fine_class_training_6class_evaluation.csv")
    summary_path = os.path.join(OUTPUT_DIR, "accuracy_summary_across_strategies.csv")

    s1_results.to_csv(s1_path, index=False, encoding="utf-8-sig")
    s2_results.to_csv(s2_path, index=False, encoding="utf-8-sig")
    s3_results.to_csv(s3_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Finished.")
    print(f"S1 results: {s1_path}")
    print(f"S2 results: {s2_path}")
    print(f"S3 results: {s3_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
