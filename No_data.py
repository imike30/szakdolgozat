import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

import xgboost as xgb


df = pd.read_csv("ml_ready_dataset.csv")


df = df.drop(columns=["TH", "M_case", "failed_optical"], errors="ignore")

optical_cols = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


X_full = df.drop(columns=optical_cols)
y = df[optical_cols]

feature_cols = list(X_full.columns)

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

y_true = y_test.values 


base_models = {}


rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
base_models["RandomForest"] = MultiOutputClassifier(rf_base)


svm_base = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
)
base_models["SVM"] = MultiOutputClassifier(svm_base)


mlp_base = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    max_iter=300,
    random_state=42
)
base_models["MLP"] = MultiOutputClassifier(mlp_base)



xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)
base_models["XGBoost"] = MultiOutputClassifier(xgb_base)



def compute_metrics(y_true, y_pred):

    subset_acc = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return subset_acc, micro_f1, macro_f1



combinations_to_test = []

# semmit nem veszünk ki
combinations_to_test.append(())

# egyesével
for col in feature_cols:
    combinations_to_test.append((col,))

# párosával
for col1, col2 in itertools.combinations(feature_cols, 2):
    combinations_to_test.append((col1, col2))

print(f"Összesen {len(combinations_to_test)} feature-kombináció kerül tesztelésre.")



results = []

for missing_tuple in combinations_to_test:
    
    missing_set = set(missing_tuple)
    remaining_features = [c for c in feature_cols if c not in missing_set]

    
    if len(remaining_features) == 0:
        continue

    
    X_train = X_train_full[remaining_features]
    X_test = X_test_full[remaining_features]

    missing_str = "none" if len(missing_tuple) == 0 else "|".join(missing_tuple)

    print(f"\n=== Kombináció: missing = {missing_str} (marad: {len(remaining_features)} feature) ===")

    for model_name, base_model in base_models.items():
        print(f"  -> Modell tanítása: {model_name} ...")
        model = clone(base_model)

       
        model.fit(X_train, y_train)

        
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred)

        subset_acc, micro_f1, macro_f1 = compute_metrics(y_true, y_pred)

        results.append({
            "missing_features": missing_str,
            "num_missing": len(missing_tuple),
            "model": model_name,
            "subset_accuracy": subset_acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        })



results_df = pd.DataFrame(results)
results_df.to_csv("feature_removal_experiments.csv", index=False)

print("\nfeature_removal_experiments.csv létrehozva.")

