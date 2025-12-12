import pandas as pd
import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


df = pd.read_csv("ml_ready_dataset.csv")


df = df.drop(columns=["TH", "M_case", "failed_optical"], errors="ignore")

optical_cols = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


X = df.drop(columns=optical_cols)
y = df[optical_cols]

feature_cols = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_true = y_test.values


models = {}


rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
models["RandomForest"] = MultiOutputClassifier(rf_base)


svm_base = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
)
models["SVM"] = MultiOutputClassifier(svm_base)


mlp_base = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    max_iter=300,
    random_state=42
)
models["MLP"] = MultiOutputClassifier(mlp_base)


xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)
models["XGBoost"] = MultiOutputClassifier(xgb_base)

# Modellek betanítása teljes feature-készlettel
for name, model in models.items():
    print(f"Modell tanítása: {name} ...")
    model.fit(X_train, y_train)

print("Minden modell betanítva.")


# ===== 3. Segédfüggvény metrikák számításához =====

def compute_metrics(y_true, y_pred):

    subset_acc = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return subset_acc, micro_f1, macro_f1


combinations_to_test = []


combinations_to_test.append(())  # nincs kinullázott feature

# egyesével
for col in feature_cols:
    combinations_to_test.append((col,))

# párosával
for col1, col2 in itertools.combinations(feature_cols, 2):
    combinations_to_test.append((col1, col2))

print(f"🔢 Összesen {len(combinations_to_test)} feature kombináció kerül tesztelésre.")


results = []

for missing_tuple in combinations_to_test:

    X_test_mod = X_test.copy()
    for col in missing_tuple:
        if col in X_test_mod.columns:
            X_test_mod[col] = 0.0

    missing_str = "none" if len(missing_tuple) == 0 else "|".join(missing_tuple)

    for model_name, model in models.items():

        y_pred = model.predict(X_test_mod)
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
results_df.to_csv("missing_feature_experiments.csv", index=False)

print("✅ Kész: missing_feature_experiments.csv létrehozva.")
print("Példa sorok:")
print(results_df.head())
