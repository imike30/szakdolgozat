import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("ml_ready_dataset_big.csv")

df = df.drop(columns=["TH", "M_case"], errors="ignore")

optical_cols = [c for c in df.columns if c.startswith("G")]
print(f"Optikai (target) oszlopok száma: {len(optical_cols)}")

feature_cols = [c for c in df.columns if c not in optical_cols]
print(f"Feature oszlopok száma: {len(feature_cols)}")

X = df[feature_cols]
y = df[optical_cols]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_true = y_test.values

def train_and_eval_rf_with_es(
    X_train_full,
    X_test,
    y_train_full,
    y_true,
    step=100,
    max_estimators=600,
    patience=2,
    random_state=42,
):

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=random_state
    )

    base_rf = RandomForestClassifier(
        n_estimators=step,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
        max_features="sqrt",
    )

    model = MultiOutputClassifier(base_rf)

    best_val_macro = -np.inf
    best_model = None
    no_improve = 0

    for n in range(step, max_estimators + step, step):
        # Fák számának növelése
        model.estimator.set_params(n_estimators=n)

        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        y_val_pred = np.asarray(y_val_pred)
        val_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        print(f"  {n} fa → val macro F1 = {val_macro:.4f}")

        if val_macro > best_val_macro + 1e-4:
            best_val_macro = val_macro
            best_model = deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("  Early stopping aktiválódott.\n")
            break

    if best_model is None:
        best_model = model

    y_pred = best_model.predict(X_test)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    subset_acc = float((y_true == y_pred).all(axis=1).mean())

    return macro_f1, micro_f1, subset_acc


print("\n=== Baseline modell tanítása (összes feature-rel) ===")
baseline_macro, baseline_micro, baseline_subset = train_and_eval_rf_with_es(
    X_train_full, X_test, y_train_full, y_true
)

print(f"Baseline macro F1:   {baseline_macro:.4f}")
print(f"Baseline micro F1:   {baseline_micro:.4f}")
print(f"Baseline subset acc: {baseline_subset:.4f}")

results = []

results.append({
    "dropped_feature": "NONE",
    "macro_f1": baseline_macro,
    "micro_f1": baseline_micro,
    "subset_accuracy": baseline_subset
})

for i, feat in enumerate(feature_cols, start=1):
    print(f"\n=== ({i}/{len(feature_cols)}) '{feat}' oszlop elhagyása ===")

    X_train_drop = X_train_full.drop(columns=[feat])
    X_test_drop = X_test.drop(columns=[feat])

    macro_f1, micro_f1, subset_acc = train_and_eval_rf_with_es(
        X_train_drop, X_test_drop, y_train_full, y_true
    )

    results.append({
        "dropped_feature": feat,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "subset_accuracy": subset_acc
    })

results_df = pd.DataFrame(results)

results_df["delta_macro_f1"] = results_df["macro_f1"] - baseline_macro
results_df["delta_micro_f1"] = results_df["micro_f1"] - baseline_micro
results_df["delta_subset_acc"] = results_df["subset_accuracy"] - baseline_subset

out_filename = "rf_feature_drop_sensitivity.csv"
results_df.to_csv(out_filename, index=False, encoding="utf-8")
print(f"\nEredmények elmentve ide: {out_filename}")

dropped_only = results_df[results_df["dropped_feature"] != "NONE"]

idx_min = dropped_only["macro_f1"].idxmin()
idx_max = dropped_only["macro_f1"].idxmax()

row_min = dropped_only.loc[idx_min]
row_max = dropped_only.loc[idx_max]

print("\n=== LEGFONTOSABB FEATURE (eltávolítása rontotta legjobban a macro F1-et) ===")
print(f"Feature: {row_min['dropped_feature']}")
print(f"macro_f1: {row_min['macro_f1']:.4f}  (baseline: {baseline_macro:.4f})")
print(f"delta_macro_f1: {row_min['delta_macro_f1']:.4f}")

print("\n=== LEGKEVÉSBÉ FONTOS FEATURE (eltávolítása rontotta legkevésbé / javított) ===")
print(f"Feature: {row_max['dropped_feature']}")
print(f"macro_f1: {row_max['macro_f1']:.4f}  (baseline: {baseline_macro:.4f})")
print(f"delta_macro_f1: {row_max['delta_macro_f1']:.4f}")
