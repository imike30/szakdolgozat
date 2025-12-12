import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb



df = pd.read_csv("ml_ready_dataset_big.csv")


df = df.drop(columns=["TH", "M_case"], errors="ignore")


optical_cols = [c for c in df.columns if c.startswith("G")]

print("Optikai oszlopok száma:", len(optical_cols))


X = df.drop(columns=optical_cols)
y = df[optical_cols]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_true = y_test.values


def evaluate_and_save(model, model_name, optical_cols, X_train, X_test, y_train, y_true):

    print(f"\n>>> Tanítás: {model_name}")
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred)

    per_link_rows = []
    f1_per_link = []
    prec_per_link = []
    rec_per_link = []


    for i, col in enumerate(optical_cols):

        report = classification_report(
            y_true[:, i],
            y_pred[:, i],
            output_dict=True,
            zero_division=0
        )

        
        if "1" in report:
            prec_1 = report["1"]["precision"]
            rec_1 = report["1"]["recall"]
            f1_1 = report["1"]["f1-score"]
            supp_1 = report["1"]["support"]
        else:
            
            prec_1 = 0.0
            rec_1 = 0.0
            f1_1 = 0.0
            supp_1 = 0

        per_link_rows.append({
            "link": col,
            "precision_1": prec_1,
            "recall_1": rec_1,
            "f1_1": f1_1,
            "support_1": supp_1,
        })

        f1_per_link.append(f1_1)
        prec_per_link.append(prec_1)
        rec_per_link.append(rec_1)

    
    per_link_df = pd.DataFrame(per_link_rows)
    per_link_filename = f"{model_name}_per_link_metrics.csv"
    per_link_df.to_csv(per_link_filename, index=False, encoding="utf-8")
    print(f"Per-link metrikák elmentve ide: {per_link_filename}")

    
    avg_prec_1 = float(np.mean(prec_per_link))
    avg_rec_1 = float(np.mean(rec_per_link))
    avg_f1_1 = float(np.mean(f1_per_link))

    
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    
    subset_acc = float((y_true == y_pred).all(axis=1).mean())

    summary = {
        "model_name": model_name,
        "avg_precision_1": avg_prec_1,
        "avg_recall_1": avg_rec_1,
        "avg_f1_1": avg_f1_1,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
    }

    return summary


models = []


rf_base = RandomForestClassifier(
    n_estimators=600,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
models.append((
    MultiOutputClassifier(rf_base),
    "RandomForest"
))


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svm_base = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)
models.append((
    MultiOutputClassifier(svm_base),
    "SVM"
))


mlp_base = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation="relu",
        max_iter=400,
        alpha=0.0005,
        random_state=42
)
)
models.append((
    MultiOutputClassifier(mlp_base),
    "MLP"
))


xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    tree_method="hist"
)

models.append((
    MultiOutputClassifier(xgb_base),
    "XGBoost"
))


summary_rows = []

for model, name in models:
    summary = evaluate_and_save(
        model=model,
        model_name=name,
        optical_cols=optical_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_true=y_true
    )
    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("models_summary_metrics.csv", index=False, encoding="utf-8")
print("\nÖsszefoglaló metrikák elmentve ide: models_summary_metrics.csv")
print(summary_df)
