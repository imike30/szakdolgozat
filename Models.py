import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import xgboost as xgb


df = pd.read_csv("ml_ready_dataset.csv")


df = df.drop(columns=["TH", "M_case"], errors="ignore")


optical_cols = [c for c in df.columns if c.startswith("G")]


X = df.drop(columns=optical_cols)
y = df[optical_cols]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_true = y_test.values 


def evaluate_multilabel_model(model, model_name):
    print(f"\n================= {model_name} =================")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred)

    for i, col in enumerate(optical_cols):
        print(f"\n--- {col} ---")
        print(classification_report(y_true[:, i], y_pred[:, i], zero_division=0))


rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf_model = MultiOutputClassifier(rf_base)

evaluate_multilabel_model(rf_model, "RandomForestClassifier")


svm_base = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
)
svm_model = MultiOutputClassifier(svm_base)

evaluate_multilabel_model(svm_model, "SVM (SVC, RBF kernel)")


mlp_base = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    max_iter=300,
    random_state=42
)
mlp_model = MultiOutputClassifier(mlp_base)

evaluate_multilabel_model(mlp_model, "MLPClassifier")


xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    tree_method="hist"
)
xgb_model = MultiOutputClassifier(xgb_base)

evaluate_multilabel_model(xgb_model, "XGBoostClassifier")
