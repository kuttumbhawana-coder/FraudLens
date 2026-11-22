from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "transactions.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
target = "is_fraud"
drop_cols = ["transaction_id", target]
X = df.drop(columns=drop_cols)
y = df[target]

numeric = ["amount", "hour", "distance_from_home_km", "device_change",
           "international", "transactions_last_24h", "account_age_days", "is_weekend"]
categorical = ["merchant_risk"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=20, random_state=42
    )
}

results = []
best_name, best_pipe, best_ap = None, None, -1
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    scores = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pred
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "average_precision": average_precision_score(y_test, scores)
    }
    results.append(row)
    if row["average_precision"] > best_ap:
        best_ap, best_name, best_pipe = row["average_precision"], name, pipe

metrics = pd.DataFrame(results).sort_values("average_precision", ascending=False)
metrics.to_csv(OUT_DIR / "model_comparison.csv", index=False)
best_pipe.fit(X_train, y_train)
joblib.dump(best_pipe, MODEL_DIR / "fraud_detection_pipeline.joblib")

best_scores = best_pipe.predict_proba(X_test)[:, 1]
best_pred = (best_scores >= 0.5).astype(int)
report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
with open(OUT_DIR / "evaluation.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "threshold": 0.5,
        "roc_auc": roc_auc_score(y_test, best_scores),
        "average_precision": average_precision_score(y_test, best_scores),
        "confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "classification_report": report,
        "test_rows": int(len(y_test)),
        "fraud_rate": float(y.mean())
    }, f, indent=2)

print("Training complete.")
print(metrics.to_string(index=False))
print(f"Best model: {best_name}")
print("Saved models/fraud_detection_pipeline.joblib")
