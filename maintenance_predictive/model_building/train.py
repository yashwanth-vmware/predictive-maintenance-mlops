import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import mlflow
import mlflow.sklearn

# XGBoost (install if missing)
try:
    from xgboost import XGBClassifier
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier


def _get_hf_token_from_colab_secrets():
    """
    Option A: If running in Colab, try to load HF_TOKEN from Colab Secrets.
    Falls back to environment variable if not in Colab.
    """
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN") or ""
    except Exception:
        pass
    return os.getenv("HF_TOKEN", "").strip()


def main():
    # ----------------------------
    # Hugging Face Space (Option 1)
    # ----------------------------
    HF_SPACE_REPO_ID = "Yashwanthsairam/predictive-maintenance-mlops"
    BASE_URL = "https://huggingface.co/spaces/{repo}/resolve/main".format(repo=HF_SPACE_REPO_ID)

    # Train/Test CSVs stored in the SAME Space repo under /splits
    train_url = "{base}/splits/train_engine_data.csv".format(base=BASE_URL)
    test_url  = "{base}/splits/test_engine_data.csv".format(base=BASE_URL)

    TARGET_COL = "Engine_Condition"

    # ----------------------------
    # Token (Option A)
    # ----------------------------
    HF_TOKEN = _get_hf_token_from_colab_secrets()
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. In Colab: add HF_TOKEN in Secrets and restart runtime."
        )

    api = HfApi(token=HF_TOKEN)

    # ----------------------------
    # MLflow setup
    # ----------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("engine_predictive_maintenance_xgb")

    # ----------------------------
    # Load train/test from HF Space URLs
    # ----------------------------
    train_df = pd.read_csv(train_url)
    test_df  = pd.read_csv(test_url)

    # Standardize column names (safety)
    train_df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in train_df.columns]
    test_df.columns  = [c.strip().replace(" ", "_").replace("-", "_") for c in test_df.columns]

    if TARGET_COL not in train_df.columns:
        raise KeyError(
            "Target column '{t}' not found. Train columns: {cols}".format(
                t=TARGET_COL, cols=list(train_df.columns)
            )
        )

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # ----------------------------
    # Model + Grid Search
    # ----------------------------
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBClassifier")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        mlflow.log_params(best_params)

        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("test_f1", float(f1))

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        mlflow.log_dict(report, "classification_report.json")

        print("✅ Best params:", best_params)
        print("✅ Test F1:", f1)

        # ----------------------------
        # Save artifacts locally
        # ----------------------------
        os.makedirs("artifacts", exist_ok=True)

        model_path = os.path.join("artifacts", "model.pkl")
        metrics_path = os.path.join("artifacts", "metrics.json")

        joblib.dump(best_model, model_path)

        with open(metrics_path, "w") as f:
            json.dump({"test_f1": float(f1), "best_params": best_params}, f, indent=2)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        # ----------------------------
        # Upload artifacts INTO THE SAME SPACE repo (Option 1)
        # ----------------------------
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="artifacts/model.pkl",
                repo_id=HF_SPACE_REPO_ID,
                repo_type="space",
            )
            api.upload_file(
                path_or_fileobj=metrics_path,
                path_in_repo="artifacts/metrics.json",
                repo_id=HF_SPACE_REPO_ID,
                repo_type="space",
            )

            # Optional marker for traceability
            best_model_txt = os.path.join("artifacts", "best_model.txt")
            with open(best_model_txt, "w") as f:
                f.write("XGBoost\n")
            api.upload_file(
                path_or_fileobj=best_model_txt,
                path_in_repo="artifacts/best_model.txt",
                repo_id=HF_SPACE_REPO_ID,
                repo_type="space",
            )

            print("✅ Uploaded model + metrics to Space repo:", HF_SPACE_REPO_ID)
            print("   - artifacts/model.pkl")
            print("   - artifacts/metrics.json")
            print("   - artifacts/best_model.txt")

        except HfHubHTTPError as e:
            raise RuntimeError("❌ Upload to Space failed: {e}".format(e=e))


if __name__ == "__main__":
    main()
