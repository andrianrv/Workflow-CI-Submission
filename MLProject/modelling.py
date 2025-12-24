import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- KONFIGURASI ---
DAGSHUB_REPO_OWNER = "andrianrv"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Andrian_Radita"

def train():
    print("--- MEMULAI MLPROJECT TRAINING ---")
    try:
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, surround=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")
    except Exception as e:
        print(f"[WARNING] Koneksi DagsHub: {e}")

    if not os.path.exists("vgsales_preprocessed.csv"):
        print("[FATAL] File csv tidak ditemukan!")
        return

    df = pd.read_csv("vgsales_preprocessed.csv")
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['Global_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Sedang melatih model...")
    with mlflow.start_run() as run:
        # SIMPAN RUN ID PENTING
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)

        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
        print(f"[SUKSES] Model dilatih. Run ID: {run_id}")

if __name__ == "__main__":
    train()