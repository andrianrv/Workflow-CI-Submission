import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- KONFIGURASI ---
DAGSHUB_REPO_OWNER = "andrianrv"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Andrian_Radita"

def train():
    print("--- MEMULAI TRAINING VIA MLFLOW PROJECT ---")
    
    # 1. Setup DagsHub
    # Saat pakai 'mlflow run', env var biasanya sudah diset, tapi init ulang aman
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, surround=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")

    # 2. Load Data
    if not os.path.exists("vgsales_preprocessed.csv"):
        print("[FATAL] Dataset tidak ditemukan!")
        return

    df = pd.read_csv("vgsales_preprocessed.csv")
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['Global_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training & Logging
    # PENTING: Jangan buat run ID manual, biarkan MLflow menangkap active run
    print("[INFO] Melatih model...")
    
    with mlflow.start_run(): 
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Log Model ke Cloud
        mlflow.sklearn.log_model(model, "model")
        print("[SUKSES] Model terkirim ke DagsHub via MLflow Run.")
        
        # Simpan LOKAL (Wajib untuk Docker Build nanti)
        local_path = "model_output"
        mlflow.sklearn.save_model(model, local_path)
        print(f"[INFO] Model lokal tersimpan di: {local_path}")

if __name__ == "__main__":
    train()