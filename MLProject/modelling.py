import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- KONFIGURASI KERAS (HARDCODED) ---
# Kita pasang alamat langsung biar tidak nyasar ke local
DAGSHUB_URI = "https://dagshub.com/andrianrv/Eksperimen_SML_Andrian_Radita.mlflow"

def train():
    print("--- MEMULAI MLPROJECT TRAINING ---")
    
    # 1. Setup Koneksi DagsHub (WAJIB ADA)
    mlflow.set_tracking_uri(DAGSHUB_URI)
    
    # Ambil token dari Environment Variable (diset oleh GitHub Actions)
    print(f"[INFO] Tracking URI: {mlflow.get_tracking_uri()}")

    # 2. Load Data
    data_path = "vgsales_preprocessed.csv"
    if not os.path.exists(data_path):
        print(f"[FATAL] File {data_path} tidak ditemukan!")
        return

    df = pd.read_csv(data_path)
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['Global_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training & Logging
    print("[INFO] Sedang melatih model...")
    
    # Pastikan run ini tercatat di server, bukan lokal
    with mlflow.start_run() as run:
        # SIMPAN RUN ID
        run_id = run.info.run_id
        print(f"[INFO] Run ID Baru: {run_id}")
        
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("r2_score", r2)
        
        # Log model ke DagsHub
        mlflow.sklearn.log_model(model, "model")
        
        print(f"[SUKSES] Model terkirim ke DagsHub! Run ID: {run_id}")

if __name__ == "__main__":
    train()