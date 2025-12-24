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
    try:
        # PERBAIKAN DI SINI:
        # Hapus 'surround=True' (penyebab error)
        # Ganti dengan 'mlflow=True' (agar otomatis konfigurasi tracking URI & credentials)
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        
        # Baris ini sebenarnya redundan jika sudah pakai mlflow=True, 
        # tapi kita biarkan saja sebagai cadangan eksplisit.
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow")
        
    except Exception as e:
        print(f"[WARNING] Setup DagsHub: {e}")

    # 2. Load Data
    if not os.path.exists("vgsales_preprocessed.csv"):
        print("[FATAL] Dataset tidak ditemukan! Pastikan file CSV ada.")
        return

    df = pd.read_csv("vgsales_preprocessed.csv")
    
    # Pastikan nama kolom sesuai dengan hasil preprocessing Anda
    # Jika preprocessing menghapus kolom tertentu, sesuaikan di sini
    X = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    y = df['Global_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training & Logging
    print("[INFO] Melatih model...")
    
    # Biarkan MLflow mengatur Run ID (Penting untuk 'mlflow run')
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # A. Log ke Cloud (DagsHub)
        mlflow.sklearn.log_model(model, "model")
        print("[SUKSES] Model terkirim ke DagsHub via MLflow Run.")
        
        # B. Simpan LOKAL (Wajib untuk Docker Build)
        local_path = "model_output"
        mlflow.sklearn.save_model(model, local_path)
        print(f"[INFO] Model lokal tersimpan di: {local_path}")

if __name__ == "__main__":
    train()