import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_models(csv_path):
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"❌ Error: File {csv_path} tidak ditemukan!")
        return

    print(f"📦 Membaca dataset dari: {csv_path}")
    data = pd.read_csv(csv_path)

    # Bersihkan kolom jika ada
    if 'filename' in data.columns:
        data.drop(columns=['filename'], inplace=True)

    # 2. Training Model Panjang (Panjang_px -> Panjang_cm)
    print("🚀 Training model prediksi panjang...")
    X_len = data[["panjang_px"]]
    y_len = data["panjang"]
    model_len = LinearRegression().fit(X_len, y_len)

    # 3. Training Model Berat (Area_px -> Berat_gram)
    print("🚀 Training model prediksi berat...")
    X_weight = data[["area_px"]]
    y_weight = data["berat"]
    model_weight = LinearRegression().fit(X_weight, y_weight)

    # 4. Evaluasi Sederhana
    mae_len = mean_absolute_error(y_len, model_len.predict(X_len))
    mae_weight = mean_absolute_error(y_weight, model_weight.predict(X_weight))
    
    print("-" * 30)
    print(f"📊 Hasil Evaluasi (MAE):")
    print(f"   - Panjang (cm) MAE: {mae_len:.4f}")
    print(f"   - Berat (gram) MAE: {mae_weight:.4f}")
    print("-" * 30)

    # 5. Simpan Model
    # Pastikan folder 'model' ada agar tidak error saat simpan
    os.makedirs("model", exist_ok=True)

    path_panjang = "model/Newmodel_panjang.pkl"
    path_berat = "model/Newmodel_berat.pkl"

    joblib.dump(model_len, path_panjang)
    joblib.dump(model_weight, path_berat)

    print(f"✅ Model berhasil disimpan di:")
    print(f"   - {path_panjang}")
    print(f"   - {path_berat}")

if __name__ == "__main__":
    # Sesuaikan path dataset kamu di sini
    DATASET_PATH = "dataset_supervised.csv" 
    train_models(DATASET_PATH)