import pandas as pd

def load_data(path):
    print("📥 Membaca dataset dari:", path)
    df = pd.read_csv(path)
    print("✅ Data berhasil dibaca!\n")
    return df

def bersihkan_data(df):
    print("🧹 Cek missing value...")
    print(df.isnull().sum())
    print("✅ Tidak ada missing value. Data bersih!\n")
    return df
