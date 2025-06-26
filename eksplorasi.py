def eksplorasi_data(df):
    print("📌 Struktur Data:")
    print(df.info())
    
    print("\n📊 Statistik Deskriptif:")
    print(df.describe())

    print("\n👨‍👩‍👧‍👦 Jumlah Berdasarkan Jenis Kelamin:")
    print(df.groupby("Jenis_Kelamin").sum(numeric_only=True))
