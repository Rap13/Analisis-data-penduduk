import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def klaster_data(df):
    print("\n🔎 Mulai proses klastering...")

    # Pilih fitur numerik
    X = df[["Bekerja", "Tidak_Bekerja", "Bersekolah", "Mengurus_Rumah_Tangga"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Klastering dengan KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels

    print("✅ Klastering selesai!\n")
    print(df[["Kelompok_Umur", "Jenis_Kelamin", "Cluster"]])

    # Evaluasi
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, cluster_labels)
    db_index = davies_bouldin_score(X_scaled, cluster_labels)

    print("\n📈 Evaluasi Klastering:")
    print(f"- Inertia: {inertia:.2f}")
    print(f"- Silhouette Score: {silhouette:.4f} (semakin tinggi semakin baik)")
    print(f"- Davies-Bouldin Index: {db_index:.4f} (semakin rendah semakin baik)")

    # Visualisasi klaster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Bekerja", y="Bersekolah", hue="Cluster", palette="viridis", s=100)
    plt.title("Klastering Penduduk berdasarkan Bekerja & Bersekolah")
    plt.xlabel("Bekerja")
    plt.ylabel("Bersekolah")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
