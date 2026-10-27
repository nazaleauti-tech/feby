import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩺", layout="wide")

# Dataset 10 pasien sesuai soal: 0 = Risiko Rendah, 1 = Risiko Tinggi
data = [
    {"Pasien":"P01","Usia":25,"BMI":21.5,"Glukosa":90,"Tekanan Darah":110,"Riwayat Keluarga":0,"Kelas":0},
    {"Pasien":"P02","Usia":30,"BMI":23.1,"Glukosa":95,"Tekanan Darah":115,"Riwayat Keluarga":0,"Kelas":0},
    {"Pasien":"P03","Usia":35,"BMI":25.0,"Glukosa":105,"Tekanan Darah":120,"Riwayat Keluarga":0,"Kelas":0},
    {"Pasien":"P04","Usia":40,"BMI":27.5,"Glukosa":120,"Tekanan Darah":125,"Riwayat Keluarga":1,"Kelas":1},
    {"Pasien":"P05","Usia":45,"BMI":30.2,"Glukosa":140,"Tekanan Darah":135,"Riwayat Keluarga":1,"Kelas":1},
    {"Pasien":"P06","Usia":50,"BMI":31.5,"Glukosa":155,"Tekanan Darah":140,"Riwayat Keluarga":1,"Kelas":1},
    {"Pasien":"P07","Usia":55,"BMI":33.0,"Glukosa":165,"Tekanan Darah":145,"Riwayat Keluarga":1,"Kelas":1},
    {"Pasien":"P08","Usia":28,"BMI":22.5,"Glukosa":92,"Tekanan Darah":112,"Riwayat Keluarga":0,"Kelas":0},
    {"Pasien":"P09","Usia":38,"BMI":26.8,"Glukosa":115,"Tekanan Darah":122,"Riwayat Keluarga":0,"Kelas":0},
    {"Pasien":"P10","Usia":48,"BMI":29.5,"Glukosa":145,"Tekanan Darah":130,"Riwayat Keluarga":1,"Kelas":1},
]
ATRIBUT = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]

def normalisasi(nilai, atribut):
    nilai_kolom = [p[atribut] for p in data]
    minimum, maksimum = min(nilai_kolom), max(nilai_kolom)
    return 0 if maksimum == minimum else (nilai - minimum) / (maksimum - minimum)

def jarak_euclidean(pasien_baru, pasien_lama):
    jumlah = 0
    for atribut in ATRIBUT:
        a = normalisasi(pasien_baru[atribut], atribut)
        b = normalisasi(pasien_lama[atribut], atribut)
        jumlah += (a - b) ** 2
    return math.sqrt(jumlah)

def knn(pasien_baru, k):
    urut = []
    for pasien in data:
        salinan = pasien.copy()
        salinan["Jarak"] = jarak_euclidean(pasien_baru, pasien)
        urut.append(salinan)
    urut.sort(key=lambda x: x["Jarak"])
    tetangga = urut[:k]
    suara = {0: 0, 1: 0}
    total_jarak = {0: 0.0, 1: 0.0}
    for p in tetangga:
        suara[p["Kelas"]] += 1
        total_jarak[p["Kelas"]] += p["Jarak"]
    if suara[0] == suara[1]:
        hasil = 0 if total_jarak[0] <= total_jarak[1] else 1
    else:
        hasil = 0 if suara[0] > suara[1] else 1
    return hasil, tetangga, suara

st.markdown("""
<style>
.block-container {max-width:1150px;padding-top:2rem}
.hero {padding:25px;border-radius:16px;color:white;background:linear-gradient(120deg,#173b72,#3478e5)}
div[data-testid="stMetric"] {background:#f3f6fc;padding:12px;border-radius:12px}
</style>
<div class="hero"><h1>Prediksi Risiko Diabetes</h1>
<p>Aplikasi Data Mining menggunakan algoritma K-Nearest Neighbors (KNN)</p></div>
""", unsafe_allow_html=True)
st.caption("Dataset contoh dari soal • Euclidean Distance • Normalisasi Min-Max")

kiri, kanan = st.columns(2, gap="large")
with kiri:
    st.subheader("Masukkan data pasien baru")
    with st.form("form_pasien"):
        c1, c2 = st.columns(2)
        with c1:
            usia = st.number_input("Usia (tahun)", 1, 120, 42)
            bmi = st.number_input("BMI", 10.0, 80.0, 28.5, step=0.1)
            glukosa = st.number_input("Glukosa", 20, 600, 130)
        with c2:
            tekanan = st.number_input("Tekanan darah", 30, 300, 128)
            keluarga = st.selectbox("Riwayat diabetes keluarga", ["1 - Ada", "0 - Tidak ada"])
            k = st.selectbox("Jumlah tetangga (k)", [1, 3, 5, 7], index=1)
        hitung = st.form_submit_button("Hitung Prediksi", use_container_width=True)

    pasien_baru = {"Usia":usia, "BMI":bmi, "Glukosa":glukosa,
                   "Tekanan Darah":tekanan,
                   "Riwayat Keluarga":1 if keluarga.startswith("1") else 0}

    if hitung or "hasil" not in st.session_state:
        st.session_state["hasil"] = knn(pasien_baru, k)
        st.session_state["nilai_k"] = k

with kanan:
    st.subheader("Hasil klasifikasi")
    hasil, tetangga, suara = st.session_state["hasil"]
    if hasil == 0:
        st.success("Prediksi: RISIKO RENDAH")
    else:
        st.error("Prediksi: RISIKO TINGGI")
    st.write(f"Kelas dipilih dari voting {st.session_state['nilai_k']} tetangga terdekat.")
    a, b, c = st.columns(3)
    a.metric("Suara rendah", suara[0])
    b.metric("Suara tinggi", suara[1])
    c.metric("Nilai k", st.session_state["nilai_k"])
    st.info("Catatan: ini hanya demonstrasi tugas Data Mining, bukan alat diagnosis medis.")

st.divider()
st.subheader("Tabel tetangga terdekat")
df_tetangga = pd.DataFrame(tetangga)
df_tetangga["Jarak"] = df_tetangga["Jarak"].round(4)
df_tetangga["Kelas"] = df_tetangga["Kelas"].map({0:"Risiko Rendah", 1:"Risiko Tinggi"})
df_tetangga.insert(0, "Urutan", range(1, len(df_tetangga)+1))
st.dataframe(df_tetangga, use_container_width=True, hide_index=True)

st.subheader("Dataset pelatihan")
df_data = pd.DataFrame(data)
df_data["Kelas"] = df_data["Kelas"].map({0:"Risiko Rendah", 1:"Risiko Tinggi"})
st.dataframe(df_data, use_container_width=True, hide_index=True)
