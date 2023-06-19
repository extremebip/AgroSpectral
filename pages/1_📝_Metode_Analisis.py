import streamlit as st
from PIL import Image

def main():
    st.set_page_config(
        page_title="Metode Analisis",
        page_icon="📝",
    )
        
    st.write(
        """
        # Metode Analisis

        AgroSpectral melakukan prediksi total produksi padi menggunakan citra satelit dengan langkah-langkah sebagai berikut:

        ## 1. Pengumpulan Data
        Terdapat dua data yang perlu dikumpulkan sebelum dapat dilakukan analisis, yaitu data produksi padi dan citra satelit.

        - Data produksi padi diperoleh dari [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/produksi-padi-berdasarkan-kabupatenkota-di-jawa-barat)
        - Citra satelit didapatkan dari *Google Earth Engine*. Citra Satelit yang digunakan adalah:
        1. Refleksi Permukaan Bumi ([MOD09A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1)). Pada citra satelit ini, terdapat 7 band yang diambil, yaitu:
        
            | Band | Range Panjang Gelombang Cahaya yang Ditangkap |
            | ----------- | ----------- |
            | Band 1 | 620-670nm |
            | Band 2 | 841-876nm |
            | Band 3 | 459-479nm |
            | Band 4 | 545-565nm |
            | Band 5 | 1230-1250nm |
            | Band 6 | 1628-1652nm |
            | Band 7 | 2105-2155nm |
            
            
            Citra satelit ini melakukan pengambilan foto setiap hari dan dilakukan komposit setiap 8 hari. Sehingga, di dalam 1 tahun, terdapat 46 citra satelit.
        
        2. Suhu Permukaan Bumi ([MYD11A2](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A2)). Pada citra satelit ini, terdapat 2 band yang diambil, yaitu:
        
            | Band | Keterangan |
            | ----------- | ----------- |
            | LST_Day_1km | Suhu permukaan bumi pada siang hari dalam Kelvin |
            | LST_Night_1km | Suhu permukaan bumi pada malam hari dalam Kelvin |
            
            Citra satelit ini melakukan pengambilan foto setiap hari dan dilakukan komposit setiap 8 hari. Sehingga, di dalam 1 tahun, terdapat 46 citra satelit.
        
        3. Tutupan Lahan Bumi ([MCD12Q1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1)). Pada citra satelit ini, hanya ada 1 band yang diambil, yaitu `LC_Type1` yang menandakan klasifikasi tanah berdasarkan *International Geosphere-Biosphere Programme* (IGBP). Citra satelit ini hanya melakukan pengambilan foto 1 kali dalam setahun.

        ## 2. Pengolahan Data
        Untuk data produksi padi dilakukan pembersihan *missing value* dan pengambilan sampel 18 kabupaten dari 27 kabupaten/kota di provinsi Jawa Barat.

        Untuk data citra satelit, dilakukan beberapa langkah-langkah pengolahan data:
        1. Mengelompokkan citra satelit MOD09A1, MYD11A2, dan MCD12Q1 ke masing-masing kabupaten dan tahun pengambilannya.
        2. Melakukan teknik *masking*, yaitu mengubah nilai pixel pada citra satelit MOD09A1 dan MYD11A2 menjadi 0 pada pixel-pixel yang bukan merupakan tanah sawah. Hal ini diketahui dari citra satelit MCD12Q1.
        3. Membuat histogram pada setiap yang memiliki range nilai 0-4999 dan dibagi menjadi 32 *bins*. Citra satelit MOD09A1 dan MYD11A2 memiliki 46 foto dengan panjang dan lebar sesuai dengan area kabupaten. Setiap nilai pixel di dalam setiap foto dibentuk menjadi histogram dan menghasilkan sebanyak 46 histogram. Proses ini diulangi ke setiap band di dalam citra satelit MOD09A1 dan MYD11A2 untuk setiap kabupaten dan setiap tahun.
        """
    )

    with st.expander("Contoh histogram"):
        st.write(
            """
            Berikut adalah contoh histogram untuk setiap band MOD09A1 (Gambar 1 - 7) dan setiap band MYD11A2 (Gambar 8 dan 9).

            Setiap histogram memiliki panjang 46 pixel (jumlah pengambilan citra dalam satu tahun) dan lebar 32 (jumlah bins dalam histogram).
            """
        )

        col1, col2, col3 = st.columns(3)

        with col2:
            img = Image.open('assets/example_hist.png')
            st.image(img)
    
    st.write(
        """
        Setelah pengolahan data selesai dilakukan, data tahun 2013-2019 digunakan untuk training dan data tahun 2020 digunakan untuk testing.

        ## 3. Pelatihan Model
        Dari pengolahan data citra satelit, histogram citra satelit digunakan untuk melatih model *Convolutional Neural Network*. Output yang didapatkan dari model CNN digunakan sebagai input ke dalam model *Gaussian Process* dan didapatkan hasil prediksi.

        Referensi Utama:

        > You, J., Li, X., Low, M., Lobell, D., & Ermon, S. (2017). Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data. AAAI.
        """
    )

if __name__ == "__main__":
    main()