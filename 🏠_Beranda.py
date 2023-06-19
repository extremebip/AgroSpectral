import streamlit as st
from PIL import Image

def main():
    st.set_page_config(
        page_title="Beranda",
        page_icon="🏠",
    )
    
    st.write(
        """
        # AgroSpectral

        **Selamat datang di aplikasi AgroSpectral!**

        AgroSpectral merupakan situs website yang memberikan prediksi **Total Produksi Padi** di provinsi Jawa Barat, Indonesia.
        Dengan menggunakan citra satelit MODIS, AgroSpectral menerapkan model *Convolutional Neural Network* dan model *Deep Gaussian Process* untuk dapat memprediksi total produksi padi.
        AgroSpectral menyediakan beberapa metode visualisasi, seperti histogram, barchart, dan peta interaktif untuk memudahkan Anda dalam menganalisa data.

        ## Manfaat
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        img1 = Image.open('assets/crop.jpg')
        st.image(img1, width=450)

    with col2:
        st.write(
            """
            ### Efisien

            AgroSpectral menggunakan data citra satelit yang bersifat terbuka, akurat dan terkini,
            sehingga AgroSpectral dapat memberikan total produksi padi dengan cepat dan efisien dibandingkan dengan
            pendekatan survei pada umumnya.
            """
        )

    col3, col4 = st.columns(2)

    with col3:
        st.write(
            """
            ### Informatif dan Intuitif

            AgroSpectral menyajikan data total produksi padi ke berbagai bentuk visualisasi yang interaktif,
            sehingga Anda dapat melakukan eksplorasi data, memvisualisasikan tren setiap tahunnya, dan membandingkan
            hasil prediksi terhadap data real dengan lebih mudah.
            """
        )

    with col4:
        img2 = Image.open('assets/analysis.png')
        st.image(img2, width=450)

if __name__ == "__main__":
    main()