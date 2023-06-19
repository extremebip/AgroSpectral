import sys
import os
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

from utilities import *

def regency_multiselect(regencies):
    return st.multiselect(
        "Pilih Kabupaten",
        regencies,
        regencies
    )

def year_slider(years):
    return st.select_slider(
        "Pilih Tahun",
        years,
        (years[0], years[-1])
    )

def dataframe_tab(yield_df):
    st.header("Data Produksi Padi")
    st.dataframe(
        yield_df[["Regency_Code", "Regency", "Padi", "Year"]],
        column_config={
            "Regency_Code": st.column_config.NumberColumn(
                "Kode Kabupaten",
                format="%d"
            ),
            "Regency": st.column_config.TextColumn("Kabupaten"),
            "Year": st.column_config.NumberColumn(
                "Tahun",
                format="%d"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

def histogram_tab(yield_df):
    st.header("Distribusi Produksi Padi")
    fig = px.histogram(yield_df, x="Padi", title="", labels={"count": "Frekuensi"})
    st.plotly_chart(fig)

def barchart_tab(yield_df):
    st.header("Produksi Padi")
    aggregate_by = st.radio(
        "By",
        ("Per kabupaten", "Per tahun", "Per kabupaten per tahun"),
        label_visibility='hidden',
        horizontal=True
    )

    if aggregate_by == "Per kabupaten":
        aggregate_df = yield_df.groupby('Regency')['Padi'].sum().reset_index()
        aggregate_df = aggregate_df.sort_values(by='Padi', ascending=False)
        fig = px.bar(aggregate_df, x='Padi', y='Regency', orientation='h', labels={"Regency": "Kabupaten"})
    elif aggregate_by == "Per tahun":
        aggregate_df = yield_df.groupby('Year')['Padi'].sum().reset_index()
        fig = px.bar(aggregate_df, x='Year', y='Padi', labels={"Year": "Tahun"})
        fig.update_layout(xaxis_type='category')
    else:
        fig = px.bar(yield_df,
                     x='Padi',
                     y='Regency',
                     color='Year',
                     orientation='h',
                     labels={"Regency": "Kabupaten", "Year": "Tahun"}
                     )
        fig.update_layout(yaxis = {"categoryorder": "sum descending"})

    st.plotly_chart(fig)

def linechart_tab(yield_df):
    st.header("Produksi Padi per Tahun")

    aggregate_df = yield_df.groupby('Year')['Padi'].sum().reset_index()
    fig = px.line(aggregate_df, x='Year', y='Padi', labels={"Year": "Tahun"})
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig)

def map_tab(yield_df):
    st.header("Produksi Padi per Kabupaten")

    aggregate_df = yield_df.groupby('Regency')['Padi'].sum().reset_index()
    aggregate_df.columns = ["Kabupaten", "Padi"]

    map = folium.Map(location=(-6.9212, 107.6157), zoom_start=8, tiles="CartoDB positron")

    choropleth = folium.Choropleth(
        geo_data='data/Jabar_SHP.json',
        data=aggregate_df,
        columns=("Kabupaten", "Padi"),
        key_on="feature.properties.KAB_KOTA",
        line_opacity=0.8,
        fill_opacity=0.8,
        highlight=True,
        fill_color="Greens",
    )
    choropleth.geojson.add_to(map)

    aggregate_df = aggregate_df.set_index('Kabupaten')
    for feature in choropleth.geojson.data['features']:
        nama_kabupaten = feature['properties']['KAB_KOTA']
        feature['properties']['Kabupaten'] = nama_kabupaten
        feature['properties']['Padi'] = 'Padi: ' + str(aggregate_df.loc[nama_kabupaten, 'Padi'] if nama_kabupaten in list(aggregate_df.index) else 'N/A')

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['Kabupaten', 'Padi'], labels=False)
    )

    st_folium(map, width=750, height=450, returned_objects=[])

def main():
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

    st.set_page_config(
        page_title="Eksplorasi Data",
        page_icon="📊",
    )

    st.markdown(
        """
        # Eksplorasi Data

        ## Data Padi
        Data yang digunakan adalah data produksi padi untuk setiap kabupaten di provinsi Jawa Barat
        dari tahun 2013 - 2020. Data dapat diunduh pada: [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/produksi-padi-berdasarkan-kabupatenkota-di-jawa-barat)
        """
    )

    yield_df = get_yield_df()

    regencies = yield_df['Regency'].unique().tolist()
    years = yield_df['Year'].unique().tolist()
    selected_regencies = regency_multiselect(regencies)
    start_year, end_year = year_slider(years)

    selected_years = [*range(start_year, end_year + 1)]

    filtered_yields = filter_yield_df(yield_df, selected_regencies, selected_years)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tabel", "Histogram", "Bar Chart", "Line Chart", "Peta"])
    with tab1:
        dataframe_tab(filtered_yields)

    with tab2:
        histogram_tab(filtered_yields)

    with tab3:
        barchart_tab(filtered_yields)

    with tab4:
        linechart_tab(filtered_yields)

    with tab5:
        map_tab(filtered_yields)

if __name__ == "__main__":
    main()