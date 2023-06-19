import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static

from utilities import *
from gp import GaussianProcess

@st.cache_data
def get_real_pred_df(code_regency_map, real_dict, pred_dict, pred_err_dict):
    combined_dict = {k: [
     code_regency_map[k], real_dict[k], pred_dict[k], pred_err_dict[k]
    ] for k in real_dict }
    result_df =  pd.DataFrame.from_dict(combined_dict).transpose()
    result_df.columns = ['Kabupaten', 'Produksi', 'Prediksi', 'Error Prediksi']
    result_df = result_df.sort_values(by='Prediksi', ascending=False)
    return result_df

def calc_metrics(true, pred):
  rmse = np.sqrt(np.mean((true - pred) ** 2))
  mae = np.mean(np.absolute(true - pred))
  mape = np.mean(np.absolute((true - pred) / true))
  return rmse, mae, mape

def run_gp_model(model, sigma=1, r_loc=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
  gp_model = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)
  gp_pred = gp_model.run(
    model["train_feat"],
    model["test_feat"],
    model["train_loc"],
    model["test_loc"],
    model["train_years"],
    model["test_years"],
    model["train_real"],
    model["model_weight"],
    model["model_bias"],
  )
  return gp_pred.squeeze(1)

def get_gp_parameters():
    sigma = 4.98057109586427
    r_loc = 0.0149434503570452
    r_year = 2.98888665618427
    sigma_e = 0.0195617243986127
    sigma_b = 0.533659083232355
    return sigma, r_loc, r_year, sigma_e, sigma_b

def show_model_evaluation(real_values, pred_values, gp_values):
    st.markdown("## Evaluasi Model")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    col6, col7 = st.columns(2)

    total_real = np.sum(real_values)
    total_pred = np.sum(pred_values)
    pred_err = abs(total_real - total_pred)
    pred_err_percent = (pred_err / total_real) * 100

    total_gp = np.sum(gp_values)
    gp_err = abs(total_real - total_gp)
    gp_err_percent = (gp_err / total_real) * 100

    with col2:
        st.metric("Total Produksi Padi (2020)", '{:.2f}'.format(total_real))
    with col4:
        st.metric("Prediksi Total Produksi Padi (Tanpa GP)", '{:.2f}'.format(total_pred))
    with col5:
        st.metric("Prediksi Total Produksi Padi (Dengan GP)", '{:.2f}'.format(total_gp))
    with col6:
        st.metric("Persentase Error Prediksi (Tanpa GP)", '{:.2f}%'.format(pred_err_percent))
    with col7:
        st.metric(
            "Persentase Error Prediksi (Dengan GP)",
            '{:.2f}%'.format(gp_err_percent),
            '{:.2f}%'.format(gp_err_percent - pred_err_percent),
            "inverse"
        )

    rmse, mae, mape = calc_metrics(real_values, pred_values)
    rmse_gp, mae_gp, mape_gp = calc_metrics(real_values, gp_values)

    metric_df = pd.DataFrame([
        ['CNN', rmse, mae, mape * 100],
        ['CNN + GP', rmse_gp, mae_gp, mape_gp * 100],
    ], columns=['Model', 'RMSE', 'MAE', 'MAPE'])
    metric_df = metric_df.transform(lambda x: round(x, 2) if type(x) != 'str' else x)
    metric_df['MAPE'] = metric_df['MAPE'].transform(lambda x: "{:.2f}%".format(x))

    st.dataframe(metric_df, use_container_width=True, hide_index=True)

def values_to_dict(indices, values):
  dict_result = {}
  for idx, val in zip(indices, values):
    regency_code = idx
    dict_result[regency_code] = val
  return dict_result

def show_predict_map(pred_result_df, gp_result_df):
    st.markdown("## Prediksi dan Error Berdasarkan Kabupaten")

    model_to_use = st.radio(
        "Pilih Model",
        ('CNN', 'CNN + GP'),
        horizontal=True,
    )

    tab1, tab2 = st.tabs(["Tabel", "Peta"])

    yield_pred = np.concatenate((pred_result_df['Prediksi'], gp_result_df['Prediksi']))
    pred_hist, pred_bin_edges = np.histogram(yield_pred, bins=6)

    pred_err = np.concatenate((pred_result_df['Error Prediksi'], gp_result_df['Error Prediksi']))
    err_hist, err_bin_edges = np.histogram(pred_err, bins=6)

    if model_to_use == 'CNN':
        yield_df = pred_result_df
    else:
        yield_df = gp_result_df

    with tab1:
        st.dataframe(
            yield_df,
            column_config={
                "Produksi": st.column_config.NumberColumn("Produksi Padi (Actual)"),
                "Prediksi": st.column_config.NumberColumn(
                    "Produksi Padi (Prediksi)",
                ),
                "Error Prediksi": st.column_config.NumberColumn(
                    "Error Prediksi",
                ),
            },
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        data_to_show = st.radio(
            "Data yang ditampilkan",
            ('Prediksi Padi', 'Error Prediksi'),
            horizontal=True,
        )

        # Default from folium
        bins = 6
        if data_to_show == 'Prediksi Padi':
            fill_color = "Greens"
            col_name = "Prediksi"
            tooltip_label = "Prediksi Padi: "
            bins = pred_bin_edges.tolist()
        else:
            fill_color = "Reds"
            col_name = "Error Prediksi"
            tooltip_label = "Error Prediksi: "
            bins = err_bin_edges.tolist()

        map = folium.Map(location=(-6.9212, 107.6157), zoom_start=8, tiles="CartoDB positron")

        choropleth = folium.Choropleth(
            geo_data='data/Jabar_SHP.json',
            data=yield_df,
            columns=("Kabupaten", col_name),
            key_on="feature.properties.KAB_KOTA",
            bins=bins,
            line_opacity=0.8,
            fill_opacity=0.8,
            highlight=True,
            fill_color=fill_color,
            legend_name=str(data_to_show + " berdasarkan model " + model_to_use),
        )
        choropleth.add_to(map)

        yield_df = yield_df.set_index('Kabupaten')
        for feature in choropleth.geojson.data['features']:
            nama_kabupaten = feature['properties']['KAB_KOTA']
            feature['properties']['Kabupaten'] = nama_kabupaten
            feature['properties'][col_name] = tooltip_label + \
                str(yield_df.loc[nama_kabupaten, col_name] if nama_kabupaten in list(yield_df.index) else 'N/A')

        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['Kabupaten', col_name], labels=False)
        )

        folium_static(map, width=750, height=450)

def main():
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)

    st.set_page_config(
        page_title="Hasil Analisis",
        page_icon="📈",
    )

    st.markdown(
       """
       # Hasil Analisis

       Evaluasi model dilakukan terhadap prediksi produksi padi pada tahun 2020.

       Model yang digunakan adalah Convolutional Neural Network (CNN) dan Deep Gaussian Process (GP).

       Analisis ini bertujuan untuk membandingkan hasil prediksi produksi padi tahun 2020 antara model CNN dengan model CNN + GP.
       """
    )

    yield_df = get_yield_df()
    code_regency_map = dict(zip(yield_df.Regency_Code, yield_df.Regency))
    model = get_model()

    real_values = model["test_real"]
    pred_values = model["test_pred"]
    sigma, r_loc, r_year, sigma_e, sigma_b = get_gp_parameters()
    gp_values = run_gp_model(model, sigma, r_loc, r_year, sigma_e, sigma_b)
    indices = model["test_indices"]

    show_model_evaluation(real_values, pred_values, gp_values)

    pred_err = np.abs(pred_values - real_values)
    gp_err = np.abs(gp_values - real_values)

    pred_dict = values_to_dict(indices, pred_values)
    pred_err_dict = values_to_dict(indices, pred_err)
    gp_dict = values_to_dict(indices, gp_values)
    gp_err_dict = values_to_dict(indices, gp_err)
    real_dict = values_to_dict(indices, real_values)

    pred_result_df = get_real_pred_df(code_regency_map, real_dict, pred_dict, pred_err_dict)
    gp_result_df = get_real_pred_df(code_regency_map, real_dict, gp_dict, gp_err_dict)
    show_predict_map(pred_result_df, gp_result_df)

if __name__ == "__main__":
    main()