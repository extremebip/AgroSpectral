import streamlit as st
import pandas as pd

@st.cache_data
def get_yield_df():
  return pd.read_csv("yield.csv")

def filter_yield_df(yield_df, filtered_regencies, filtered_years):
  return yield_df.loc[(yield_df['Regency'].isin(filtered_regencies)) & (yield_df['Year'].isin(filtered_years))]