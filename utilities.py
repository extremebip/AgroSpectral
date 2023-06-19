import torch
import streamlit as st
import pandas as pd

@st.cache_data
def get_yield_df():
  return pd.read_csv("data/yield.csv")

@st.cache_resource
def get_model():
  return torch.load("data/model.pth.tar", map_location="cpu")

def filter_yield_df(yield_df, filtered_regencies, filtered_years):
  return yield_df.loc[(yield_df['Regency'].isin(filtered_regencies)) & (yield_df['Year'].isin(filtered_years))]