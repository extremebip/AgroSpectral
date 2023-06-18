import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from utilities import *

def main():
    st.set_page_config(
        page_title="Beranda",
        page_icon="🏠",
    )
    
    st.write("# Main Page")

if __name__ == "__main__":
    main()