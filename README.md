# AgroSpectral

Agrospectral is a website application that shows predictions of **Rice Yield Production** in the province of West Java, Indonesia made using Python and Streamlit. By using remote sensing data, specifically MODIS dataset, Agrospectral applies **Convolutional Neural Network** model and **Gaussian Process** model to predict rice yield production total. In this web application, Agrospectral provides several visualization method, such as histogram, barchart, and interactive map.

## Installation
You need to install all required packages which are listed in the *requirements.txt* to run this web app.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the required package.

```bash
pip install -r requirements.txt
```

## Run Application on Local Environment
After installing all the required libraries, you can run this application on your local machine by running this command. Please make sure that you add **streamlit** command to your PATH environment variable.

```bash
streamlit run 🏠_Beranda.py
```