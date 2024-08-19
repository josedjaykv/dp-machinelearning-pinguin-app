import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'): # .expander es para hacer un desplegable
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X = df.drop('species', axis=1) # Eliminamos la columna especies indicando que es una columna(axis=1)
    X

    st.write('**y**')
    y = df.species
    y

    st.write('Diferences types of species')
    species1 = df.species.unique()
    species1

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Data preparations
with st.sidebar:
    st.header('Input Features')
    # "species","","","","","",""
    island = st.selectbox(
        'Island',
        (X.island.unique()), 
        index=None,
        placeholder='Select an Island'
    )
    gender = st.selectbox(
        'Gender',
        (X.sex.unique()),
        index=None,
        placeholder='Select a gender'
    )
    bill_length_mm = st.slider('Bill length (mm)', 30.0, 60.0, 43.9, step=0.05)
    bill_depth_mm = st.slider('Bill depth (mm)', 10.0, 30.0, 17.2, step=0.05)
    flipper_length_mm = st.slider('Flipper length (mm)', 160.0, 240.0, 200.2, step=0.05)
    body_mass_g = st.slider('Body mass (mm)', 2700.0, 6300.0, 4200.0, step=0.05)




