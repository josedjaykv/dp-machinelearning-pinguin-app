import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'): # .expander es para hacer un desplegable
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species', axis=1) # Eliminamos la columna especies indicando que es una columna(axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.species
    y_raw

    st.write('Diferences types of species')
    species1 = df.species.unique()
    species1

with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')



# Input features
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', (X_raw.island.unique()))
    bill_length_mm = st.slider('Bill length (mm)', 30.0, 60.0, 43.9, step=0.05)
    bill_depth_mm = st.slider('Bill depth (mm)', 10.0, 30.0, 17.2, step=0.05)
    flipper_length_mm = st.slider('Flipper length (mm)', 160.0, 240.0, 200.2, step=0.05)
    body_mass_g = st.slider('Body mass (mm)', 2700.0, 6300.0, 4200.0, step=0.05)
    gender = st.selectbox('Gender', (X_raw.sex.unique()))

    # Create a Data Frame for the input features
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
    st.write('**Input penguin**')
    input_df
    st.write('**Combined penguin data**')
    input_penguins



## Data preparation
# Ecode X
econde = ['island', 'gender']
df_penguins = pd.get_dummies(input_penguins, prefix=econde)
input_row = df_penguins[:1]

# Econde y
# Lo que estamos haciendo aquí es asignarle un número a cada tipo de especie
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    input_row
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('Enconde y')
        y
    with right_column:
        st.write('No enconde')    
        y_raw
