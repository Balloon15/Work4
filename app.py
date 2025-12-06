import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Загрузка данных
@st.cache
def load_data():
    data = pd.read_csv('nyc-rolling-sales.csv')
    return data

# Очистка данных
def clean_data(df):
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].str.replace(',', '').str.strip(), errors='coerce')
    df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    df.dropna(inplace=True)
    return df

# Основной код приложения
data = load_data()
data = clean_data(data)

# Навигация
st.sidebar.title('Навигация')
page = st.sidebar.radio('Выберите страницу:', ['Гистограммы', 'Box Plots', 'Bar Charts'])

# Определение числовых и категориальных признаков
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

if page == 'Гистограммы':
    st.title('Гистограммы для числовых переменных')
    
    # Выбор числового признака
    selected_numeric = st.selectbox('Выберите числовой признак:', numeric_columns)
    
    # Гистограмма
    st.subheader(f'Гистограмма для {selected_numeric}')
    fig_hist = px.histogram(data, x=selected_numeric, title=f'Гистограмма {selected_numeric}')
    st.plotly_chart(fig_hist)

elif page == 'Box Plots':
    st.title('Box Plots для числовых переменных')
    
    # Выбор числового признака
    selected_numeric = st.selectbox('Выберите числовой признак для Box Plot:', numeric_columns)
    
    # Box Plot
    st.subheader(f'Box Plot для {selected_numeric}')
    fig_box = px.box(data, y=selected_numeric, title=f'Box Plot {selected_numeric}')
    st.plotly_chart(fig_box)

elif page == 'Bar Charts':
    st.title('Bar Charts для категориальных переменных')
    
    # Выбор категориального признака
    selected_categorical = st.selectbox('Выберите категориальный признак:', categorical_columns)
    
    # Bar Chart
    st.subheader(f'Bar Chart для {selected_categorical}')
    fig_bar = px.bar(data, x=selected_categorical, title=f'Bar Chart {selected_categorical}', 
                     color=selected_categorical, 
                     text_auto=True)
    st.plotly_chart(fig_bar)

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
