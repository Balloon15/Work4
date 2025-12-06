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
    df.dropna(inplace=True)
    return df

# Основной код приложения
data = load_data()
data = clean_data(data)

# Навигация
st.sidebar.title('Навигация')
page = st.sidebar.radio('Выберите страницу:', ['Визуализация исходных данных', 'Результаты анализа'])

if page == 'Визуализация исходных данных':
    st.write('Первые строки данных:')
    st.dataframe(data.head())

    st.write('Общее количество записей:', data.shape[0])
    st.write('Количество пропусков по колонкам:')
    st.write(data.isnull().sum())

    st.title('Базовые статистики')
    
    # Статистики
    st.subheader('Ключевые показатели')
    st.metric(label="Общее количество записей", value=data.shape[0])
    st.metric(label="Средняя цена продажи", value=data['SALE PRICE'].mean())
    st.metric(label="Медиана цены продажи", value=data['SALE PRICE'].median())
    st.metric(label="Стандартное отклонение цены продажи", value=data['SALE PRICE'].std())

    st.title('Визуализации данных')

    # Выбор признака для визуализации
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Фильтры
    selected_numeric = st.selectbox('Выберите числовой признак для гистограммы:', numeric_columns)
    selected_categorical = st.selectbox('Выберите категориальный признак для bar chart:', categorical_columns)

    # Гистограмма
    st.subheader(f'Гистограмма для {selected_numeric}')
    fig_hist = px.histogram(data, x=selected_numeric, title=f'Гистограмма {selected_numeric}')
    st.plotly_chart(fig_hist)

    # Box Plot
    st.subheader(f'Box Plot для {selected_numeric}')
    fig_box = px.box(data, y=selected_numeric, title=f'Box Plot {selected_numeric}')
    st.plotly_chart(fig_box)

    # Bar Chart
    st.subheader(f'Bar Chart для {selected_categorical}')
    fig_bar = px.bar(data, x=selected_categorical, title=f'Bar Chart {selected_categorical}')
    st.plotly_chart(fig_bar)

    # Корреляционная матрица
    st.subheader('Корреляционная матрица')
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    st.pyplot(plt)

    # Scatter Plot
    x_axis = st.selectbox('Выберите признак для оси X:', numeric_columns)
    y_axis = st.selectbox('Выберите признак для оси Y:', numeric_columns)
    
    st.subheader(f'Scatter Plot между {x_axis} и {y_axis}')
    fig_scatter = px.scatter(data, x=x_axis, y=y_axis, title=f'Scatter Plot между {x_axis} и {y_axis}')
    st.plotly_chart(fig_scatter)

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
