import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

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
page = st.sidebar.radio('Выберите страницу:', ['Обзор', 'Таблица данных'])

if page == 'Обзор':
    st.title('Обзор датасета')
    st.write('Первые строки данных:')
    st.dataframe(data.head())

elif page == 'Таблица данных':
    st.title('Интерактивная таблица данных')

    # Создание конфигурации для таблицы
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationPageSize=10)  # Пагинация
    gb.configure_default_column(editable=False, groupable=True)  # Настройки столбцов
    grid_options = gb.build()

    # Отображение таблицы
    AgGrid(data, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True)

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
