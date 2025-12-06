import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Загрузка данных
@st.cache
def load_data():
    data = pd.read_csv( 'nyc-rolling-sales.csv')
    return data

# Очистка данных
def clean_data(df):
    # Преобразование SALE PRICE, LAND SQUARE FEET, GROSS SQUARE FEET в числовой формат
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].str.replace(',', '').str.strip(), errors='coerce')
    df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    df = df.dropna(subset=['SALE PRICE'])  # Удаление строк с отсутствующими значениями в SALE PRICE
    return df

# Загрузка и очистка данных
data = load_data()
data = clean_data(data)

# Навигация
st.sidebar.title('Навигация')
page = st.sidebar.radio('Выберите страницу:', ['Главная', 'Анализ по району', 'Статистика'])

if page == 'Главная':
    st.title('Добро пожаловать в дашборд по продажам недвижимости в Нью-Йорке!')
    st.write('Этот дашборд позволяет анализировать продажи недвижимости по различным параметрам.')

elif page == 'Анализ по району':
    st.title('Анализ по району')
    
    # Выбор района для анализа
    neighborhoods = data['NEIGHBORHOOD'].unique()
    selected_neighborhood = st.selectbox('Выберите район', neighborhoods)

    # Фильтрация данных по выбранному району
    filtered_data = data[data['NEIGHBORHOOD'] == selected_neighborhood]

    # Вычисление средней цены продажи
    average_price = filtered_data.groupby('YEAR BUILT')['SALE PRICE'].mean().reset_index()

    # Визуализация
    st.subheader(f'Средняя цена продажи в районе {selected_neighborhood}')
    plt.figure(figsize=(10, 5))
    plt.plot(average_price['YEAR BUILT'], average_price['SALE PRICE'], marker='o')
    plt.title(f'Средняя цена продажи по годам в районе {selected_neighborhood}')
    plt.xlabel('Год постройки')
    plt.ylabel('Средняя цена продажи')
    plt.grid()
    st.pyplot(plt)

elif page == 'Статистика':
    st.title('Статистика по продажам')
    st.subheader('Общая статистика по данным')
    st.write(data.describe())

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
