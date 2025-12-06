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
    # Преобразование столбцов в числовой формат
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].str.replace(',', '').str.strip(), errors='coerce')
    df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'].str.replace(',', '').str.strip(), errors='coerce')
    
    # Преобразование SALE DATE в datetime
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
    
    # Удаление строк с пропусками
    df.dropna(inplace=True)
    
    return df

# Основной код приложения
data = load_data()
data = clean_data(data)

# Определение числовых и категориальных признаков
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

# Навигация
st.sidebar.title('Навигация')
page = st.sidebar.radio('Выберите страницу:', ['Корреляционная матрица', 'Гистограммы', 'Box Plots', 'Bar Charts', 'Scatter Plots', 'Pie Charts', 'Фильтры'])

# Фильтры
if page == 'Фильтры':
    st.title('Фильтры для данных')

    # Dropdown для выбора колонки
    selected_column = st.selectbox('Выберите колонку:', numeric_columns + categorical_columns)

    # Sliders для диапазонов значений (только для числовых колонок)
    if selected_column in numeric_columns:
        min_value = float(data[selected_column].min())
        max_value = float(data[selected_column].max())
        range_values = st.slider('Выберите диапазон значений:', min_value, max_value, (min_value, max_value))
        filtered_data = data[(data[selected_column] >= range_values[0]) & (data[selected_column] <= range_values[1])]
    else:
        filtered_data = data

    # Date pickers для временных данных
    if selected_column == 'SALE DATE':
        start_date = st.date_input('Выберите начальную дату:', data['SALE DATE'].min().date())
        end_date = st.date_input('Выберите конечную дату:', data['SALE DATE'].max().date())
        filtered_data = filtered_data[(filtered_data['SALE DATE'] >= pd.to_datetime(start_date)) & (filtered_data['SALE DATE'] <= pd.to_datetime(end_date))]

    st.subheader('Отфильтрованные данные:')
    st.write(filtered_data)

elif page == 'Корреляционная матрица':
    st.title('Корреляционная матрица')
    
    # Удаление нечисловых столбцов
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Вычисление корреляционной матрицы
    corr = numeric_data.corr()
    
    # Настройка визуализации
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Корреляционная матрица')

    # Отображение heatmap в Streamlit
    st.pyplot(plt)

elif page == 'Гистограммы':
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

elif page == 'Scatter Plots':
    st.title('Scatter Plots для пар признаков')
    
    # Выбор пар числовых признаков
    selected_x = st.selectbox('Выберите признак по оси X:', numeric_columns)
    selected_y = st.selectbox('Выберите признак по оси Y:', numeric_columns)
    
    # Scatter Plot
    st.subheader(f'Scatter Plot: {selected_x} vs {selected_y}')
    fig_scatter = px.scatter(data, x=selected_x, y=selected_y, title=f'Scatter Plot: {selected_x} vs {selected_y}')
    st.plotly_chart(fig_scatter)

elif page == 'Pie Charts':
    st.title('Pie Charts для пропорций категорий')
    
    # Выбор категориального признака
    selected_categorical = st.selectbox('Выберите категориальный признак:', categorical_columns)
    
    # Подсчет пропорций
    counts = data[selected_categorical].value_counts()
    
    # Pie Chart
    st.subheader(f'Pie Chart для {selected_categorical}')
    fig_pie = px.pie(counts, values=counts.values, names=counts.index, title=f'Pie Chart для {selected_categorical}')
    st.plotly_chart(fig_pie)

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
