import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder  # Импорт необходимых компонентов для AgGrid

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
page = st.sidebar.radio('Страница 1:', ['Таблица с первыми строками', 'Базовые статистики', 'Гистограммы и Bar Charts', 
                                               'Корреляционная матрица', 'Scatter Plots и Pie Charts', 'Фильтры'])

page2 = st.sidebar.radio('Страница 2:', ['Графики моделей', 'Ключевые метрики', 'Интерпретация', 
                                               'Примеры визуализаций', 'Фильтры'])

if page == 'Таблица с первыми строками':
    st.title('Таблица данных')

    # Создание конфигурации для таблицы
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationPageSize=10)  # Пагинация
    gb.configure_default_column(editable=False, groupable=True)  # Настройки столбцов
    grid_options = gb.build()

    # Отображение таблицы
    AgGrid(data, gridOptions=grid_options, enable_enterprise_modules=True, allow_unsafe_jscode=True)

elif page == 'Базовые статистики':
    st.title('Ключевые показатели эффективности (KPI)')

    # Общее количество записей
    total_records = data.shape[0]
    st.metric(label="Общее количество записей", value=total_records)

    # Пропуски по колонкам
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        st.subheader('Пропуски по колонкам:')
        st.write(missing_values)

    # Базовые статистики для числовых колонок
    st.subheader('Базовые статистики для числовых колонок:')
    stats = data[numeric_columns].describe().T[['mean', '50%', 'std']]
    stats.columns = ['Среднее', 'Медиана', 'Стандартное отклонение']
    
    # Визуализация KPI в виде карточек
    for column in numeric_columns:
        mean_value = stats.loc[column, 'Среднее']
        median_value = stats.loc[column, 'Медиана']
        std_value = stats.loc[column, 'Стандартное отклонение']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f'Среднее {column}', value=f"{mean_value:.2f}")
        with col2:
            st.metric(label=f'Медиана {column}', value=f"{median_value:.2f}")
        with col3:
            st.metric(label=f'Стандартное отклонение {column}', value=f"{std_value:.2f}")

elif page == 'Гистограммы и Bar Charts':
    st.title('Гистограммы для числовых переменных')
    
    # Выбор числового признака
    selected_numeric = st.selectbox('Выберите числовой признак:', numeric_columns)
    
    # Гистограмма
    st.subheader(f'Гистограмма для {selected_numeric}')
    fig_hist = px.histogram(data, x=selected_numeric, title=f'Гистограмма {selected_numeric}')
    st.plotly_chart(fig_hist)

    st.title('Pie Charts для пропорций категорий')
    
    # Выбор категориального признака
    selected_categorical = st.selectbox('Выберите категориальный признак:', categorical_columns)
    
    # Подсчет пропорций
    counts = data[selected_categorical].value_counts()
    
    # Pie Chart
    st.subheader(f'Pie Chart для {selected_categorical}')
    fig_pie = px.pie(counts, values=counts.values, names=counts.index, title=f'Pie Chart для {selected_categorical}')
    st.plotly_chart(fig_pie) 

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

elif page == 'Scatter Plots и Pie Charts':
    st.title('Scatter Plots для пар признаков')
    
    # Выбор пар числовых признаков
    selected_x = st.selectbox('Выберите признак по оси X:', numeric_columns)
    selected_y = st.selectbox('Выберите признак по оси Y:', numeric_columns)
    
    # Scatter Plot
    st.subheader(f'Scatter Plot: {selected_x} vs {selected_y}')
    fig_scatter = px.scatter(data, x=selected_x, y=selected_y, title=f'Scatter Plot: {selected_x} vs {selected_y}')
    st.plotly_chart(fig_scatter)

    st.title('Bar Charts для категориальных переменных')
    
    # Выбор категориального признака
    selected_categorical = st.selectbox('Выберите категориальный признак:', categorical_columns)
    
    # Bar Chart
    st.subheader(f'Bar Chart для {selected_categorical}')
    fig_bar = px.bar(data, x=selected_categorical, title=f'Bar Chart {selected_categorical}', 
                     color=selected_categorical, 
                     text_auto=True)
    st.plotly_chart(fig_bar)

elif page == 'Фильтры':
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

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
