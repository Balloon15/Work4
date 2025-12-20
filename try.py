import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="NYC Property Sales Dashboard",    
    layout="wide",
    initial_sidebar_state="expanded"
)

# Словарь переводов названий колонок на русский
COLUMN_TRANSLATIONS = {
    # Основные идентификаторы
    'Unnamed: 0': 'ID',
    'BOROUGH': 'Городской округ',
    'NEIGHBORHOOD': 'Район',
    'BUILDING CLASS CATEGORY': 'Категория класса здания',
    'TAX CLASS AT PRESENT': 'Налоговый класс (текущий)',
    'BLOCK': 'Блок',
    'LOT': 'Участок',
    'EASE-MENT': 'Сервитут',
    'BUILDING CLASS AT PRESENT': 'Класс здания (текущий)',
    
    # Адресная информация
    'ADDRESS': 'Адрес',
    'APARTMENT NUMBER': 'Номер квартиры',
    'ZIP CODE': 'Почтовый индекс',
    
    # Характеристики здания
    'RESIDENTIAL UNITS': 'Жилые единицы',
    'COMMERCIAL UNITS': 'Коммерческие единицы',
    'TOTAL UNITS': 'Всего единиц',
    'LAND SQUARE FEET': 'Площадь земли (кв. фут)',
    'GROSS SQUARE FEET': 'Общая площадь (кв. фут)',
    'YEAR BUILT': 'Год постройки',
    
    # Информация о продаже
    'TAX CLASS AT TIME OF SALE': 'Налоговый класс (на момент продажи)',
    'BUILDING CLASS AT TIME OF SALE': 'Класс здания (на момент продажи)',
    'SALE PRICE': 'Цена продажи',
    'SALE DATE': 'Дата продажи',
}

# Функция для перевода названий колонок
def translate_columns(df):
    df = df.copy()
    translated_cols = []
    for col in df.columns:
        # Убираем лишние пробелы в названиях колонок
        col_stripped = col.strip()
        translated_cols.append(COLUMN_TRANSLATIONS.get(col_stripped, col_stripped))
    df.columns = translated_cols
    return df

# Функция для обратного перевода
def reverse_translate_column(russian_name):
    for eng, rus in COLUMN_TRANSLATIONS.items():
        if rus == russian_name:
            return eng
    return russian_name

# Загрузка данных с очисткой выбросов
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("nyc-rolling-sales.csv")
        
        # Убираем лишние пробелы в названиях колонок
        data.columns = data.columns.str.strip()
        
        numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                          'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                          'TOTAL UNITS']
        
        for col in numeric_columns:
            if col in data.columns:
                # Преобразуем строки в числа, заменяя нечисловые значения на NaN
                data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
        
        if 'SALE DATE' in data.columns:
            data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
        
        # ОЧИСТКА ВЫБРОСОВ В ЦЕНАХ
        if 'SALE PRICE' in data.columns:
            # 1. Удаляем нулевые и отрицательные цены
            data = data[data['SALE PRICE'] > 0]
            
            # 2. Удаляем слишком низкие цены (< $10,000)
            data = data[data['SALE PRICE'] >= 10000]
            
            # 3. Удаляем экстремально высокие цены (> $50 миллионов)
            data = data[data['SALE PRICE'] <= 50_000_000]
            
            # 4. Статистическая очистка (IQR метод)
            q1 = data['SALE PRICE'].quantile(0.25)
            q3 = data['SALE PRICE'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data = data[(data['SALE PRICE'] >= lower_bound) & (data['SALE PRICE'] <= upper_bound)]
        
        # Очистка года постройки
        if 'YEAR BUILT' in data.columns:
            current_year = datetime.now().year
            data = data[(data['YEAR BUILT'] >= 1700) & (data['YEAR BUILT'] <= current_year)]
            data = data[data['YEAR BUILT'] > 0]
        
        # Очистка площади - ОЧЕНЬ ВАЖНО для цены за кв.фут
        if 'GROSS SQUARE FEET' in data.columns:
            # Удаляем нулевые и отрицательные площади
            data = data[data['GROSS SQUARE FEET'] > 0]
            # Удаляем слишком маленькие площади (менее 100 кв.фут)
            data = data[data['GROSS SQUARE FEET'] >= 100]
            # Удаляем слишком большие площади
            data = data[data['GROSS SQUARE FEET'] <= 1000000]
        
        if 'LAND SQUARE FEET' in data.columns:
            data = data[data['LAND SQUARE FEET'] > 0]
            data = data[data['LAND SQUARE FEET'] >= 100]
            data = data[data['LAND SQUARE FEET'] <= 1000000]
        
        return data
        
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return pd.DataFrame()

# Загружаем данные
df = load_data()

if df.empty:
    st.error("Данные не загружены. Проверьте наличие файла 'nyc-rolling-sales.csv'.")
    st.stop()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ рынка", "Прогнозные модели", "Таблица переводов"]
)

# Добавляем фильтры в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("Фильтры данных")

# Создаем копию с русскими названиями для фильтров
df_russian = translate_columns(df.copy())

# Фильтр по району
if 'NEIGHBORHOOD' in df.columns:
    neighborhoods = ['Все'] + sorted(df['NEIGHBORHOOD'].dropna().unique().tolist())
    selected_neighborhood = st.sidebar.selectbox(
        COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'), 
        neighborhoods
    )
else:
    selected_neighborhood = 'Все'

# Фильтр по типу здания
if 'BUILDING CLASS CATEGORY' in df.columns:
    building_classes = ['Все'] + sorted(df['BUILDING CLASS CATEGORY'].dropna().unique().tolist())
    selected_building_class = st.sidebar.selectbox(
        COLUMN_TRANSLATIONS.get('BUILDING CLASS CATEGORY', 'Категория класса здания'), 
        building_classes
    )
else:
    selected_building_class = 'Все'

# Фильтр по году постройки
year_range = (1800, 2023)
if 'YEAR BUILT' in df.columns:
    valid_years = df[df['YEAR BUILT'] > 0]['YEAR BUILT']
    
    if not valid_years.empty:
        min_year = int(max(valid_years.min(), 1700))
        max_year = int(min(valid_years.max(), datetime.now().year))
        
        year_range = st.sidebar.slider(
            "Год постройки",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (1800, 2023)

# Фильтр по цене
price_range = (10000, 5000000)
if 'SALE PRICE' in df.columns:
    realistic_min_price = int(max(df['SALE PRICE'].min(), 10000))
    realistic_max_price = int(min(df['SALE PRICE'].max(), 5000000))
    
    price_range = st.sidebar.slider(
        "Цена продажи ($)",
        min_value=realistic_min_price,
        max_value=realistic_max_price,
        value=(realistic_min_price, realistic_max_price),
        step=10000
    )

# Применяем фильтры
filtered_df = df.copy()

if selected_neighborhood != 'Все' and 'NEIGHBORHOOD' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['NEIGHBORHOOD'] == selected_neighborhood]

if selected_building_class != 'Все' and 'BUILDING CLASS CATEGORY' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['BUILDING CLASS CATEGORY'] == selected_building_class]

if 'YEAR BUILT' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['YEAR BUILT'] >= year_range[0]) & 
        (filtered_df['YEAR BUILT'] <= year_range[1])
    ]

if 'SALE PRICE' in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df['SALE PRICE'] >= price_range[0]) & 
        (filtered_df['SALE PRICE'] <= price_range[1])
    ]

# Создаем производные колонки с ОЧЕНЬ ВАЖНЫМИ ПРОВЕРКАМИ
if 'SALE DATE' in filtered_df.columns:
    filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.month
    filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
    
# Цена за квадратный фут - ОЧЕНЬ ВАЖНО ИСПРАВИТЬ ПРОБЛЕМУ С ДЕЛЕНИЕМ
if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
    # Убедимся, что нет нулевых или очень маленьких площадей
    temp_df = filtered_df[(filtered_df['GROSS SQUARE FEET'] > 10) & (filtered_df['SALE PRICE'] > 0)].copy()
    if not temp_df.empty:
        filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET']
        # Заменяем бесконечные значения и NaN
        filtered_df['PRICE_PER_SQFT'] = filtered_df['PRICE_PER_SQFT'].replace([np.inf, -np.inf], np.nan)
        # Удаляем аномальные значения цены за кв.фут
        filtered_df = filtered_df[(filtered_df['PRICE_PER_SQFT'] > 1) & (filtered_df['PRICE_PER_SQFT'] < 5000)]
    else:
        filtered_df['PRICE_PER_SQFT'] = np.nan
    
if 'YEAR BUILT' in filtered_df.columns:
    filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']
    # Ограничиваем возраст реалистичными значениями
    filtered_df = filtered_df[(filtered_df['BUILDING_AGE'] >= 0) & (filtered_df['BUILDING_AGE'] <= 300)]

# Создаем DataFrame с русскими названиями для отображения
filtered_df_russian = translate_columns(filtered_df.copy())

# Страница 4: Таблица переводов
if page == "Таблица переводов":
    st.title("Таблица переводов названий колонок")
    
    translation_table = pd.DataFrame({
        'Оригинальное название (англ.)': list(COLUMN_TRANSLATIONS.keys()),
        'Перевод (рус.)': list(COLUMN_TRANSLATIONS.values())
    })
    
    st.dataframe(
        translation_table,
        use_container_width=True,
        height=600
    )
    
    st.markdown("---")
    st.write(f"Всего строк в данных: {len(df)}")
    st.write(f"Отфильтровано строк: {len(filtered_df)}")

# Страница 1: Визуализация данных
elif page == "Визуализация данных":
    st.title("Визуализация данных о продажах недвижимости Нью-Йорка")
    
    if filtered_df.empty:
        st.warning("Нет данных, соответствующих выбранным фильтрам.")
    else:
        # KPI карточки
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Всего записей", len(filtered_df))
        
        with col2:
            if 'SALE PRICE' in filtered_df.columns:
                median_price = filtered_df['SALE PRICE'].median()
                st.metric("Медианная цена ($)", f"{median_price:,.0f}")
        
        with col3:
            if 'SALE DATE' in filtered_df.columns:
                unique_months = filtered_df['SALE_MONTH'].nunique()
                st.metric("Месяцев данных", unique_months)
        
        with col4:
            if 'NEIGHBORHOOD' in filtered_df.columns:
                unique_neighborhoods = filtered_df['NEIGHBORHOOD'].nunique()
                st.metric("Количество районов", unique_neighborhoods)

        st.markdown("---")
        
        # Таблица с данными
        st.subheader("Просмотр данных")
        
        # Выбор колонок для отображения
        all_columns_russian = filtered_df_russian.columns.tolist()
        selected_columns_russian = st.multiselect(
            "Выберите колонки для отображения:",
            all_columns_russian,
            default=all_columns_russian[:min(8, len(all_columns_russian))]
        )
        
        # Пагинация
        page_size = st.selectbox("Строк на странице:", [10, 25, 50, 100])
        page_number = st.number_input("Номер страницы:", min_value=1, value=1)
        
        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        
        if selected_columns_russian:
            display_df = filtered_df_russian[selected_columns_russian].iloc[start_idx:end_idx]
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        
        # Экспорт данных
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать отфильтрованные данные (CSV)",
            data=csv,
            file_name="filtered_nyc_property_sales.csv",
            mime="text/csv",
        )
        
        st.markdown("---")
        
        # Базовые статистики
        st.subheader("Базовая статистика")
        
        if st.checkbox("Показать статистики по числовым колонкам"):
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_df = filtered_df[numeric_cols].describe().T
                stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                stats_df.columns = ['Кол-во', 'Среднее', 'Стд. откл.', 'Мин.', '25%', 'Медиана', '75%', 'Макс.']
                
                # Преобразуем индексы в русские названия
                stats_df_index_russian = []
                for idx in stats_df.index:
                    rus_name = COLUMN_TRANSLATIONS.get(idx, idx)
                    stats_df_index_russian.append(rus_name)
                stats_df.index = stats_df_index_russian
                
                st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
        
        st.markdown("---")
            
        # Визуализации
        col1, col2 = st.columns(2)
        
        with col1:
            if 'SALE PRICE' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df, 
                    x='SALE PRICE',
                    nbins=50,
                    title="Распределение цен на недвижимость",
                    labels={'SALE PRICE': 'Цена продажи ($)'}
                )
                fig.update_layout(xaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
                
            if 'YEAR BUILT' in filtered_df.columns:
                valid_year_data = filtered_df[filtered_df['YEAR BUILT'] > 0]
                if not valid_year_data.empty:
                    fig = px.histogram(
                        valid_year_data,
                        x='YEAR BUILT',
                        nbins=30,
                        title="Распределение по году постройки",
                        labels={'YEAR BUILT': 'Год'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'BOROUGH' in filtered_df.columns:
                borough_names = {
                    1: 'Manhattan',
                    2: 'Brooklyn', 
                    3: 'Queens',
                    4: 'Bronx',
                    5: 'Staten Island'
                }
                filtered_df['BOROUGH_NAME'] = filtered_df['BOROUGH'].map(borough_names)
                
                borough_counts = filtered_df['BOROUGH_NAME'].value_counts()
                fig = px.pie(
                    values=borough_counts.values,
                    names=borough_counts.index,
                    title="Распределение продаж по округам",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
                
            if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
                # Удаляем выбросы для лучшей визуализации
                scatter_data = filtered_df.copy()
                if len(scatter_data) > 1000:
                    scatter_data = scatter_data.sample(1000)
                
                fig = px.scatter(
                    scatter_data,
                    x='GROSS SQUARE FEET',
                    y='SALE PRICE',
                    title="Цена vs Общая площадь",
                    labels={
                        'GROSS SQUARE FEET': 'Площадь (кв. фут)',
                        'SALE PRICE': 'Цена ($)'
                    },
                    opacity=0.6
                )
                fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
        
        # Сезонность внутри года
        if 'SALE_MONTH' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
            st.markdown("---")
            st.subheader("Сезонные паттерны внутри года")
            
            monthly_stats = filtered_df.groupby('SALE_MONTH').agg({
                'SALE PRICE': ['median', 'count'],
                'GROSS SQUARE FEET': 'median'
            }).reset_index()
            
            monthly_stats.columns = ['Месяц', 'Медианная цена', 'Количество продаж', 'Медианная площадь']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    monthly_stats,
                    x='Месяц',
                    y='Количество продаж',
                    title='Количество продаж по месяцам',
                    color='Количество продаж'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    monthly_stats,
                    x='Месяц',
                    y='Медианная цена',
                    title='Медианная цена по месяцам',
                    markers=True
                )
                fig.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)

# Страница 2: Анализ рынка
elif page == "Анализ рынка":
    st.title("Анализ рынка недвижимости Нью-Йорка")
    
    if filtered_df.empty:
        st.warning("Нет данных для анализа.")
    else:
        analysis_type = st.selectbox(
            "Выберите тип анализа:",
            ["Анализ по районам", "Анализ по типам зданий", "Стоимость квадратного фута", "Возраст vs Цена"]
        )
        
        if analysis_type == "Анализ по районам":
            st.subheader("Сравнение районов")
            
            if 'NEIGHBORHOOD' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
                # Топ-15 районов по медианной цене
                neighborhood_stats = filtered_df.groupby('NEIGHBORHOOD').agg({
                    'SALE PRICE': ['median', 'count'],
                    'GROSS SQUARE FEET': 'median'
                }).round(2)
                
                neighborhood_stats.columns = ['Медианная цена', 'Количество продаж', 'Медианная площадь']
                
                # Добавляем цену за кв.фут (с проверкой деления на ноль)
                neighborhood_stats['Цена за кв.фут'] = np.where(
                    neighborhood_stats['Медианная площадь'] > 0,
                    neighborhood_stats['Медианная цена'] / neighborhood_stats['Медианная площадь'],
                    np.nan
                )
                neighborhood_stats = neighborhood_stats.dropna(subset=['Цена за кв.фут'])
                
                # Сортируем по медианной цене
                top_neighborhoods = neighborhood_stats.sort_values('Медианная цена', ascending=False).head(15)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        top_neighborhoods.reset_index(),
                        x='NEIGHBORHOOD',
                        y='Медианная цена',
                        title='Топ-15 районов по медианной цене',
                        color='Медианная цена'
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(yaxis_tickformat=',')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(neighborhood_stats) > 1:
                        fig = px.scatter(
                            neighborhood_stats.reset_index(),
                            x='Количество продаж',
                            y='Медианная цена',
                            size='Количество продаж',
                            color='Цена за кв.фут',
                            hover_name='NEIGHBORHOOD',
                            title='Соотношение цены и количества продаж',
                            size_max=40
                        )
                        fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Анализ по типам зданий":
            st.subheader("Анализ по типам недвижимости")
            
            if 'BUILDING CLASS CATEGORY' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
                building_stats = filtered_df.groupby('BUILDING CLASS CATEGORY').agg({
                    'SALE PRICE': ['median', 'count'],
                    'GROSS SQUARE FEET': 'median'
                }).round(2)
                
                building_stats.columns = ['Медианная цена', 'Количество продаж', 'Медианная площадь']
                
                # Добавляем цену за кв.фут
                building_stats['Цена за кв.фут'] = np.where(
                    building_stats['Медианная площадь'] > 0,
                    building_stats['Медианная цена'] / building_stats['Медианная площадь'],
                    np.nan
                )
                building_stats = building_stats.dropna(subset=['Цена за кв.фут'])
                
                # Топ-10 типов по цене за кв.фут
                top_buildings = building_stats.nlargest(10, 'Цена за кв.фут')
                
                fig = px.bar(
                    top_buildings.reset_index(),
                    x='BUILDING CLASS CATEGORY',
                    y='Цена за кв.фут',
                    title='Топ-10 самых дорогих типов недвижимости (цена за кв.фут)',
                    color='Цена за кв.фут'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
                fig.update_layout(yaxis_tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Стоимость квадратного фута":
                st.subheader("Анализ стоимости квадратного фута")
                
                if 'PRICE_PER_SQFT' in filtered_df.columns and not filtered_df['PRICE_PER_SQFT'].isna().all():
                    # Создаем копию для анализа
                    analysis_df = filtered_df[filtered_df['PRICE_PER_SQFT'].notna()].copy()
                    
                    # Рассчитываем медианную цену за кв.фут
                    median_price_sqft = analysis_df['PRICE_PER_SQFT'].median()
                    
                    # Определяем верхнюю границу для выбросов
                    q3 = analysis_df['PRICE_PER_SQFT'].quantile(0.75)
                    q1 = analysis_df['PRICE_PER_SQFT'].quantile(0.25)
                    iqr = q3 - q1
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Ограничиваем максимальную реалистичную цену
                    reasonable_max = 3000
                    final_upper_bound = min(upper_bound, reasonable_max)
                    
                    # Заменяем аномально большие выбросы на медианную цену
                    processed_df = analysis_df.copy()
                    large_outliers_mask = processed_df['PRICE_PER_SQFT'] > final_upper_bound
                    processed_df.loc[large_outliers_mask, 'PRICE_PER_SQFT'] = median_price_sqft
                    
                    # ТОЛЬКО ДВА ГРАФИКА
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Гистограмма распределения
                        fig = px.histogram(
                            processed_df,
                            x='PRICE_PER_SQFT',
                            nbins=50,
                            title="Распределение цены за кв.фут",
                            labels={'PRICE_PER_SQFT': 'Цена за кв.фут ($)'},
                            color_discrete_sequence=['#636EFA']
                        )
                        fig.add_vline(
                            x=median_price_sqft,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Медиана: ${median_price_sqft:.0f}"
                        )
                        fig.update_layout(
                            xaxis_range=[0, 1500],
                            xaxis_title="Цена за кв.фут ($)",
                            yaxis_title="Количество объектов"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # График по округам
                        if 'BOROUGH' in processed_df.columns:
                            borough_map = {
                                1: 'Manhattan',
                                2: 'Brooklyn', 
                                3: 'Queens',
                                4: 'Bronx',
                                5: 'Staten Island'
                            }
                            
                            temp_df = processed_df.copy()
                            temp_df['BOROUGH_NAME'] = temp_df['BOROUGH'].map(borough_map)
                            
                            # Группируем и рассчитываем медиану
                            borough_stats = temp_df.groupby('BOROUGH_NAME')['PRICE_PER_SQFT'].agg(['median', 'count']).reset_index()
                            borough_stats.columns = ['Округ', 'Медианная цена', 'Количество']
                            borough_stats = borough_stats.sort_values('Медианная цена', ascending=False)
                            
                            # Создаем столбчатую диаграмму
                            fig = px.bar(
                                borough_stats,
                                x='Округ',
                                y='Медианная цена',
                                title='Медианная цена за кв.фут по округам',
                                labels={'Медианная цена': 'Цена за кв.фут ($)'},
                                color='Медианная цена',
                                color_continuous_scale='Viridis',
                                text='Медианная цена'
                            )
                            # Форматируем текст на столбцах
                            fig.update_traces(
                                texttemplate='$%{text:.0f}',
                                textposition='outside'
                            )
                            fig.update_layout(
                                yaxis_tickformat='$,.0f',
                                xaxis_title="Округ",
                                yaxis_title="Цена за кв.фут ($)",
                                yaxis_range=[0, max(borough_stats['Медианная цена']) * 1.2]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Нет данных о цене за квадратный фут.")
# Страница 3: Прогнозные модели
elif page == "Прогнозные модели":
    st.title("Прогнозные модели на основе данных")
    
    if filtered_df.empty:
        st.warning("Нет данных для построения моделей.")
    else:
        model_type = st.selectbox(
            "Выберите модель:",
            ["Прогноз цены на основе характеристик", "Анализ сезонности", "Классификация по ценовым категориям"]
        )
        
        # Модель 1: Прогноз цены на основе характеристик
        if model_type == "Прогноз цены на основе характеристик":
            st.subheader("Прогноз цены на основе характеристик объекта")
            
            if len(filtered_df) < 100:
                st.error("Слишком мало данных для построения модели.")
            else:
                # Подготовка данных для модели
                st.write("**Подготовка данных...**")
                
                # Выбираем релевантные признаки
                features = []
                if 'GROSS SQUARE FEET' in filtered_df.columns:
                    features.append('GROSS SQUARE FEET')
                if 'BOROUGH' in filtered_df.columns:
                    features.append('BOROUGH')
                if 'YEAR BUILT' in filtered_df.columns:
                    features.append('YEAR BUILT')
                if 'TOTAL UNITS' in filtered_df.columns:
                    features.append('TOTAL UNITS')
                if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
                    features.append('BUILDING CLASS CATEGORY')
                if 'LAND SQUARE FEET' in filtered_df.columns:
                    features.append('LAND SQUARE FEET')
                
                if len(features) < 3:
                    st.error("Недостаточно признаков для построения модели.")
                else:
                    # Создаем копию данных для модели
                    model_df = filtered_df.copy()
                    
                    # Удаляем пропуски
                    for feature in features + ['SALE PRICE']:
                        if feature in model_df.columns:
                            model_df = model_df.dropna(subset=[feature])
                    
                    if len(model_df) < 50:
                        st.error("Недостаточно данных после очистки пропусков.")
                    else:
                        # Преобразуем категориальные переменные
                        X = model_df[features].copy()
                        y = model_df['SALE PRICE']
                        
                        # Кодируем категориальные переменные
                        categorical_cols = X.select_dtypes(include=['object']).columns
                        if len(categorical_cols) > 0:
                            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                        
                        # Разделяем данные
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Обучаем модель
                        st.write("**Обучение модели Random Forest...**")
                        try:
                            model = RandomForestRegressor(
                                n_estimators=50,
                                max_depth=10,
                                random_state=42,
                                n_jobs=-1
                            )
                            
                            model.fit(X_train, y_train)
                            
                            # Прогноз и оценка
                            y_pred = model.predict(X_test)
                            
                            # Метрики
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("MAE", f"${mae:,.0f}")
                            with col2:
                                st.metric("RMSE", f"${rmse:,.0f}")
                            with col3:
                                st.metric("R²", f"{r2:.3f}")
                            
                            # Визуализация предсказаний
                            fig = go.Figure()
                            
                            n_points = min(100, len(y_test))
                            fig.add_trace(go.Scatter(
                                x=y_test.values[:n_points],
                                y=y_pred[:n_points],
                                mode='markers',
                                name='Предсказания',
                                marker=dict(size=8, opacity=0.6)
                            ))
                            
                            max_val = max(y_test.max(), y_pred.max())
                            min_val = min(y_test.min(), y_pred.min())
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Идеальное предсказание',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                title='Сравнение реальных и предсказанных цен',
                                xaxis_title='Реальная цена ($)',
                                yaxis_title='Предсказанная цена ($)',
                                xaxis_tickformat=',',
                                yaxis_tickformat=','
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Важность признаков
                            st.subheader("Важность признаков")
                            
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = pd.DataFrame({
                                    'Признак': X.columns,
                                    'Важность': model.feature_importances_
                                }).sort_values('Важность', ascending=False).head(15)
                                
                                fig = px.bar(
                                    feature_importance,
                                    x='Важность',
                                    y='Признак',
                                    orientation='h',
                                    title='Топ-15 важнейших признаков',
                                    color='Важность'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Ошибка при обучении модели: {str(e)}")
        
        # Модель 2: Анализ сезонности
        elif model_type == "Анализ сезонности":
            st.subheader("Анализ сезонных паттернов")
            
            if 'SALE_MONTH' not in filtered_df.columns or 'SALE PRICE' not in filtered_df.columns:
                st.error("В данных отсутствует необходимая информация.")
            else:
                # Анализ сезонности по месяцам
                monthly_analysis = filtered_df.groupby('SALE_MONTH').agg({
                    'SALE PRICE': ['median', 'count'],
                    'GROSS SQUARE FEET': 'median'
                }).reset_index()
                
                monthly_analysis.columns = ['Месяц', 'Медианная цена', 'Количество продаж', 'Медианная площадь']
                
                # Нормализуем данные
                monthly_analysis['Норм. цена'] = monthly_analysis['Медианная цена'] / monthly_analysis['Медианная цена'].mean()
                monthly_analysis['Норм. количество'] = monthly_analysis['Количество продаж'] / monthly_analysis['Количество продаж'].mean()
                
                # Визуализация сезонности
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Сезонность цен', 'Сезонность количества продаж'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_analysis['Месяц'],
                        y=monthly_analysis['Медианная цена'],
                        name='Медианная цена',
                        marker_color='royalblue'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_analysis['Месяц'],
                        y=monthly_analysis['Норм. цена'],
                        name='Норм. цена',
                        line=dict(color='red', width=3),
                        yaxis='y2'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_analysis['Месяц'],
                        y=monthly_analysis['Количество продаж'],
                        name='Количество продаж',
                        marker_color='lightgreen'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=monthly_analysis['Месяц'],
                        y=monthly_analysis['Норм. количество'],
                        name='Норм. количество',
                        line=dict(color='orange', width=3),
                        yaxis='y2'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    title_text="Анализ сезонности продаж"
                )
                
                fig.update_xaxes(title_text="Месяц", row=1, col=1)
                fig.update_xaxes(title_text="Месяц", row=2, col=1)
                fig.update_yaxes(title_text="Цена ($)", tickformat=',', row=1, col=1)
                fig.update_yaxes(title_text="Нормализованное значение", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Количество продаж", row=2, col=1)
                fig.update_yaxes(title_text="Нормализованное значение", row=2, col=1, secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Модель 3: Классификация по ценовым категориям
        elif model_type == "Классификация по ценовым категориям":
            st.subheader("Классификация объектов по ценовым категориям")
            
            if 'SALE PRICE' not in filtered_df.columns:
                st.error("В данных отсутствует информация о цене продажи.")
            elif len(filtered_df) < 100:
                st.error("Слишком мало данных для построения модели.")
            else:
                try:
                    # Создаем целевые категории
                    classification_df = filtered_df.copy()
                    
                    # Определяем границы категорий
                    price_33 = classification_df['SALE PRICE'].quantile(0.33)
                    price_66 = classification_df['SALE PRICE'].quantile(0.66)
                    
                    classification_df['PRICE_CATEGORY'] = pd.cut(
                        classification_df['SALE PRICE'],
                        bins=[0, price_33, price_66, classification_df['SALE PRICE'].max() + 1],
                        labels=['Дешевый', 'Средний', 'Дорогой']
                    )
                    
                    # Удаляем строки с NaN в категориях
                    classification_df = classification_df.dropna(subset=['PRICE_CATEGORY'])
                    
                    # Преобразуем в числовой формат
                    le = LabelEncoder()
                    classification_df['PRICE_CATEGORY_ENCODED'] = le.fit_transform(classification_df['PRICE_CATEGORY'])
                    
                    # Анализ распределения категорий
                    category_counts = classification_df['PRICE_CATEGORY'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=category_counts.values,
                            names=category_counts.index,
                            title='Распределение по ценовым категориям',
                            hole=0.4
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Характеристики по категориям
                        category_stats = classification_df.groupby('PRICE_CATEGORY').agg({
                            'SALE PRICE': ['median', 'min', 'max'],
                            'GROSS SQUARE FEET': 'median',
                            'YEAR BUILT': 'median'
                        }).round(2)
                        
                        category_stats.columns = ['Медианная цена', 'Минимальная цена', 'Максимальная цена',
                                                'Медианная площадь', 'Медианный год постройки']
                        
                        category_stats['Цена за кв.фут'] = category_stats['Медианная цена'] / category_stats['Медианная площадь']
                        
                        st.write("**Характеристики по категориям:**")
                        st.dataframe(
                            category_stats.style.format({
                                'Медианная цена': '${:,.0f}',
                                'Минимальная цена': '${:,.0f}',
                                'Максимальная цена': '${:,.0f}',
                                'Медианная площадь': '{:,.0f}',
                                'Медианный год постройки': '{:.0f}',
                                'Цена за кв.фут': '${:.2f}'
                            }),
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"Ошибка при подготовке данных: {str(e)}")
