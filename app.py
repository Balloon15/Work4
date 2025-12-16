import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
    'EASE-MENT': 'Сервитут (ограниченное пользование чужой собственностью)',
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
    
    # Производные поля
    'SALE_MONTH': 'Месяц продажи',
    'PRICE_CATEGORY': 'Ценовая категория'
}

# Функция для перевода названий колонок
def translate_columns(df):
    translated_cols = []
    for col in df.columns:
        translated_cols.append(COLUMN_TRANSLATIONS.get(col, col))
    df.columns = translated_cols
    return df

# Функция для обратного перевода (для фильтров)
def reverse_translate_column(russian_name):
    for eng, rus in COLUMN_TRANSLATIONS.items():
        if rus == russian_name:
            return eng
    return russian_name

# Загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv("nyc-rolling-sales.csv")
    
    numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                       'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                       'TOTAL UNITS', 'ZIP CODE']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].replace(' -  ', np.nan).replace(' - ', np.nan).replace(' -', np.nan), errors='coerce')
    
    if 'SALE DATE' in data.columns:
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
    
    if 'SALE PRICE' in data.columns:
        data = data[data['SALE PRICE'] > 0]
    
    if 'YEAR BUILT' in data.columns:
        data = data[data['YEAR BUILT'] > 0]
    
    return data

# Загружаем данные
df = load_data()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация исходных данных", "Анализ рынка недвижимости", "Таблица переводов"]
)

# Добавляем фильтры в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("Фильтры данных")

# Создаем копию с русскими названиями для фильтров
df_russian = translate_columns(df.copy())

# Фильтр по району
neighborhoods = ['Все'] + sorted(df['NEIGHBORHOOD'].dropna().unique().tolist())
selected_neighborhood = st.sidebar.selectbox(
    COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'), 
    neighborhoods
)

# Фильтр по типу здания
building_classes = ['Все'] + sorted(df['BUILDING CLASS CATEGORY'].dropna().unique().tolist())
selected_building_class = st.sidebar.selectbox(
    COLUMN_TRANSLATIONS.get('BUILDING CLASS CATEGORY', 'Категория класса здания'), 
    building_classes
)

# Фильтр по году постройки
if 'YEAR BUILT' in df.columns:
    valid_years = df[df['YEAR BUILT'] > 0]['YEAR BUILT']
    
    if not valid_years.empty:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        min_year = max(min_year, 1700)
        
        year_range = st.sidebar.slider(
            COLUMN_TRANSLATIONS.get('YEAR BUILT', 'Год постройки'),
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (1800, 2023)

# Фильтр по цене
if 'SALE PRICE' in df.columns:
    min_price = float(df['SALE PRICE'].min())
    max_price = float(df['SALE PRICE'].max())
    price_range = st.sidebar.slider(
        COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи') + " ($)",
        min_value=float(min_price),
        max_value=float(max_price),
        value=(float(min_price), float(max_price))
    )

# Применяем фильтры
filtered_df = df.copy()

if selected_neighborhood != 'Все':
    filtered_df = filtered_df[filtered_df['NEIGHBORHOOD'] == selected_neighborhood]

if selected_building_class != 'Все':
    filtered_df = filtered_df[filtered_df['BUILDING CLASS CATEGORY'] == selected_building_class]

if 'YEAR BUILT' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['YEAR BUILT'] >= year_range[0]) & 
        (filtered_df['YEAR BUILT'] <= year_range[1])
    ]

if 'SALE PRICE' in df.columns:
    filtered_df = filtered_df[
        (filtered_df['SALE PRICE'] >= price_range[0]) & 
        (filtered_df['SALE PRICE'] <= price_range[1])
    ]

# Создаем DataFrame с русскими названиями для отображения
filtered_df_russian = translate_columns(filtered_df.copy())

if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
    # Цена за квадратный фут
    filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET']
    filtered_df_russian['Цена за кв.фут'] = filtered_df['PRICE_PER_SQFT']
    
if all(col in filtered_df.columns for col in ['SALE PRICE', 'TOTAL UNITS']):
    # Цена за единицу (для многоквартирных домов)
    filtered_df['PRICE_PER_UNIT'] = filtered_df['SALE PRICE'] / filtered_df['TOTAL UNITS'].replace(0, np.nan)
    filtered_df_russian['Цена за единицу'] = filtered_df['PRICE_PER_UNIT']
    
if 'YEAR BUILT' in filtered_df.columns:
    # Возраст здания
    filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']
    filtered_df_russian['Возраст здания'] = filtered_df['BUILDING_AGE']

# Страница 3: Таблица переводов
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

# Страница 1: Визуализация исходных данных
elif page == "Визуализация исходных данных":
    st.title("Визуализация исходных данных")
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего записей", len(filtered_df))
    
    with col2:
        if 'SALE PRICE' in filtered_df.columns:
            avg_price = filtered_df['SALE PRICE'].mean()
            st.metric("Средняя цена ($)", f"{avg_price:,.0f}")
    
    with col3:
        if 'YEAR BUILT' in filtered_df.columns:
            valid_years_filtered = filtered_df[filtered_df['YEAR BUILT'] > 0]['YEAR BUILT']
            if not valid_years_filtered.empty:
                avg_year = valid_years_filtered.mean()
                st.metric("Средний год постройки", f"{avg_year:.0f}")
            else:
                st.metric("Средний год постройки", "Нет данных")
    
    with col4:
        unique_neighborhoods = filtered_df['NEIGHBORHOOD'].nunique()
        st.metric("Количество районов", unique_neighborhoods)
    
    st.markdown("---")
    
    # Таблица с данными
    st.subheader("Просмотр данных")
    
    all_columns_russian = filtered_df_russian.columns.tolist()
    selected_columns_russian = st.multiselect(
        "Выберите колонки для отображения:",
        all_columns_russian,
        default=all_columns_russian[:10] if len(all_columns_russian) > 10 else all_columns_russian
    )
    
    selected_columns_english = []
    for rus_col in selected_columns_russian:
        eng_col = reverse_translate_column(rus_col)
        selected_columns_english.append(eng_col if eng_col in filtered_df.columns else rus_col)
    
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
    
    if selected_columns_english:
        export_df = filtered_df[selected_columns_english]
    else:
        export_df = filtered_df
    
    csv = export_df.to_csv(index=False).encode('utf-8')
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
        numeric_cols_english = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_english:
            numeric_cols_russian = [COLUMN_TRANSLATIONS.get(col, col) for col in numeric_cols_english]
            
            stats_df = filtered_df[numeric_cols_english].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df.columns = ['Кол-во', 'Среднее', 'Стд. откл.', 'Мин.', '25%', 'Медиана', '75%', 'Макс.']
            stats_df.index = numeric_cols_russian
            
            st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Визуализации
    st.subheader("Визуализации данных")
    
    viz_type = st.selectbox(
        "Выберите тип визуализации:",
        ["Распределение цен", "Распределение по районам", "Распределение по году постройки", 
         "Корреляционная матрица", "Scatter plot: Цена vs Площадь"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if viz_type == "Распределение цен" and 'SALE PRICE' in filtered_df.columns:
            fig = px.histogram(
                filtered_df_russian, 
                x=COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи'),
                nbins=50,
                title="Распределение цен на недвижимость",
                labels={COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи'): 'Цена продажи ($)'}
            )
            fig.update_layout(xaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Распределение по районам":
            top_neighborhoods = filtered_df['NEIGHBORHOOD'].value_counts().head(15)
            fig = px.bar(
                x=top_neighborhoods.index,
                y=top_neighborhoods.values,
                title="Топ 15 районов по количеству продаж",
                labels={'x': COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'), 
                       'y': 'Количество продаж'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Распределение по году постройки" and 'YEAR BUILT' in filtered_df.columns:
            valid_year_data = filtered_df_russian[filtered_df_russian[COLUMN_TRANSLATIONS.get('YEAR BUILT', 'Год постройки')] > 0]
            
            if not valid_year_data.empty:
                fig = px.histogram(
                    valid_year_data,
                    x=COLUMN_TRANSLATIONS.get('YEAR BUILT', 'Год постройки'),
                    nbins=30,
                    title="Распределение по году постройки",
                    labels={COLUMN_TRANSLATIONS.get('YEAR BUILT', 'Год постройки'): 'Год постройки'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Нет корректных данных о годе постройки для визуализации")
    
    with col2:
        if viz_type == "Корреляционная матрица":
            numeric_cols_english = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols_english) > 1:
                corr_matrix = filtered_df[numeric_cols_english].corr()
                
                numeric_cols_russian = [COLUMN_TRANSLATIONS.get(col, col) for col in numeric_cols_english]
                corr_matrix.index = numeric_cols_russian
                corr_matrix.columns = numeric_cols_russian
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    title="Корреляционная матрица",
                    color_continuous_scale='RdBu',
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif viz_type == "Scatter plot: Цена vs Площадь":
            if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
                fig = px.scatter(
                    filtered_df_russian,
                    x=COLUMN_TRANSLATIONS.get('GROSS SQUARE FEET', 'Общая площадь (кв. фут)'),
                    y=COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи'),
                    color=COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'),
                    title="Цена vs Общая площадь",
                    labels={
                        COLUMN_TRANSLATIONS.get('GROSS SQUARE FEET', 'Общая площадь (кв. фут)'): 'Общая площадь (кв. фут)',
                        COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи'): 'Цена продажи ($)'
                    },
                    opacity=0.6
                )
                fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart для категорий
    st.markdown("---")
    st.subheader("Категориальный анализ")
    
    cat_col_options = {
        'BOROUGH': COLUMN_TRANSLATIONS.get('BOROUGH', 'Боро'),
        'TAX CLASS AT PRESENT': COLUMN_TRANSLATIONS.get('TAX CLASS AT PRESENT', 'Налоговый класс (текущий)'),
        'BUILDING CLASS CATEGORY': COLUMN_TRANSLATIONS.get('BUILDING CLASS CATEGORY', 'Категория класса здания')
    }
    
    cat_col_english = st.selectbox(
        "Выберите категориальную переменную:",
        list(cat_col_options.keys()),
        format_func=lambda x: cat_col_options[x]
    )
    
    if cat_col_english in filtered_df.columns:
        cat_col_russian = COLUMN_TRANSLATIONS.get(cat_col_english, cat_col_english)
        fig = px.pie(
            filtered_df_russian,
            names=cat_col_russian,
            title=f"Распределение по {cat_col_russian.lower()}",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

# Страница 2: Анализ рынка недвижимости
else:
    st.title("📊 Комплексный анализ рынка недвижимости")
    
    # Информация о выбранных данных
    st.info(f"Анализ на основе {len(filtered_df)} записей")
    
    # Разделы анализа
    analysis_section = st.selectbox(
        "Выберите раздел анализа:",
        ["📈 Обзор рынка", "🏢 Анализ по типам зданий", "🗺️ Географический анализ", 
         "💰 Анализ доходности", "⚖️ Анализ спроса и предложения", "🔮 Прогнозный анализ"]
    )
    
    # Секция 1: Обзор рынка
    if analysis_section == "📈 Обзор рынка":
        st.subheader("Ключевые показатели рынка")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'SALE PRICE' in filtered_df.columns:
                median_price = filtered_df['SALE PRICE'].median()
                st.metric("Медианная цена ($)", f"{median_price:,.0f}")
        
        with col2:
            if 'GROSS SQUARE FEET' in filtered_df.columns:
                avg_sqft = filtered_df['GROSS SQUARE FEET'].mean()
                st.metric("Средняя площадь (кв.фут)", f"{avg_sqft:,.0f}")
        
        with col3:
            if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
                valid_data = filtered_df[(filtered_df['SALE PRICE'] > 0) & (filtered_df['GROSS SQUARE FEET'] > 0)]
                if not valid_data.empty:
                    price_per_sqft = (valid_data['SALE PRICE'] / valid_data['GROSS SQUARE FEET']).mean()
                    st.metric("Средняя цена за кв.фут ($)", f"{price_per_sqft:.2f}")
                else:
                    st.metric("Средняя цена за кв.фут ($)", "Нет данных")
        
        with col4:
            if 'YEAR BUILT' in filtered_df.columns:
                valid_years_filtered = filtered_df[filtered_df['YEAR BUILT'] > 0]['YEAR BUILT']
                if not valid_years_filtered.empty:
                    oldest_building = valid_years_filtered.min()
                    st.metric("Самое старое здание (год)", f"{oldest_building:.0f}")
                else:
                    st.metric("Самое старое здание (год)", "Нет данных")
        
        st.markdown("---")
        
        # Анализ трендов
        st.subheader("Анализ трендов")
        
        if 'SALE DATE' in filtered_df.columns:
            filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
            filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.to_period('M').astype(str)
            
            # Годовые тренды
            yearly_trend = filtered_df.groupby('SALE_YEAR').agg({
                'SALE PRICE': ['count', 'mean', 'median']
            }).reset_index()
            
            yearly_trend.columns = ['Год', 'Количество продаж', 'Средняя цена', 'Медианная цена']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    yearly_trend,
                    x='Год',
                    y='Количество продаж',
                    title="Количество продаж по годам",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    yearly_trend,
                    x='Год',
                    y='Медианная цена',
                    title="Медианная цена по годам",
                    markers=True
                )
                fig.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
            
            # Расчет годового роста
            if len(yearly_trend) > 1:
                yearly_trend['Рост цен (%)'] = yearly_trend['Медианная цена'].pct_change() * 100
                avg_growth = yearly_trend['Рост цен (%)'].mean()
                st.metric("Среднегодовой рост цен", f"{avg_growth:.1f}%")
    
    # Секция 2: Анализ по типам зданий
    elif analysis_section == "🏢 Анализ по типам зданий":
        st.subheader("Анализ по типам недвижимости")
        
        if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
            # Группировка по типам зданий
            building_analysis = filtered_df.groupby('BUILDING CLASS CATEGORY').agg({
                'SALE PRICE': ['count', 'mean', 'median', 'std'],
                'GROSS SQUARE FEET': 'mean',
                'TOTAL UNITS': 'mean'
            }).round(2)
            
            building_analysis.columns = ['Количество', 'Средняя цена', 'Медианная цена', 
                                         'Стд. отклонение', 'Средняя площадь', 'Среднее кол-во единиц']
            
            # Топ-10 самых дорогих типов
            top_buildings = building_analysis.sort_values('Средняя цена', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    top_buildings.reset_index(),
                    x='BUILDING CLASS CATEGORY',
                    y='Средняя цена',
                    title='Топ-10 самых дорогих типов недвижимости',
                    color='Средняя цена',
                    color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Распределение типов по количеству продаж
                building_counts = building_analysis.sort_values('Количество', ascending=False).head(10)
                
                fig = px.pie(
                    building_counts.reset_index(),
                    values='Количество',
                    names='BUILDING CLASS CATEGORY',
                    title='Распределение продаж по типам недвижимости',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Детальная таблица
            st.subheader("Детальная статистика по типам зданий")
            
            # Добавляем цену за кв.фут
            building_analysis['Цена за кв.фут'] = building_analysis['Средняя цена'] / building_analysis['Средняя площадь']
            
            st.dataframe(
                building_analysis.sort_values('Средняя цена', ascending=False).style.format({
                    'Количество': '{:,.0f}',
                    'Средняя цена': '${:,.0f}',
                    'Медианная цена': '${:,.0f}',
                    'Стд. отклонение': '${:,.0f}',
                    'Средняя площадь': '{:,.0f}',
                    'Среднее кол-во единиц': '{:,.1f}',
                    'Цена за кв.фут': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Анализ рентабельности по типам
            st.subheader("Анализ рентабельности по типам")
            
            if 'RESIDENTIAL UNITS' in filtered_df.columns and 'COMMERCIAL UNITS' in filtered_df.columns:
                # Добавляем тип использования
                filtered_df['PROPERTY_TYPE'] = np.where(
                    filtered_df['COMMERCIAL UNITS'] > 0,
                    'Смешанная',
                    np.where(filtered_df['RESIDENTIAL UNITS'] > 0, 'Жилая', 'Другая')
                )
                
                type_analysis = filtered_df.groupby('PROPERTY_TYPE').agg({
                    'SALE PRICE': ['count', 'mean', 'median'],
                    'GROSS SQUARE FEET': 'mean'
                }).round(2)
                
                type_analysis.columns = ['Количество', 'Средняя цена', 'Медианная цена', 'Средняя площадь']
                type_analysis['Цена за кв.фут'] = type_analysis['Средняя цена'] / type_analysis['Средняя площадь']
                
                fig = px.bar(
                    type_analysis.reset_index(),
                    x='PROPERTY_TYPE',
                    y='Цена за кв.фут',
                    title='Стоимость квадратного фута по типам использования',
                    color='Цена за кв.фут',
                    text='Цена за кв.фут'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    # Секция 3: Географический анализ
    elif analysis_section == "🗺️ Географический анализ":
        st.subheader("Географический анализ рынка")
        
        if 'BOROUGH' in filtered_df.columns:
            # Создаем названия для Borough
            borough_names = {
                1: 'Manhattan',
                2: 'Brooklyn', 
                3: 'Queens',
                4: 'Bronx',
                5: 'Staten Island'
            }
            
            filtered_df['BOROUGH_NAME'] = filtered_df['BOROUGH'].map(borough_names)
            
            # Анализ по Borough
            borough_analysis = filtered_df.groupby('BOROUGH_NAME').agg({
                'SALE PRICE': ['count', 'mean', 'median', 'std'],
                'GROSS SQUARE FEET': 'mean',
                'YEAR BUILT': 'mean'
            }).round(2)
            
            borough_analysis.columns = ['Количество', 'Средняя цена', 'Медианная цена', 
                                        'Стд. отклонение', 'Средняя площадь', 'Средний год постройки']
            
            # Добавляем цену за кв.фут
            borough_analysis['Цена за кв.фут'] = borough_analysis['Средняя цена'] / borough_analysis['Средняя площадь']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Цены по Borough
                fig = px.bar(
                    borough_analysis.reset_index(),
                    x='BOROUGH_NAME',
                    y='Средняя цена',
                    title='Средняя цена по городским округам',
                    color='Средняя цена',
                    color_continuous_scale='thermal',
                    text='Средняя цена'
                )
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Количество продаж по Borough
                fig = px.pie(
                    borough_analysis.reset_index(),
                    values='Количество',
                    names='BOROUGH_NAME',
                    title='Доля продаж по городским округам',
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap районов
            st.subheader("Анализ по районам")
            
            if 'NEIGHBORHOOD' in filtered_df.columns:
                # Топ-20 районов по количеству продаж
                neighborhood_stats = filtered_df.groupby('NEIGHBORHOOD').agg({
                    'SALE PRICE': ['count', 'mean', 'median']
                }).round(2)
                
                neighborhood_stats.columns = ['Количество продаж', 'Средняя цена', 'Медианная цена']
                neighborhood_stats = neighborhood_stats.sort_values('Количество продаж', ascending=False).head(20)
                
                fig = px.scatter(
                    neighborhood_stats.reset_index(),
                    x='Средняя цена',
                    y='Количество продаж',
                    size='Количество продаж',
                    color='Средняя цена',
                    hover_name='NEIGHBORHOOD',
                    title='Соотношение цены и количества продаж по районам',
                    size_max=60
                )
                fig.update_layout(xaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
                
                # Топ дорогих и дешевых районов
                col1, col2 = st.columns(2)
                
                with col1:
                    expensive_neighborhoods = neighborhood_stats.nlargest(10, 'Средняя цена')
                    st.write("**Топ-10 самых дорогих районов:**")
                    for idx, (neighborhood, row) in enumerate(expensive_neighborhoods.iterrows(), 1):
                        st.write(f"{idx}. {neighborhood}: ${row['Средняя цена']:,.0f}")
                
                with col2:
                    affordable_neighborhoods = neighborhood_stats.nsmallest(10, 'Средняя цена')
                    st.write("**Топ-10 самых доступных районов:**")
                    for idx, (neighborhood, row) in enumerate(affordable_neighborhoods.iterrows(), 1):
                        st.write(f"{idx}. {neighborhood}: ${row['Средняя цена']:,.0f}")
    
    # Секция 4: Анализ доходности
    elif analysis_section == "💰 Анализ доходности":
        st.subheader("Анализ доходности недвижимости")
        
        # Создаем дополнительные метрики для анализа доходности
        if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET', 'TOTAL UNITS', 'YEAR BUILT']):
            # Цена за кв.фут
            filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET']
            
            # Цена за единицу (для многоквартирных домов)
            filtered_df['PRICE_PER_UNIT'] = filtered_df['SALE PRICE'] / filtered_df['TOTAL UNITS'].replace(0, np.nan)
            
            # Возраст здания
            filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_price_sqft = filtered_df['PRICE_PER_SQFT'].mean()
                st.metric("Средняя цена за кв.фут", f"${avg_price_sqft:.2f}")
            
            with col2:
                avg_price_unit = filtered_df['PRICE_PER_UNIT'].dropna().mean()
                st.metric("Средняя цена за единицу", f"${avg_price_unit:,.0f}")
            
            with col3:
                avg_age = filtered_df['BUILDING_AGE'].mean()
                st.metric("Средний возраст зданий", f"{avg_age:.0f} лет")
            
            with col4:
                if 'SALE DATE' in filtered_df.columns:
                    # Расчет годовой доходности (упрощенный)
                    filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
                    yearly_returns = filtered_df.groupby('SALE_YEAR')['PRICE_PER_SQFT'].mean().pct_change().mean() * 100
                    st.metric("Среднегодовая доходность", f"{yearly_returns:.1f}%")
            
            st.markdown("---")
            
            # Анализ зависимости цены от возраста
            st.subheader("Зависимость цены от возраста здания")
            
            # Группировка по возрастным категориям
            age_bins = [0, 10, 20, 30, 50, 100, 200]
            age_labels = ['0-10 лет', '11-20 лет', '21-30 лет', '31-50 лет', '51-100 лет', '100+ лет']
            
            filtered_df['AGE_CATEGORY'] = pd.cut(filtered_df['BUILDING_AGE'], bins=age_bins, labels=age_labels)
            
            age_analysis = filtered_df.groupby('AGE_CATEGORY').agg({
                'PRICE_PER_SQFT': ['mean', 'median', 'count'],
                'SALE PRICE': 'mean'
            }).round(2)
            
            age_analysis.columns = ['Средняя цена за кв.фут', 'Медианная цена за кв.фут', 'Количество', 'Средняя цена']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    age_analysis.reset_index(),
                    x='AGE_CATEGORY',
                    y='Средняя цена за кв.фут',
                    title='Стоимость квадратного фута по возрасту здания',
                    color='Средняя цена за кв.фут',
                    text='Средняя цена за кв.фут'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    filtered_df.sample(min(1000, len(filtered_df))),
                    x='BUILDING_AGE',
                    y='PRICE_PER_SQFT',
                    trendline="lowess",
                    title='Зависимость цены за кв.фут от возраста здания',
                    labels={'BUILDING_AGE': 'Возраст здания (лет)', 'PRICE_PER_SQFT': 'Цена за кв.фут ($)'},
                    opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Анализ ROI по типам зданий
            st.subheader("Доходность по типам недвижимости")
            
            if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
                roi_analysis = filtered_df.groupby('BUILDING CLASS CATEGORY').agg({
                    'PRICE_PER_SQFT': 'mean',
                    'PRICE_PER_UNIT': 'mean',
                    'SALE PRICE': ['count', 'median']
                }).round(2)
                
                roi_analysis.columns = ['Цена за кв.фут', 'Цена за единицу', 'Количество продаж', 'Медианная цена']
                roi_analysis = roi_analysis.sort_values('Цена за кв.фут', ascending=False).head(15)
                
                fig = px.scatter(
                    roi_analysis.reset_index(),
                    x='Цена за кв.фут',
                    y='Цена за единицу',
                    size='Количество продаж',
                    color='Медианная цена',
                    hover_name='BUILDING CLASS CATEGORY',
                    title='Доходность по типам недвижимости',
                    size_max=50
                )
                fig.update_layout(xaxis_tickformat='$,.2f', yaxis_tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
    
    # Секция 5: Анализ спроса и предложения
    elif analysis_section == "⚖️ Анализ спроса и предложения":
        st.subheader("Анализ спроса и предложения")
        
        # Анализ предложения по размерам
        if 'GROSS SQUARE FEET' in filtered_df.columns:
            # Создание категорий по площади
            size_bins = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, float('inf')]
            size_labels = ['<500 кв.фут', '500-1000', '1000-1500', '1500-2000', 
                          '2000-3000', '3000-5000', '5000-10000', '>10000']
            
            filtered_df['SIZE_CATEGORY'] = pd.cut(filtered_df['GROSS SQUARE FEET'], 
                                                  bins=size_bins, 
                                                  labels=size_labels)
            
            size_analysis = filtered_df.groupby('SIZE_CATEGORY').agg({
                'SALE PRICE': ['count', 'mean', 'median'],
                'PRICE_PER_SQFT': 'mean'
            }).round(2)
            
            size_analysis.columns = ['Количество', 'Средняя цена', 'Медианная цена', 'Цена за кв.фут']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    size_analysis.reset_index(),
                    x='SIZE_CATEGORY',
                    y='Количество',
                    title='Распределение предложения по размерам',
                    color='Средняя цена',
                    text='Количество'
                )
                fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(
                    size_analysis.reset_index(),
                    x='SIZE_CATEGORY',
                    y='Цена за кв.фут',
                    title='Стоимость квадратного фута по размерам',
                    markers=True
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Анализ ликвидности
            st.subheader("Анализ ликвидности рынка")
            
            if 'SALE DATE' in filtered_df.columns:
                # Ежемесячные объемы продаж
                monthly_volume = filtered_df.groupby('SALE_MONTH').size().reset_index(name='Количество продаж')
                monthly_volume.columns = ['Месяц', 'Количество продаж']
                
                # Расчет скорости продаж (упрощенный)
                avg_monthly_sales = monthly_volume['Количество продаж'].mean()
                total_inventory = len(filtered_df)
                months_supply = total_inventory / avg_monthly_sales if avg_monthly_sales > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Среднемесячные продажи", f"{avg_monthly_sales:.0f}")
                with col2:
                    st.metric("Текущее предложение", f"{total_inventory:,}")
                with col3:
                    st.metric("Месяцев предложения", f"{months_supply:.1f}")
                
                # График сезонности
                fig = px.line(
                    monthly_volume,
                    x='Месяц',
                    y='Количество продаж',
                    title='Сезонность продаж недвижимости',
                    markers=True
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Анализ ценовых диапазонов
            st.subheader("Анализ ценовых диапазонов")
            
            if 'SALE PRICE' in filtered_df.columns:
                # Создание ценовых категорий
                price_bins = [0, 500000, 1000000, 2000000, 5000000, 10000000, 50000000, float('inf')]
                price_labels = ['<$500K', '$500K-$1M', '$1M-$2M', '$2M-$5M', '$5M-$10M', '$10M-$50M', '>$50M']
                
                filtered_df['PRICE_RANGE'] = pd.cut(filtered_df['SALE PRICE'], bins=price_bins, labels=price_labels)
                
                price_range_analysis = filtered_df.groupby('PRICE_RANGE').agg({
                    'SALE PRICE': 'count',
                    'GROSS SQUARE FEET': 'mean',
                    'PRICE_PER_SQFT': 'mean'
                }).round(2)
                
                price_range_analysis.columns = ['Количество', 'Средняя площадь', 'Цена за кв.фут']
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Распределение по ценовым диапазонам', 'Стоимость кв.фута по диапазонам'),
                    shared_xaxes=True
                )
                
                fig.add_trace(
                    go.Bar(x=price_range_analysis.index, y=price_range_analysis['Количество'], 
                          name='Количество', marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=price_range_analysis.index, y=price_range_analysis['Цена за кв.фут'],
                              name='Цена за кв.фут', line=dict(color='red', width=3)),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
    
    # Секция 6: Прогнозный анализ
    elif analysis_section == "🔮 Прогнозный анализ":
        st.subheader("Прогнозный анализ рынка недвижимости")
        
        if 'SALE DATE' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
            # Подготовка данных для прогноза
            forecast_df = filtered_df.copy()
            forecast_df['TIME_INDEX'] = (forecast_df['SALE DATE'] - forecast_df['SALE DATE'].min()).dt.days
            
            # Группировка по месяцам для прогноза
            forecast_df['YEAR_MONTH'] = forecast_df['SALE DATE'].dt.to_period('M')
            monthly_data = forecast_df.groupby('YEAR_MONTH').agg({
                'SALE PRICE': 'median',
                'TIME_INDEX': 'first'
            }).reset_index()
            
            monthly_data['YEAR_MONTH'] = monthly_data['YEAR_MONTH'].astype(str)
            
            if len(monthly_data) > 3:
                # Линейная регрессия для прогноза
                X = monthly_data[['TIME_INDEX']].values
                y = monthly_data['SALE PRICE'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Прогноз на 6 месяцев вперед
                last_time = monthly_data['TIME_INDEX'].max()
                future_months = 6
                future_days = np.arange(last_time, last_time + 30 * future_months, 30)
                future_prices = model.predict(future_days.reshape(-1, 1))
                
                # Визуализация прогноза
                fig = go.Figure()
                
                # Исторические данные
                fig.add_trace(go.Scatter(
                    x=monthly_data['YEAR_MONTH'],
                    y=monthly_data['SALE PRICE'],
                    mode='lines+markers',
                    name='Исторические данные',
                    line=dict(color='blue', width=2)
                ))
                
                # Прогноз
                future_dates = pd.date_range(
                    start=forecast_df['SALE DATE'].max(),
                    periods=future_months + 1,
                    freq='M'
                )[1:]
                
                fig.add_trace(go.Scatter(
                    x=future_dates.strftime('%Y-%m'),
                    y=future_prices,
                    mode='lines+markers',
                    name='Прогноз',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Прогноз цен на недвижимость на 6 месяцев',
                    xaxis_title='Месяц',
                    yaxis_title='Медианная цена ($)',
                    yaxis_tickformat=',',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Расчет ожидаемого роста
                current_price = monthly_data['SALE PRICE'].iloc[-1]
                forecasted_price = future_prices[-1]
                expected_growth = ((forecasted_price - current_price) / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Текущая медианная цена", f"${current_price:,.0f}")
                with col2:
                    st.metric("Прогноз через 6 месяцев", f"${forecasted_price:,.0f}")
                with col3:
                    st.metric("Ожидаемый рост", f"{expected_growth:.1f}%", 
                             delta=f"{expected_growth:.1f}%")
                
                # Рекомендации на основе анализа
                st.markdown("---")
                st.subheader("💡 Рекомендации для инвесторов")
                
                recommendations = []
                
                # Генерация рекомендаций на основе анализа
                if expected_growth > 5:
                    recommendations.append("📈 **Рынок растет** - благоприятное время для инвестиций")
                elif expected_growth < -2:
                    recommendations.append("🛒 **Цены снижаются** - хорошие возможности для покупки")
                else:
                    recommendations.append("⚖️ **Стабильный рынок** - подходит для долгосрочных инвестиций")
                
                # Анализ по типам зданий для рекомендаций
                if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
                    building_growth = filtered_df.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().nlargest(3)
                    if len(building_growth) > 0:
                        top_type = building_growth.index[0]
                        recommendations.append(f"🏢 **Рекомендуемый тип**: {top_type} - показывает лучшую доходность")
                
                # Анализ по районам для рекомендаций
                if 'NEIGHBORHOOD' in filtered_df.columns:
                    neighborhood_growth = filtered_df.groupby('NEIGHBORHOOD')['SALE PRICE'].mean().nlargest(3)
                    if len(neighborhood_growth) > 0:
                        top_area = neighborhood_growth.index[0]
                        recommendations.append(f"📍 **Перспективный район**: {top_area} - высокий потенциал роста")
                
                # Вывод рекомендаций
                st.write("**Ключевые рекомендации:**")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Риски и ограничения
                st.markdown("---")
                st.subheader("⚠️ Риски и ограничения")
                
                risks = [
                    "Прогноз основан на исторических данных и может не учитывать будущие экономические изменения",
                    "Рынок недвижимости подвержен сезонным колебаниям",
                    "Рекомендации носят информационный характер и не являются финансовым советом",
                    "Необходимо учитывать индивидуальные финансовые возможности и цели"
                ]
                
                for i, risk in enumerate(risks, 1):
                    st.write(f"{i}. {risk}")
            else:
                st.warning("Недостаточно данных для построения прогноза. Требуется более 3 месяцев данных.")

# Информация в футере
st.sidebar.markdown("---")

# Добавляем возможность сброса фильтров
if st.sidebar.button("Сбросить все фильтры"):
    st.rerun()
