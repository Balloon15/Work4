import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import io

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

# Функция для обратного перевода
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
    
    # Очистка данных
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
    ["Визуализация исходных данных", "Результаты анализа", "Анализ выбросов", "Таблица переводов"]
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
        
        # Показываем информацию о корректных данных
        total_records = len(df)
        valid_year_records = len(df[df['YEAR BUILT'] > 0])
        st.sidebar.caption(f"Корректных данных о годе постройки: {valid_year_records}/{total_records}")
    else:
        year_range = (1800, 2023)
        st.sidebar.warning("Нет корректных данных о годе постройки")

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

# Страница 1: Визуализация исходных данных
if page == "Визуализация исходных данных":
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
    
    # Выбор колонок для отображения (используем русские названия)
    all_columns_russian = filtered_df_russian.columns.tolist()
    selected_columns_russian = st.multiselect(
        "Выберите колонки для отображения:",
        all_columns_russian,
        default=all_columns_russian[:10] if len(all_columns_russian) > 10 else all_columns_russian
    )
    
    # Преобразуем выбранные русские названия обратно в английские для фильтрации
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
        # Отображаем таблицу с русскими названиями колонок
        display_df = filtered_df_russian[selected_columns_russian].iloc[start_idx:end_idx]
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    # Экспорт данных (используем оригинальные английские названия)
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
            # Преобразуем английские названия в русские для отображения
            numeric_cols_russian = [COLUMN_TRANSLATIONS.get(col, col) for col in numeric_cols_english]
            
            stats_df = filtered_df[numeric_cols_english].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df.columns = ['Кол-во', 'Среднее', 'Стд. откл.', 'Мин.', '25%', 'Медиана', '75%', 'Макс.']
            stats_df.index = numeric_cols_russian
            
            st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
    
    st.markdown("---")
    
    # Визуализации
    st.subheader("Визуализации данных")
    
    # Выбор типа графика
    viz_type = st.selectbox(
        "Выберите тип визуализации:",
        ["Распределение цен", "Распределение по районам", "Распределение по году постройки", 
         "Корреляционная матрица", "Scatter plot: Цена vs Площадь", 
         "Распределение районов по округам"]
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
            # Фильтруем только корректные годы для визуализации
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
        
        elif viz_type == "Распределение районов по округам":
            if 'BOROUGH' in filtered_df.columns and 'NEIGHBORHOOD' in filtered_df.columns:
                # Подсчет уникальных районов по Borough
                borough_neighborhood_count = filtered_df.groupby('BOROUGH')['NEIGHBORHOOD'].nunique().reset_index()
                borough_neighborhood_count.columns = ['BOROUGH', 'Количество районов']
                
                # Переводим номера Borough в названия
                borough_neighborhood_count['Городской округ'] = borough_neighborhood_count['BOROUGH'].map({
                    1: 'Manhattan',
                    2: 'Brooklyn', 
                    3: 'Queens',
                    4: 'Bronx',
                    5: 'Staten Island'
                })
                
                fig = px.bar(
                    borough_neighborhood_count,
                    x='Городской округ',
                    y='Количество районов',
                    title='Количество уникальных районов по городским округам',
                    text='Количество районов',
                    color='Количество районов',
                    color_continuous_scale='Plasma'
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Дополнительная статистика
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Всего уникальных районов", 
                        filtered_df['NEIGHBORHOOD'].nunique()
                    )
                
                with col2:
                    st.metric(
                        "Среднее районов на округ", 
                        f"{filtered_df.groupby('BOROUGH')['NEIGHBORHOOD'].nunique().mean():.1f}"
                    )
    
    with col2:
        if viz_type == "Корреляционная матрица":
            numeric_cols_english = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols_english) > 1:
                corr_matrix = filtered_df[numeric_cols_english].corr()
                
                # Переводим названия колонок для отображения
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

# Страница 2: Результаты анализа
elif page == "Результаты анализа":
    st.title("Результаты анализа")
    
    # Информация о выбранных данных
    st.info(f"Анализ на основе {len(filtered_df)} записей")
    
    # KPI для анализа
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
            # Избегаем деления на ноль
            valid_data = filtered_df[(filtered_df['SALE PRICE'] > 0) & (filtered_df['GROSS SQUARE FEET'] > 0)]
            if not valid_data.empty:
                price_per_sqft = (valid_data['SALE PRICE'] / valid_data['GROSS SQUARE FEET']).mean()
                st.metric("Средняя цена за кв.фут ($)", f"{price_per_sqft:.2f}")
            else:
                st.metric("Средняя цена за кв.фут ($)", "Нет данных")
    
    with col4:
        if 'YEAR BUILT' in filtered_df.columns:
            # Используем только корректные годы
            valid_years_filtered = filtered_df[filtered_df['YEAR BUILT'] > 0]['YEAR BUILT']
            if not valid_years_filtered.empty:
                oldest_building = valid_years_filtered.min()
                st.metric("Самое старое здание (год)", f"{oldest_building:.0f}")
            else:
                st.metric("Самое старое здание (год)", "Нет данных")
    
    st.markdown("---")
    
    # Анализ трендов
    st.subheader("Анализ трендов")
    
    # Анализ по месяцам (если есть дата)
    if 'SALE DATE' in filtered_df.columns:
        filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.to_period('M').astype(str)
        filtered_df_russian['Месяц продажи'] = filtered_df['SALE_MONTH']
        
        monthly_sales = filtered_df.groupby('SALE_MONTH').agg({
            'SALE PRICE': ['count', 'mean', 'median']
        }).reset_index()
        
        monthly_sales.columns = ['Month', 'Number of Sales', 'Average Price', 'Median Price']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                monthly_sales,
                x='Month',
                y='Number of Sales',
                title="Количество продаж по месяцам",
                markers=True
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                monthly_sales,
                x='Month',
                y='Median Price',
                title="Медианная цена по месяцам",
                markers=True
            )
            fig.update_layout(yaxis_tickformat=',')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Сравнение районов
    st.subheader("Сравнение районов")
    
    if 'NEIGHBORHOOD' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
        neighborhood_stats = filtered_df.groupby('NEIGHBORHOOD').agg({
            'SALE PRICE': ['count', 'mean', 'median', 'std']
        }).round(2).reset_index()
        
        neighborhood_stats.columns = [
            COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'), 
            'Количество продаж', 
            'Средняя цена', 
            'Медианная цена', 
            'Стд. отклонение'
        ]
        
        # Сортировка
        sort_by = st.selectbox(
            "Сортировать районы по:",
            ['Количество продаж', 'Средняя цена', 'Медианная цена']
        )
        
        top_n = st.slider("Показать топ N районов:", 5, 20, 10)
        
        neighborhood_stats_sorted = neighborhood_stats.sort_values(sort_by, ascending=False).head(top_n)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                neighborhood_stats_sorted,
                x=COLUMN_TRANSLATIONS.get('NEIGHBORHOOD', 'Район'),
                y=sort_by,
                title=f"Топ {top_n} районов по {sort_by.lower()}",
                color=sort_by,
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=45)
            if 'цена' in sort_by.lower():
                fig.update_layout(yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                neighborhood_stats_sorted.style.format({
                    'Средняя цена': '{:,.0f}',
                    'Медианная цена': '{:,.0f}',
                    'Стд. отклонение': '{:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
    
    st.markdown("---")
    
    # Анализ распределения районов по округам
    st.subheader("Распределение районов по городским округам")
    
    if 'BOROUGH' in filtered_df.columns and 'NEIGHBORHOOD' in filtered_df.columns:
        # Создаем словарь для перевода номеров Borough в названия
        borough_names = {
            1: 'Manhattan',
            2: 'Brooklyn', 
            3: 'Queens',
            4: 'Bronx',
            5: 'Staten Island'
        }
        
        # Группируем данные по Borough и считаем уникальные районы
        borough_neighborhood_stats = filtered_df.groupby('BOROUGH').agg({
            'NEIGHBORHOOD': ['nunique', 'count']
        }).reset_index()
        
        borough_neighborhood_stats.columns = ['Borough ID', 'Количество районов', 'Всего записей']
        
        # Добавляем названия Borough
        borough_neighborhood_stats['Городской округ'] = borough_neighborhood_stats['Borough ID'].map(borough_names)
        
        # Сортируем по количеству районов
        borough_neighborhood_stats = borough_neighborhood_stats.sort_values('Количество районов', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Столбчатая диаграмма
            fig = px.bar(
                borough_neighborhood_stats,
                x='Городской округ',
                y='Количество районов',
                title='Количество районов по городским округам',
                color='Количество районов',
                color_continuous_scale='Viridis',
                text='Количество районов'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Круговая диаграмма
            fig = px.pie(
                borough_neighborhood_stats,
                values='Количество районов',
                names='Городской округ',
                title='Доля районов по округам',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Таблица со статистикой
        st.dataframe(
            borough_neighborhood_stats[['Городской округ', 'Количество районов', 'Всего записей']].style.format({
                'Количество районов': '{:,.0f}',
                'Всего записей': '{:,.0f}'
            }),
            use_container_width=True,
            height=200
        )
        
        # Инсайты
        st.markdown("**Инсайты:**")
        
        # Находим округ с наибольшим количеством районов
        max_neighborhoods = borough_neighborhood_stats.loc[borough_neighborhood_stats['Количество районов'].idxmax()]
        min_neighborhoods = borough_neighborhood_stats.loc[borough_neighborhood_stats['Количество районов'].idxmin()]
        
        st.write(f"• **{max_neighborhoods['Городской округ']}** имеет наибольшее количество районов ({max_neighborhoods['Количество районов']})")
        st.write(f"• **{min_neighborhoods['Городской округ']}** имеет наименьшее количество районов ({min_neighborhoods['Количество районов']})")
        
        # Вычисляем среднее количество записей на район
        avg_records_per_neighborhood = borough_neighborhood_stats['Всего записей'].sum() / borough_neighborhood_stats['Количество районов'].sum()
        st.write(f"• В среднем на один район приходится {avg_records_per_neighborhood:.1f} записей о продажах")
    
    st.markdown("---")
    
    # Интерактивная карта районов по округам
    st.subheader("Исследование районов по городским округам")
    
    if 'BOROUGH' in filtered_df.columns and 'NEIGHBORHOOD' in filtered_df.columns:
        # Выбор Borough для детального анализа
        selected_borough_name = st.selectbox(
            "Выберите городской округ для детального анализа:",
            list(borough_names.values())
        )
        
        # Получаем ID выбранного Borough
        selected_borough_id = [k for k, v in borough_names.items() if v == selected_borough_name][0]
        
        # Фильтруем данные по выбранному Borough
        borough_data = filtered_df[filtered_df['BOROUGH'] == selected_borough_id]
        
        if not borough_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                neighborhoods_count = borough_data['NEIGHBORHOOD'].nunique()
                st.metric("Количество районов", neighborhoods_count)
            
            with col2:
                total_sales = len(borough_data)
                st.metric("Всего продаж", f"{total_sales:,.0f}")
            
            with col3:
                if 'SALE PRICE' in borough_data.columns:
                    avg_price = borough_data['SALE PRICE'].mean()
                    st.metric("Средняя цена ($)", f"{avg_price:,.0f}")
            
            # Топ районов по количеству продаж в выбранном Borough
            top_neighborhoods = borough_data['NEIGHBORHOOD'].value_counts().head(10)
            
            fig = px.bar(
                x=top_neighborhoods.index,
                y=top_neighborhoods.values,
                title=f"Топ 10 районов в {selected_borough_name} по количеству продаж",
                labels={'x': 'Район', 'y': 'Количество продаж'},
                color=top_neighborhoods.values,
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Детальная таблица по районам Borough
            neighborhood_details = borough_data.groupby('NEIGHBORHOOD').agg({
                'SALE PRICE': ['count', 'mean', 'median', 'min', 'max']
            }).round(2).reset_index()
            
            neighborhood_details.columns = [
                'Район', 
                'Количество продаж', 
                'Средняя цена', 
                'Медианная цена',
                'Минимальная цена',
                'Максимальная цена'
            ]
            
            neighborhood_details = neighborhood_details.sort_values('Количество продаж', ascending=False)
            
            # Пагинация для таблицы
            neighborhoods_page_size = st.selectbox("Районов на странице:", [10, 20, 50], key='neighborhoods_page')
            neighborhoods_page_number = st.number_input("Номер страницы:", min_value=1, value=1, key='neighborhoods_page_num')
            
            start_idx = (neighborhoods_page_number - 1) * neighborhoods_page_size
            end_idx = start_idx + neighborhoods_page_size
            
            st.dataframe(
                neighborhood_details.iloc[start_idx:end_idx].style.format({
                    'Средняя цена': '${:,.0f}',
                    'Медианная цена': '${:,.0f}',
                    'Минимальная цена': '${:,.0f}',
                    'Максимальная цена': '${:,.0f}'
                }),
                use_container_width=True,
                height=400
            )
    
    st.markdown("---")
    
    # Анализ ценовых сегментов
    st.subheader("Анализ ценовых сегментов")
    
    if 'SALE PRICE' in filtered_df.columns:
        # Создание ценовых категорий
        price_bins = [0, 1000000, 5000000, 10000000, 50000000, float('inf')]
        price_labels = ['< $1M', '$1M-$5M', '$5M-$10M', '$10M-$50M', '> $50M']
        
        filtered_df['PRICE_CATEGORY'] = pd.cut(
            filtered_df['SALE PRICE'],
            bins=price_bins,
            labels=price_labels,
            include_lowest=True
        )
        
        filtered_df_russian['Ценовая категория'] = filtered_df['PRICE_CATEGORY']
        
        price_dist = filtered_df['PRICE_CATEGORY'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=price_dist.values,
                names=price_dist.index,
                title="Распределение по ценовым категориям",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Статистика по ценовым категориям
            category_stats = filtered_df.groupby('PRICE_CATEGORY').agg({
                'SALE PRICE': ['count', 'mean', 'median'],
                'GROSS SQUARE FEET': 'mean'
            }).round(2).reset_index()
            
            category_stats.columns = ['Ценовая категория', 'Количество', 'Средняя цена', 'Медианная цена', 'Средняя площадь']
            
            # Вычисляем цену за кв.фут только для строк с корректной площадью
            valid_area_mask = category_stats['Средняя площадь'] > 0
            category_stats['Цена за кв.фут'] = np.where(
                valid_area_mask,
                category_stats['Средняя цена'] / category_stats['Средняя площадь'],
                np.nan
            )
            
            st.dataframe(
                category_stats.style.format({
                    'Средняя цена': '{:,.0f}',
                    'Медианная цена': '{:,.0f}',
                    'Средняя площадь': '{:,.0f}',
                    'Цена за кв.фут': '{:.2f}'
                }),
                use_container_width=True,
                height=300
            )
    
    st.markdown("---")
    
    # Инсайты
    st.subheader("Ключевые инсайты")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("##### Основные наблюдения:")
        
        if 'SALE PRICE' in filtered_df.columns:
            # Самый дорогой район
            if 'NEIGHBORHOOD' in filtered_df.columns:
                most_expensive = filtered_df.groupby('NEIGHBORHOOD')['SALE PRICE'].mean().idxmax()
                most_expensive_price = filtered_df.groupby('NEIGHBORHOOD')['SALE PRICE'].mean().max()
                st.write(f"**Самый дорогой район**: {most_expensive} (средняя цена: ${most_expensive_price:,.0f})")
            
            # Динамика цен
            if 'SALE DATE' in filtered_df.columns:
                recent_prices = filtered_df[filtered_df['SALE DATE'] > '2017-01-01']['SALE PRICE'].mean()
                older_prices = filtered_df[filtered_df['SALE DATE'] < '2017-01-01']['SALE PRICE'].mean()
                if older_prices > 0:
                    price_change = ((recent_prices - older_prices) / older_prices) * 100
                    st.write(f"**Изменение цен**: {price_change:+.1f}% с начала 2017 года")
    
    # Дополнительные опции анализа
    st.markdown("---")
    st.subheader("Дополнительные опции анализа")
    
    if st.button("Запустить углубленный анализ"):
        with st.spinner("Выполняется анализ..."):
            # Здесь можно добавить более сложный анализ
            st.success("Анализ завершен!")
            
            # Пример дополнительного анализа
            if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
                # Фильтруем только корректные данные
                valid_corr_data = filtered_df[(filtered_df['SALE PRICE'] > 0) & (filtered_df['GROSS SQUARE FEET'] > 0)]
                if not valid_corr_data.empty:
                    correlation = valid_corr_data['SALE PRICE'].corr(valid_corr_data['GROSS SQUARE FEET'])
                    st.write(f"**Корреляция цена-площадь**: {correlation:.3f}")
                    
                    if correlation > 0.7:
                        st.info("Сильная положительная корреляция: цена сильно зависит от площади")
                    elif correlation > 0.3:
                        st.warning("Умеренная корреляция: площадь влияет на цену, но есть другие факторы")
                    else:
                        st.info("Слабая корреляция: цена мало зависит от площади")
                else:
                    st.warning("Недостаточно данных для анализа корреляции")

# Страница 3: Анализ выбросов
elif page == "Анализ выбросов":
    st.title("📊 Анализ выбросов в данных")
    
    # Вступление
    st.markdown("""
    На этой странице представлен анализ выбросов (outliers) в данных о продажах недвижимости Нью-Йорка.
    Выбросы - это значения, которые значительно отличаются от остальных наблюдений и могут искажать статистический анализ.
    """)
    
    # Методы обнаружения выбросов
    st.markdown("---")
    st.subheader("🔍 Методы обнаружения выбросов")
    
    method = st.radio(
        "Выберите метод обнаружения выбросов:",
        ["Метод IQR (межквартильный размах)", "Метод Z-score", "Все методы"],
        horizontal=True
    )
    
    # Выбор колонок для анализа
    st.markdown("---")
    st.subheader("📈 Выберите переменные для анализа")
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Основные числовые колонки для анализа
    main_numeric_cols = ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET', 
                         'YEAR BUILT', 'TOTAL UNITS']
    
    available_cols = [col for col in main_numeric_cols if col in numeric_cols]
    
    selected_cols = st.multiselect(
        "Выберите переменные для анализа выбросов:",
        [COLUMN_TRANSLATIONS.get(col, col) for col in available_cols],
        default=[COLUMN_TRANSLATIONS.get('SALE PRICE', 'Цена продажи')]
    )
    
    # Преобразуем обратно в английские названия
    selected_cols_eng = [reverse_translate_column(col) for col in selected_cols]
    
    if selected_cols_eng:
        st.markdown("---")
        
        # 1. Статистика по выбросам
        st.subheader("📊 Статистика выбросов")
        
        outliers_stats = []
        
        for col in selected_cols_eng:
            if col in filtered_df.columns:
                data = filtered_df[col].dropna()
                
                if len(data) > 0:
                    # Базовые статистики
                    q1 = data.quantile(0.25)
                    q3 = data.quantile(0.75)
                    iqr = q3 - q1
                    
                    # Границы для IQR метода
                    lower_bound_iqr = q1 - 1.5 * iqr
                    upper_bound_iqr = q3 + 1.5 * iqr
                    
                    # Выбросы по IQR
                    outliers_iqr = data[(data < lower_bound_iqr) | (data > upper_bound_iqr)]
                    
                    # Z-score метод
                    z_scores = np.abs(stats.zscore(data))
                    outliers_zscore = data[z_scores > 3]
                    
                    outliers_stats.append({
                        'Переменная': COLUMN_TRANSLATIONS.get(col, col),
                        'Всего значений': len(data),
                        'Выбросов (IQR)': len(outliers_iqr),
                        '% выбросов (IQR)': f"{(len(outliers_iqr) / len(data) * 100):.2f}%",
                        'Выбросов (Z-score >3)': len(outliers_zscore),
                        '% выбросов (Z-score)': f"{(len(outliers_zscore) / len(data) * 100):.2f}%",
                        'Мин. значение': f"{data.min():,.2f}",
                        'Макс. значение': f"{data.max():,.2f}",
                        'Медиана': f"{data.median():,.2f}"
                    })
        
        if outliers_stats:
            stats_df = pd.DataFrame(outliers_stats)
            st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Визуализации для каждой выбранной колонки
        st.subheader("📉 Визуализация выбросов")
        
        for i, col in enumerate(selected_cols_eng):
            if col in filtered_df.columns:
                st.markdown(f"#### {COLUMN_TRANSLATIONS.get(col, col)}")
                
                data = filtered_df[col].dropna()
                
                if len(data) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box plot
                        fig = px.box(
                            filtered_df_russian,
                            y=COLUMN_TRANSLATIONS.get(col, col),
                            title=f"Box plot для {COLUMN_TRANSLATIONS.get(col, col)}",
                            points="all"
                        )
                        
                        # Добавляем аннотации для выбросов
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Находим выбросы
                        outliers = data[data > upper_bound]
                        
                        if len(outliers) > 0:
                            # Добавляем линию для верхней границы
                            fig.add_hline(
                                y=upper_bound,
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Верхняя граница: {upper_bound:,.2f}",
                                annotation_position="bottom right"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Гистограмма с выделением выбросов
                        fig = px.histogram(
                            filtered_df_russian,
                            x=COLUMN_TRANSLATIONS.get(col, col),
                            nbins=50,
                            title=f"Распределение {COLUMN_TRANSLATIONS.get(col, col)}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Scatter plot для выбросов по времени (если есть дата)
                    if 'SALE DATE' in filtered_df.columns and i == 0:
                        st.markdown("##### Выбросы по времени")
                        
                        # Создаем флаг выбросов
                        q1 = data.quantile(0.25)
                        q3 = data.quantile(0.75)
                        iqr = q3 - q1
                        upper_bound = q3 + 1.5 * iqr
                        
                        filtered_df_with_outliers = filtered_df.copy()
                        filtered_df_with_outliers['is_outlier'] = filtered_df_with_outliers[col] > upper_bound
                        
                        fig = px.scatter(
                            filtered_df_with_outliers,
                            x='SALE DATE',
                            y=col,
                            color='is_outlier',
                            title=f"Выбросы {COLUMN_TRANSLATIONS.get(col, col)} по времени",
                            labels={
                                'SALE DATE': 'Дата продажи',
                                col: COLUMN_TRANSLATIONS.get(col, col),
                                'is_outlier': 'Выброс'
                            },
                            color_discrete_map={True: 'red', False: 'blue'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
        
        # 3. Матрица scatter plots для многомерного анализа
        st.subheader("🔗 Многомерный анализ выбросов")
        
        if len(selected_cols_eng) >= 2:
            # Выбираем две основные переменные
            col_x = st.selectbox(
                "Выберите переменную для оси X:",
                [COLUMN_TRANSLATIONS.get(col, col) for col in selected_cols_eng],
                index=0
            )
            
            col_y = st.selectbox(
                "Выберите переменную для оси Y:",
                [COLUMN_TRANSLATIONS.get(col, col) for col in selected_cols_eng],
                index=min(1, len(selected_cols_eng)-1)
            )
            
            col_x_eng = reverse_translate_column(col_x)
            col_y_eng = reverse_translate_column(col_y)
            
            if col_x_eng in filtered_df.columns and col_y_eng in filtered_df.columns:
                # Создаем флаг выбросов для обеих переменных
                data_x = filtered_df[col_x_eng].dropna()
                data_y = filtered_df[col_y_eng].dropna()
                
                if len(data_x) > 0 and len(data_y) > 0:
                    # Вычисляем выбросы для обеих переменных
                    q1_x = data_x.quantile(0.25)
                    q3_x = data_x.quantile(0.75)
                    iqr_x = q3_x - q1_x
                    upper_bound_x = q3_x + 1.5 * iqr_x
                    
                    q1_y = data_y.quantile(0.25)
                    q3_y = data_y.quantile(0.75)
                    iqr_y = q3_y - q1_y
                    upper_bound_y = q3_y + 1.5 * iqr_y
                    
                    # Флаг выбросов
                    filtered_df['outlier_x'] = filtered_df[col_x_eng] > upper_bound_x
                    filtered_df['outlier_y'] = filtered_df[col_y_eng] > upper_bound_y
                    filtered_df['is_outlier'] = filtered_df['outlier_x'] | filtered_df['outlier_y']
                    
                    # Scatter plot с выделением выбросов
                    fig = px.scatter(
                        filtered_df,
                        x=col_x_eng,
                        y=col_y_eng,
                        color='is_outlier',
                        title=f"Многомерные выбросы: {col_x} vs {col_y}",
                        labels={
                            col_x_eng: col_x,
                            col_y_eng: col_y,
                            'is_outlier': 'Выброс'
                        },
                        color_discrete_map={True: 'red', False: 'blue'},
                        hover_data=['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY']
                    )
                    
                    # Добавляем линии границ
                    fig.add_vline(
                        x=upper_bound_x,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Граница {col_x}",
                        annotation_position="top right"
                    )
                    
                    fig.add_hline(
                        y=upper_bound_y,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Гранциа {col_y}",
                        annotation_position="bottom right"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Статистика по многомерным выбросам
                    outlier_count = filtered_df['is_outlier'].sum()
                    total_count = len(filtered_df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Многомерных выбросов", outlier_count)
                    with col2:
                        st.metric("Всего записей", total_count)
                    with col3:
                        st.metric("Доля выбросов", f"{(outlier_count/total_count*100):.2f}%")
        
        # 4. Детальный просмотр выбросов
        st.markdown("---")
        st.subheader("🔎 Детальный просмотр выбросов")
        
        # Выбор колонки для детального анализа
        detail_col = st.selectbox(
            "Выберите переменную для детального просмотра выбросов:",
            [COLUMN_TRANSLATIONS.get(col, col) for col in selected_cols_eng]
        )
        
        detail_col_eng = reverse_translate_column(detail_col)
        
        if detail_col_eng in filtered_df.columns:
            data = filtered_df[detail_col_eng].dropna()
            
            if len(data) > 0:
                # Вычисляем границы
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                
                # Получаем выбросы
                outliers_df = filtered_df[filtered_df[detail_col_eng] > upper_bound].copy()
                
                # Сортируем по значению выброса
                outliers_df = outliers_df.sort_values(detail_col_eng, ascending=False)
                
                # Добавляем информацию о том, насколько значение превышает границу
                outliers_df['excess_percentage'] = ((outliers_df[detail_col_eng] - upper_bound) / upper_bound * 100).round(2)
                
                st.write(f"**Найдено выбросов: {len(outliers_df)}**")
                st.write(f"**Верхняя граница: {upper_bound:,.2f}**")
                
                # Показываем топ выбросов
                if len(outliers_df) > 0:
                    # Выбираем колонки для отображения
                    display_cols = [
                        'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 
                        'SALE PRICE', 'GROSS SQUARE FEET', 'YEAR BUILT',
                        detail_col_eng, 'excess_percentage'
                    ]
                    
                    available_display_cols = [col for col in display_cols if col in outliers_df.columns]
                    
                    # Переводим названия колонок
                    outliers_display = outliers_df[available_display_cols].copy()
                    
                    # Переименовываем для отображения
                    rename_dict = {}
                    for col in available_display_cols:
                        if col == detail_col_eng:
                            rename_dict[col] = f"{detail_col} (значение)"
                        elif col == 'excess_percentage':
                            rename_dict[col] = 'Превышение границы (%)'
                        else:
                            rename_dict[col] = COLUMN_TRANSLATIONS.get(col, col)
                    
                    outliers_display = outliers_display.rename(columns=rename_dict)
                    
                    # Форматирование чисел
                    st.dataframe(
                        outliers_display.style.format({
                            f"{detail_col} (значение)": '{:,.2f}',
                            'Цена продажи': '{:,.2f}',
                            'Общая площадь (кв. фут)': '{:,.2f}',
                            'Превышение границы (%)': '{:,.2f}%'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Экспорт выбросов
                    csv = outliers_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Скачать данные выбросов (CSV)",
                        data=csv,
                        file_name="nyc_property_outliers.csv",
                        mime="text/csv",
                    )
        
        # 5. Рекомендации по обработке выбросов
        st.markdown("---")
        st.subheader("💡 Рекомендации по обработке выбросов")
        
        st.markdown("""
        ### Что делать с выбросами?
        
        1. **Анализ природы выбросов**:
           - Проверьте, не являются ли выбросы ошибками в данных
           - Проанализируйте, представляют ли они реальные редкие случаи (например, продажи элитной недвижимости)
        
        2. **Методы обработки**:
           - **Удаление**: Если выбросы являются ошибками или сильно искажают анализ
           - **Трансформация**: Логарифмирование данных для уменьшения влияния выбросов
           - **Винсоризация**: Замена выбросов на граничные значения
           - **Сохранение**: Если выбросы представляют интересные случаи для анализа
        
        3. **Для данного датасета**:
           - Выбросы в цене могут представлять реальные продажи элитной недвижимости
           - Выбросы в площади могут быть коммерческими объектами
           - Рекомендуется анализировать выбросы отдельно от основной массы данных
        """)
        
        # Быстрое действие: создание очищенного датасета
        if st.button("🔄 Создать очищенную версию данных (без выбросов)"):
            with st.spinner("Удаляем выбросы..."):
                cleaned_df = filtered_df.copy()
                
                for col in selected_cols_eng:
                    if col in cleaned_df.columns:
                        data = cleaned_df[col].dropna()
                        if len(data) > 0:
                            q1 = data.quantile(0.25)
                            q3 = data.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            # Удаляем выбросы
                            mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
                            cleaned_df = cleaned_df[mask | cleaned_df[col].isna()]
                
                st.success(f"Данные очищены! Осталось {len(cleaned_df)} записей из {len(filtered_df)}")
                
                # Показываем сравнение
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Исходные данные", len(filtered_df))
                with col2:
                    st.metric("Очищенные данные", len(cleaned_df))
                
                # Скачать очищенные данные
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать очищенные данные (CSV)",
                    data=csv,
                    file_name="nyc_property_cleaned.csv",
                    mime="text/csv",
                )
    
    else:
        st.warning("Выберите хотя бы одну переменную для анализа выбросов")

# Страница 4: Таблица переводов
elif page == "Таблица переводов":
    st.title("Таблица переводов названий колонок")
    
    # Создаем таблицу с переводами
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

# Информация в футере
st.sidebar.markdown("---")

# Добавляем возможность сброса фильтров
if st.sidebar.button("Сбросить все фильтры"):
    st.rerun()
