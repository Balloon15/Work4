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
    # Переводит названия колонок DataFrame на русский язык
    translated_cols = []
    for col in df.columns:
        translated_cols.append(COLUMN_TRANSLATIONS.get(col, col))
    df.columns = translated_cols
    return df

# Функция для обратного перевода (для фильтров)
def reverse_translate_column(russian_name):
    # Возвращает оригинальное название колонки по русскому переводу
    for eng, rus in COLUMN_TRANSLATIONS.items():
        if rus == russian_name:
            return eng
    return russian_name

# Загрузка данных
@st.cache_data
def load_data():
    # Читаем данные из строкового буфера
    data = pd.read_csv("nyc-rolling-sales.csv")
    
    # Преобразование типов данных в числовые форматы  
    numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                       'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                       'TOTAL UNITS', 'ZIP CODE']
    
    for col in numeric_columns:
        if col in data.columns:
            # Заменяем нечисловые значения и преобразуем
            data[col] = pd.to_numeric(data[col].replace(' -  ', np.nan).replace(' - ', np.nan).replace(' -', np.nan), errors='coerce')
    
    # Преобразуем дату
    if 'SALE DATE' in data.columns:
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
    
    # Очистка цены продажи
    if 'SALE PRICE' in data.columns:
        # Удаляем строки с некорректными ценами
        data = data[data['SALE PRICE'] > 0]
    
    # Очистка года постройки - удаляем некорректные значения (<= 0 или слишком старые)
    if 'YEAR BUILT' in data.columns:
        data = data[data['YEAR BUILT'] > 0]
    
    return data

# Загружаем данные
df = load_data()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация исходных данных", "Результаты анализа", "Таблица переводов"]
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

# Фильтр по году постройки (ИСПРАВЛЕНО)
if 'YEAR BUILT' in df.columns:
    # Фильтруем только корректные годы (> 0)
    valid_years = df[df['YEAR BUILT'] > 0]['YEAR BUILT']
    
    if not valid_years.empty:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        
        # Проверяем, что минимальный год разумен (не раньше 1700)
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
        # Если нет корректных данных о годе постройки
        year_range = (1800, 2023)  # Значения по умолчанию для Нью-Йорка
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

# Страница 3: Таблица переводов
if page == "Таблица переводов":
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
            # Используем только корректные годы для расчета среднего
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
else:
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

# Информация в футере
st.sidebar.markdown("---")

# Добавляем возможность сброса фильтров
if st.sidebar.button("Сбросить все фильтры"):
    st.rerun()
