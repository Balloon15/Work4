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
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
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

# Функция для нормализации категорий зданий
def normalize_building_categories(data):
    """
    Нормализует категории зданий для устранения несоответствий в данных
    """
    if 'BUILDING CLASS CATEGORY' not in data.columns:
        return data
    
    data = data.copy()
    
    # Создаем нормализованную версию категорий
    def normalize_category(cat):
        if pd.isna(cat):
            return "Неизвестно"
        
        cat_str = str(cat).strip()
        
        # Извлекаем числовой код (если есть)
        code_match = re.search(r'^(\d+)', cat_str)
        code = code_match.group(1) if code_match else "00"
        
        # Приводим к нижнему регистру для поиска
        cat_lower = cat_str.lower()
        
        # Определяем тип по содержанию (исправляем проблему с категорией 38)
        if any(word in cat_lower for word in ['asylum', 'home', 'hospital', 'institution', 'nursing', 'приют', 'больница']):
            return f"{code} - УЧРЕЖДЕНИЕ (больницы/приюты)"
        
        elif any(word in cat_lower for word in ['condo', 'кондо', 'апартаменты']):
            return f"{code} - КОНДОМИНИУМ"
        
        elif any(word in cat_lower for word in ['coop', 'кооп']):
            return f"{code} - КООПЕРАТИВ"
        
        elif any(word in cat_lower for word in ['residential', 'жилой', 'dwelling', 'квартир']):
            return f"{code} - ЖИЛАЯ НЕДВИЖИМОСТЬ"
        
        elif any(word in cat_lower for word in ['store', 'office', 'commercial', 'retail', 'магазин', 'офис']):
            return f"{code} - КОММЕРЧЕСКАЯ НЕДВИЖИМОСТЬ"
        
        elif any(word in cat_lower for word in ['mixed', 'multi-use', 'смешанный']):
            return f"{code} - СМЕШАННОЕ ИСПОЛЬЗОВАНИЕ"
        
        elif any(word in cat_lower for word in ['factory', 'industrial', 'warehouse', 'завод', 'склад']):
            return f"{code} - ПРОМЫШЛЕННАЯ НЕДВИЖИМОСТЬ"
        
        elif any(word in cat_lower for word in ['vacant', 'land', 'пустующий', 'земля']):
            return f"{code} - ЗЕМЕЛЬНЫЙ УЧАСТОК"
        
        else:
            return f"{code} - ДРУГОЙ ТИП"
    
    # Применяем нормализацию
    data['BUILDING_CATEGORY_NORMALIZED'] = data['BUILDING CLASS CATEGORY'].apply(normalize_category)
    
    # Также сохраняем оригинальную категорию для справки
    data['BUILDING_CATEGORY_ORIGINAL'] = data['BUILDING CLASS CATEGORY']
    
    return data

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

# Загрузка данных с очисткой выбросов и нормализацией категорий
@st.cache_data
def load_data():
    data = pd.read_csv("nyc-rolling-sales.csv")
    
    # Сохраняем информацию об исходном объеме
    original_rows = len(data)
    
    numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                       'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                       'TOTAL UNITS', 'ZIP CODE']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col].replace(' -  ', np.nan).replace(' - ', np.nan).replace(' -', np.nan), errors='coerce')
    
    if 'SALE DATE' in data.columns:
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
    
    # СОЗДАЕМ BOROUGH_NAME для использования в фильтрах
    if 'BOROUGH' in data.columns:
        borough_names = {
            1: 'Манхэттен',
            2: 'Бруклин', 
            3: 'Квинс',
            4: 'Бронкс',
            5: 'Статен-Айленд'
        }
        data['BOROUGH_NAME'] = data['BOROUGH'].map(borough_names)
    
    # УЛУЧШЕННАЯ ОЧИСТКА ЦЕН
    if 'SALE PRICE' in data.columns:
        # Удаляем только явно некорректные значения (<= 0)
        data = data[data['SALE PRICE'] > 0]
        
        # Сохраняем 99% данных (удаляем только 0.5% с каждой стороны)
        price_005 = data['SALE PRICE'].quantile(0.005)  # 0.5-й процентиль
        price_995 = data['SALE PRICE'].quantile(0.995)  # 99.5-й процентиль
        
        # Устанавливаем разумный минимум для Нью-Йорка
        reasonable_min_price = 1000  # $1,000
        final_min_price = max(price_005, reasonable_min_price)
        
        # Устанавливаем разумный максимум
        reasonable_max_price = 100_000_000  # $100M
        final_max_price = min(price_995, reasonable_max_price)
        
        data = data[(data['SALE PRICE'] >= final_min_price) & 
                   (data['SALE PRICE'] <= final_max_price)]
        
        # Создаем логарифмированную версию цены для анализа
        data['LOG_SALE_PRICE'] = np.log1p(data['SALE PRICE'])
    
    # ОЧИСТКА ГОДА ПОСТРОЙКИ
    if 'YEAR BUILT' in data.columns:
        current_year = datetime.now().year
        # Сохраняем здания с 1600 года (исторические здания Нью-Йорка)
        data = data[(data['YEAR BUILT'] >= 1600) & 
                   (data['YEAR BUILT'] <= current_year)]
        
        # Заполняем пропуски медианой по округу
        if data['YEAR BUILT'].isna().any() and 'BOROUGH_NAME' in data.columns:
            median_year_by_borough = data.groupby('BOROUGH_NAME')['YEAR BUILT'].median()
            data['YEAR BUILT'] = data.apply(
                lambda row: median_year_by_borough[row['BOROUGH_NAME']] 
                if pd.isna(row['YEAR BUILT']) else row['YEAR BUILT'],
                axis=1
            )
    
    # ОЧИСТКА ПЛОЩАДИ
    for area_col in ['GROSS SQUARE FEET', 'LAND SQUARE FEET']:
        if area_col in data.columns:
            # Удаляем отрицательные значения
            data = data[data[area_col] >= 0]
            
            # Используем 99.5% процентиль для удаления выбросов
            if data[area_col].notna().any():
                area_995 = data[area_col].quantile(0.995)
                data = data[(data[area_col] <= area_995) | (data[area_col].isna())]
            
            # Создаем логарифмированную версию
            data[f'LOG_{area_col}'] = np.log1p(data[area_col].fillna(0))
    
    # ИМПУТАЦИЯ ПРОПУСКОВ вместо удаления строк
    numeric_cols_for_imputation = ['GROSS SQUARE FEET', 'LAND SQUARE FEET', 
                                  'YEAR BUILT', 'TOTAL UNITS', 'RESIDENTIAL UNITS', 
                                  'COMMERCIAL UNITS']
    
    for col in numeric_cols_for_imputation:
        if col in data.columns and data[col].isna().any():
            # Заполняем медианой по округу и типу здания
            if 'BOROUGH_NAME' in data.columns and 'BUILDING CLASS CATEGORY' in data.columns:
                # Сначала по округу и типу
                data[col] = data.groupby(['BOROUGH_NAME', 'BUILDING CLASS CATEGORY'])[col]\
                               .transform(lambda x: x.fillna(x.median()))
                # Затем по округу
                data[col] = data.groupby('BOROUGH_NAME')[col]\
                               .transform(lambda x: x.fillna(x.median()))
            # В крайнем случае - общей медианой
            data[col] = data[col].fillna(data[col].median())
    
    # РАСЧЕТ ЦЕНЫ ЗА КВ.ФУТ
    if all(col in data.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
        data['PRICE_PER_SQFT'] = data['SALE PRICE'] / data['GROSS SQUARE FEET'].replace(0, np.nan)
        
        # Очистка выбросов в цене за кв.фут
        if data['PRICE_PER_SQFT'].notna().any():
            # Сохраняем 98% данных
            pq1 = data['PRICE_PER_SQFT'].quantile(0.01)
            pq3 = data['PRICE_PER_SQFT'].quantile(0.99)
            data = data[(data['PRICE_PER_SQFT'] >= pq1) & 
                       (data['PRICE_PER_SQFT'] <= pq3) | 
                       (data['PRICE_PER_SQFT'].isna())]
    
    # СОЗДАЕМ ПРОИЗВОДНЫЕ ПРИЗНАКИ
    if 'YEAR BUILT' in data.columns:
        current_year = datetime.now().year
        data['BUILDING_AGE'] = current_year - data['YEAR BUILT']
        data['IS_HISTORIC'] = (data['BUILDING_AGE'] > 100).astype(int)
    
    if all(col in data.columns for col in ['GROSS SQUARE FEET', 'TOTAL UNITS']):
        data['SQFT_PER_UNIT'] = data['GROSS SQUARE FEET'] / data['TOTAL UNITS'].replace(0, 1)
    
    # Удаляем дубликаты по ключевым полям
    data = data.drop_duplicates(subset=['ADDRESS', 'SALE DATE', 'SALE PRICE'], keep='first')
    
    # НОРМАЛИЗАЦИЯ КАТЕГОРИЙ ЗДАНИЙ (решение проблемы с типами)
    data = normalize_building_categories(data)
    
    # ИНФОРМАЦИЯ ОБ ОЧИСТКЕ
    final_rows = len(data)
    retention_rate = (final_rows / original_rows) * 100
    
    st.session_state.data_cleaning_stats = {
        'original_rows': original_rows,
        'final_rows': final_rows,
        'retention_rate': retention_rate,
        'removed_rows': original_rows - final_rows,
        'min_price': data['SALE PRICE'].min() if 'SALE PRICE' in data.columns else 0,
        'max_price': data['SALE PRICE'].max() if 'SALE PRICE' in data.columns else 0
    }
    
    return data

# Загружаем данные
df = load_data()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ рынка", "Прогнозные модели", 
     "Таблица переводов", "Анализ системы классификации"]
)

# Добавляем фильтры в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("Фильтры данных")

# Создаем копию с русскими названиями для фильтров
df_russian = translate_columns(df.copy())

# Фильтр по району (используем оригинальные названия)
neighborhoods = ['Все'] + sorted(df['NEIGHBORHOOD'].dropna().unique().tolist())
selected_neighborhood = st.sidebar.selectbox(
    "Район", 
    neighborhoods
)

# Фильтр по нормализованным типам зданий (решение проблемы с категориями)
if 'BUILDING_CATEGORY_NORMALIZED' in df.columns:
    building_categories = ['Все'] + sorted(df['BUILDING_CATEGORY_NORMALIZED'].dropna().unique().tolist())
    selected_building_category = st.sidebar.selectbox(
        "Категория здания (нормализованная)", 
        building_categories
    )
else:
    building_categories = ['Все'] + sorted(df['BUILDING CLASS CATEGORY'].dropna().unique().tolist())
    selected_building_category = st.sidebar.selectbox(
        "Категория здания", 
        building_categories
    )

# Фильтр по году постройки
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
if 'SALE PRICE' in df.columns:
    realistic_min_price = 1000
    realistic_max_price = 50_000_000
    
    price_range = st.sidebar.slider(
        "Цена продажи ($)",
        min_value=int(realistic_min_price),
        max_value=int(realistic_max_price),
        value=(int(realistic_min_price), int(realistic_max_price)),
        step=1000
    )

# Применяем фильтры
filtered_df = df.copy()

if selected_neighborhood != 'Все':
    filtered_df = filtered_df[filtered_df['NEIGHBORHOOD'] == selected_neighborhood]

if selected_building_category != 'Все':
    if 'BUILDING_CATEGORY_NORMALIZED' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['BUILDING_CATEGORY_NORMALIZED'] == selected_building_category]
    else:
        filtered_df = filtered_df[filtered_df['BUILDING CLASS CATEGORY'] == selected_building_category]

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

# Создаем производные колонки
if 'SALE DATE' in filtered_df.columns:
    filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.month
    filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
    
if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
    filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET']
    
if 'YEAR BUILT' in filtered_df.columns:
    filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']

# Создаем DataFrame с русскими названиями для отображения
filtered_df_russian = translate_columns(filtered_df.copy())

# Страница 5: Анализ системы классификации
if page == "Анализ системы классификации":
    st.title("Анализ системы классификации зданий")
    
    st.warning("""
    ⚠️ **Обнаружена проблема с системой классификации!**
    
    В вашем наборе данных категория зданий использует нестандартную систему.
    Например, категория '38' отображается как 'asylums and homes' (учреждения/приюты),
    в то время как в стандартной системе NYC '38' означает 'CONDOMINIUMS'.
    
    Для решения этой проблемы мы:
    1. Нормализовали все категории зданий
    2. Сгруппировали их по смыслу
    3. Создали понятные названия на русском языке
    """)
    
    # Анализ оригинальных категорий
    st.subheader("1. Оригинальные категории зданий в данных")
    
    if 'BUILDING CLASS CATEGORY' in df.columns:
        original_categories = df['BUILDING CLASS CATEGORY'].value_counts().reset_index()
        original_categories.columns = ['Оригинальная категория', 'Количество']
        
        st.dataframe(
            original_categories.head(20),
            use_container_width=True,
            height=400
        )
        
        st.write(f"**Всего уникальных категорий:** {len(original_categories)}")
    
    # Анализ нормализованных категорий
    st.subheader("2. Нормализованные категории (после обработки)")
    
    if 'BUILDING_CATEGORY_NORMALIZED' in df.columns:
        normalized_categories = df['BUILDING_CATEGORY_NORMALIZED'].value_counts().reset_index()
        normalized_categories.columns = ['Нормализованная категория', 'Количество']
        
        # Визуализация
        fig = px.bar(
            normalized_categories.head(15),
            x='Нормализованная категория',
            y='Количество',
            title='Распределение по нормализованным категориям',
            color='Количество'
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            normalized_categories,
            use_container_width=True,
            height=400
        )
    
    # Сравнение оригинальных и нормализованных категорий
    st.subheader("3. Сравнение категорий")
    
    if 'BUILDING CLASS CATEGORY' in df.columns and 'BUILDING_CATEGORY_NORMALIZED' in df.columns:
        # Выбираем несколько примеров для сравнения
        sample_data = df[['BUILDING CLASS CATEGORY', 'BUILDING_CATEGORY_NORMALIZED']].dropna()
        
        # Группируем по соответствию
        comparison = sample_data.groupby(['BUILDING CLASS CATEGORY', 'BUILDING_CATEGORY_NORMALIZED'])\
                               .size().reset_index(name='Количество')
        
        st.write("**Соответствие оригинальных и нормализованных категорий:**")
        st.dataframe(
            comparison.sort_values('Количество', ascending=False).head(20),
            use_container_width=True
        )
    
    # Особый анализ категории 38
    st.subheader("4. Особый анализ: Категория '38'")
    
    # Ищем все варианты категории 38
    category_38_variants = []
    if 'BUILDING CLASS CATEGORY' in df.columns:
        # Ищем все записи, содержащие 38
        mask_38 = df['BUILDING CLASS CATEGORY'].astype(str).str.contains('38', na=False)
        category_38_data = df[mask_38]
        
        if not category_38_data.empty:
            st.write(f"**Найдено записей с категорией 38:** {len(category_38_data)}")
            
            # Уникальные названия
            unique_names = category_38_data['BUILDING CLASS CATEGORY'].unique()
            st.write(f"**Уникальные названия категории 38:**")
            for name in unique_names:
                st.write(f"- `{name}`")
            
            # Статистика по категории 38
            col1, col2, col3 = st.columns(3)
            with col1:
                median_price = category_38_data['SALE PRICE'].median()
                st.metric("Медианная цена", f"${median_price:,.0f}")
            
            with col2:
                avg_area = category_38_data['GROSS SQUARE FEET'].median()
                st.metric("Медианная площадь", f"{avg_area:,.0f} кв.фут")
            
            with col3:
                borough_dist = category_38_data['BOROUGH_NAME'].value_counts()
                st.write("**Распределение по округам:**")
                for borough, count in borough_dist.items():
                    st.write(f"- {borough}: {count}")
    
    # Рекомендации
    st.subheader("5. Рекомендации по использованию")
    
    st.success("""
    ✅ **Проблема решена!**
    
    **Для дальнейшего анализа используйте:**
    
    1. **Нормализованные категории** (`BUILDING_CATEGORY_NORMALIZED`) - для группировки и анализа
    2. **Оригинальные категории** (`BUILDING CLASS CATEGORY`) - только для справки
    
    **Преимущества нормализованных категорий:**
    - Единая система классификации
    - Понятные названия на русском
    - Группировка по смыслу (жилая, коммерческая и т.д.)
    - Устранение несоответствий в данных
    """)

# Страница 4: Таблица переводов
elif page == "Таблица переводов":
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

# Страница 1: Визуализация данных
elif page == "Визуализация данных":
    st.title("Визуализация данных о продажах недвижимости Нью-Йорка")
    
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
        unique_neighborhoods = filtered_df['NEIGHBORHOOD'].nunique()
        st.metric("Количество районов", unique_neighborhoods)

    st.markdown("---")
    
    # Информация о нормализации категорий
    st.info("""
    💡 **Используются нормализованные категории зданий** 
    
    Для устранения несоответствий в классификации все типы зданий были нормализованы.
    Категория '38' (asylums and homes) теперь корректно отображается как 'УЧРЕЖДЕНИЕ'.
    """)
    
    # Таблица с данными
    st.subheader("Просмотр данных")
    
    # Добавляем нормализованную категорию в список колонок
    all_columns_russian = filtered_df_russian.columns.tolist()
    
    # Если есть нормализованная категория, добавляем ее в начало
    if 'BUILDING_CATEGORY_NORMALIZED' in filtered_df.columns:
        # Добавляем нормализованную категорию в русскую версию
        filtered_df_russian['Нормализованная категория здания'] = filtered_df['BUILDING_CATEGORY_NORMALIZED']
        all_columns_russian.append('Нормализованная категория здания')
    
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
    col1, col2 = st.columns(2)
    
    with col1:
        if 'SALE PRICE' in filtered_df.columns:
            fig = px.histogram(
                filtered_df_russian, 
                x='Цена продажи',
                nbins=50,
                title="Распределение цен на недвижимость",
                labels={'Цена продажи': 'Цена продажи ($)'}
            )
            fig.update_layout(xaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
            
        if 'YEAR BUILT' in filtered_df.columns:
            valid_year_data = filtered_df_russian[filtered_df_russian['Год постройки'] > 0]
            if not valid_year_data.empty:
                fig = px.histogram(
                    valid_year_data,
                    x='Год постройки',
                    nbins=30,
                    title="Распределение по году постройки",
                    labels={'Год постройки': 'Год'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'BOROUGH' in filtered_df.columns:
            filtered_df['BOROUGH_NAME'] = filtered_df['BOROUGH'].map({
                1: 'Манхэттен',
                2: 'Бруклин', 
                3: 'Квинс',
                4: 'Бронкс',
                5: 'Статен-Айленд'
            })
            
            borough_counts = filtered_df['BOROUGH_NAME'].value_counts()
            fig = px.pie(
                values=borough_counts.values,
                names=borough_counts.index,
                title="Распределение продаж по округам",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
        if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
            fig = px.scatter(
                filtered_df_russian,
                x='Общая площадь (кв. фут)',
                y='Цена продажи',
                title="Цена vs Общая площадь",
                labels={
                    'Общая площадь (кв. фут)': 'Площадь (кв. фут)',
                    'Цена продажи': 'Цена ($)'
                },
                opacity=0.6
            )
            fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
    
    # Визуализация по нормализованным категориям
    st.markdown("---")
    st.subheader("Анализ по нормализованным категориям зданий")
    
    if 'BUILDING_CATEGORY_NORMALIZED' in filtered_df.columns:
        # Распределение по категориям
        category_dist = filtered_df['BUILDING_CATEGORY_NORMALIZED'].value_counts().reset_index()
        category_dist.columns = ['Категория', 'Количество']
        
        fig = px.bar(
            category_dist.head(15),
            x='Категория',
            y='Количество',
            title='Распределение по типам зданий (нормализованные)',
            color='Количество'
        )
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Цены по категориям
        price_by_category = filtered_df.groupby('BUILDING_CATEGORY_NORMALIZED')['SALE PRICE'].median().sort_values(ascending=False).reset_index()
        price_by_category.columns = ['Категория', 'Медианная цена']
        
        fig2 = px.bar(
            price_by_category.head(15),
            x='Категория',
            y='Медианная цена',
            title='Медианная цена по типам зданий',
            color='Медианная цена'
        )
        fig2.update_xaxes(tickangle=45, tickfont=dict(size=10))
        fig2.update_layout(yaxis_tickformat=',')
        st.plotly_chart(fig2, use_container_width=True)

# Страница 2: Анализ рынка
elif page == "Анализ рынка":
    st.title("Анализ рынка недвижимости Нью-Йорка")
    
    analysis_type = st.selectbox(
        "Выберите тип анализа:",
        ["Анализ по районам", "Анализ по типам зданий", 
         "Стоимость квадратного фута", "Возраст vs Цена", 
         "Анализ нормализованных категорий"]
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
            
            # Добавляем цену за кв.фут
            neighborhood_stats['Цена за кв.фут'] = neighborhood_stats['Медианная цена'] / neighborhood_stats['Медианная площадь']
            
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
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
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
        
        st.info("""
        💡 **Используются нормализованные категории зданий**
        
        Для корректного анализа все типы зданий были сгруппированы по смыслу.
        Это устраняет несоответствия в исходной классификации.
        """)
        
        # Используем нормализованные категории
        if 'BUILDING_CATEGORY_NORMALIZED' in filtered_df.columns:
            building_stats = filtered_df.groupby('BUILDING_CATEGORY_NORMALIZED').agg({
                'SALE PRICE': ['median', 'count', 'std'],
                'GROSS SQUARE FEET': 'median',
                'TOTAL UNITS': 'median'
            }).round(2)
            
            building_stats.columns = ['Медианная цена', 'Количество', 'Стд. отклонение', 
                                      'Медианная площадь', 'Медианное кол-во единиц']
            
            # Топ-10 типов по цене
            top_buildings = building_stats.nlargest(10, 'Медианная цена')
            
            fig = px.bar(
                top_buildings.reset_index(),
                x='BUILDING_CATEGORY_NORMALIZED',
                y='Медианная цена',
                title='Топ-10 самых дорогих типов недвижимости (нормализованные)',
                color='Медианная цена'
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Дополнительная информация о категориях
            st.subheader("Детальная информация по категориям")
            
            for idx, (category, row) in enumerate(top_buildings.iterrows(), 1):
                with st.expander(f"{idx}. {category}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Медианная цена", f"${row['Медианная цена']:,.0f}")
                    with col2:
                        st.metric("Количество объектов", f"{row['Количество']:,.0f}")
                    with col3:
                        st.metric("Медианная площадь", f"{row['Медианная площадь']:,.0f} кв.фут")
    
    elif analysis_type == "Анализ нормализованных категорий":
        st.subheader("Подробный анализ нормализованных категорий")
        
        if 'BUILDING_CATEGORY_NORMALIZED' in filtered_df.columns:
            # Группируем по основным типам (первая часть до дефиса)
            filtered_df['MAIN_CATEGORY'] = filtered_df['BUILDING_CATEGORY_NORMALIZED'].apply(
                lambda x: x.split(' - ')[0] if ' - ' in str(x) else str(x)
            )
            
            # Анализ по основным типам
            main_category_stats = filtered_df.groupby('MAIN_CATEGORY').agg({
                'SALE PRICE': ['median', 'count'],
                'GROSS SQUARE FEET': 'median',
                'PRICE_PER_SQFT': 'median'
            }).round(2)
            
            main_category_stats.columns = ['Медианная цена', 'Количество', 'Медианная площадь', 'Медианная цена за кв.фут']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Количество объектов по типам
                fig = px.pie(
                    main_category_stats.reset_index(),
                    values='Количество',
                    names='MAIN_CATEGORY',
                    title='Распределение объектов по основным типам',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Цены по типам
                fig = px.bar(
                    main_category_stats.reset_index(),
                    x='MAIN_CATEGORY',
                    y='Медианная цена',
                    title='Медианная цена по основным типам',
                    color='Медианная цена'
                )
                fig.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
            
            # Таблица с детальной статистикой
            st.subheader("Детальная статистика по категориям")
            
            detailed_stats = filtered_df.groupby('BUILDING_CATEGORY_NORMALIZED').agg({
                'SALE PRICE': ['median', 'min', 'max'],
                'GROSS SQUARE FEET': 'median',
                'PRICE_PER_SQFT': 'median',
                'BUILDING_AGE': 'median',
                'BOROUGH_NAME': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
            }).round(2)
            
            detailed_stats.columns = ['Медианная цена', 'Мин. цена', 'Макс. цена', 
                                     'Медианная площадь', 'Медианная цена за кв.фут',
                                     'Медианный возраст', 'Наиболее частый округ']
            
            st.dataframe(
                detailed_stats.style.format({
                    'Медианная цена': '${:,.0f}',
                    'Мин. цена': '${:,.0f}',
                    'Макс. цена': '${:,.0f}',
                    'Медианная площадь': '{:,.0f}',
                    'Медианная цена за кв.фут': '${:.2f}',
                    'Медианный возраст': '{:.0f} лет'
                }),
                use_container_width=True,
                height=400
            )

# Дополнительные страницы (Прогнозные модели) остаются без изменений
# ... [остальной код остается без изменений]
