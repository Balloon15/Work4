import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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

# ИСПРАВЛЕННАЯ функция загрузки данных с более щадящей очисткой
@st.cache_data
def load_data():
    # Загрузка данных
    data = pd.read_csv("nyc-rolling-sales.csv")
    
    # Сохраняем информацию об исходном объеме
    original_rows = len(data)
    
    # ПРЕОБРАЗОВАНИЕ ТИПОВ ДАННЫХ с более умной обработкой
    numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                       'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                       'TOTAL UNITS']
    
    for col in numeric_columns:
        if col in data.columns:
            # Улучшенная обработка строковых значений
            data[col] = data[col].astype(str).replace({
                ' -  ': np.nan, ' - ': np.nan, ' -': np.nan,
                ' ': np.nan, '': np.nan, '0': np.nan, '0.0': np.nan
            })
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Преобразование даты
    if 'SALE DATE' in data.columns:
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
    
    # Создание BOROUGH_NAME на раннем этапе
    if 'BOROUGH' in data.columns:
        borough_names = {
            1: 'Manhattan',
            2: 'Brooklyn', 
            3: 'Queens',
            4: 'Bronx',
            5: 'Staten Island'
        }
        data['BOROUGH_NAME'] = data['BOROUGH'].map(borough_names)
    
    # ИСПРАВЛЕННАЯ ОЧИСТКА: МЕНЕЕ АГРЕССИВНЫЙ ПОДХОД
    
    # 1. Очистка цен - сохраняем больше данных
    if 'SALE PRICE' in data.columns:
        # Удаляем только явно некорректные значения
        data = data[data['SALE PRICE'].notna()]
        
        # Вместо жесткой границы $10K используем процентили
        price_01 = data['SALE PRICE'].quantile(0.01)  # 1-й процентиль
        price_99 = data['SALE PRICE'].quantile(0.99)  # 99-й процентиль
        
        # Сохраняем 98% данных вместо удаления по жестким границам
        data = data[(data['SALE PRICE'] >= price_01) & 
                   (data['SALE PRICE'] <= price_99)]
        
        # Логарифмирование для работы с логнормальным распределением
        data['LOG_SALE_PRICE'] = np.log1p(data['SALE PRICE'])
    
    # 2. Очистка года постройки - более реалистичные границы
    if 'YEAR BUILT' in data.columns:
        current_year = datetime.now().year
        # Сохраняем здания с 1600 года (исторические здания Нью-Йорка)
        data = data[(data['YEAR BUILT'] >= 1600) & 
                   (data['YEAR BUILT'] <= current_year)]
        # Вместо удаления нулевых - заполняем медианой по району
        if data['YEAR BUILT'].isna().any():
            median_year_by_borough = data.groupby('BOROUGH_NAME')['YEAR BUILT'].median()
            data['YEAR BUILT'] = data.apply(
                lambda row: median_year_by_borough[row['BOROUGH_NAME']] 
                if pd.isna(row['YEAR BUILT']) else row['YEAR BUILT'],
                axis=1
            )
    
    # 3. Очистка площади - сохраняем больше вариативности
    for area_col in ['GROSS SQUARE FEET', 'LAND SQUARE FEET']:
        if area_col in data.columns:
            # Удаляем только отрицательные значения
            data = data[data[area_col] >= 0]
            # Вместо удаления больших значений - используем логарифм
            data[f'LOG_{area_col}'] = np.log1p(data[area_col].fillna(0))
    
    # 4. Импутация пропусков вместо удаления
    numeric_cols_for_imputation = ['GROSS SQUARE FEET', 'LAND SQUARE FEET', 
                                  'YEAR BUILT', 'TOTAL UNITS']
    
    for col in numeric_cols_for_imputation:
        if col in data.columns:
            # Заполняем медианой по району и типу здания
            if 'BOROUGH_NAME' in data.columns and 'BUILDING CLASS CATEGORY' in data.columns:
                data[col] = data.groupby(['BOROUGH_NAME', 'BUILDING CLASS CATEGORY'])[col]\
                               .transform(lambda x: x.fillna(x.median()))
            else:
                data[col] = data[col].fillna(data[col].median())
    
    # 5. Создаем производные признаки ДО фильтрации
    if all(col in data.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
        data['PRICE_PER_SQFT'] = data['SALE PRICE'] / data['GROSS SQUARE FEET'].replace(0, np.nan)
        # Очистка выбросов в цене за кв.фут
        if data['PRICE_PER_SQFT'].notna().any():
            pq1 = data['PRICE_PER_SQFT'].quantile(0.01)
            pq3 = data['PRICE_PER_SQFT'].quantile(0.99)
            data = data[(data['PRICE_PER_SQFT'] >= pq1) & 
                       (data['PRICE_PER_SQFT'] <= pq3) | 
                       (data['PRICE_PER_SQFT'].isna())]
    
    if 'YEAR BUILT' in data.columns:
        data['BUILDING_AGE'] = current_year - data['YEAR BUILT']
        data['IS_HISTORIC'] = (data['BUILDING_AGE'] > 100).astype(int)
    
    if all(col in data.columns for col in ['GROSS SQUARE FEET', 'TOTAL UNITS']):
        data['SQFT_PER_UNIT'] = data['GROSS SQUARE FEET'] / data['TOTAL UNITS'].replace(0, 1)
    
    # Удаляем дубликаты
    data = data.drop_duplicates(subset=['ADDRESS', 'SALE DATE', 'SALE PRICE'], keep='first')
    
    # Сохраняем статистику очистки
    final_rows = len(data)
    retention_rate = (final_rows / original_rows) * 100
    
    st.sidebar.info(f"""
    **📊 Статистика очистки:**
    - Исходных строк: {original_rows:,}
    - После очистки: {final_rows:,}
    - Сохранено: {retention_rate:.1f}% данных
    """)
    
    return data

# Загружаем данные
df = load_data()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["Визуализация данных", "Анализ рынка", "Прогнозные модели", "Таблица переводов"]
)

# Информация о данных в сайдбаре
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 О данных

**Что есть:**
- Продажи за 12 месяцев (2016-2017)
- 5 округов Нью-Йорка
- Характеристики объектов (площадь, район, год постройки)

**Ограничения:**
- Только 12 месяцев данных
- Нет долгосрочных трендов
- Нельзя предсказать рыночные циклы
""")

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
    price_min = int(df['SALE PRICE'].min())
    price_max = int(df['SALE PRICE'].max())
    
    price_range = st.sidebar.slider(
        "Цена продажи ($)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=10000
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

# Предупреждение о малом объеме данных
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Объем данных")

st.sidebar.write(f"**После фильтров:** {len(filtered_df):,} строк")

if len(filtered_df) < 1000:
    st.sidebar.warning(f"""
    ⚠️ **Мало данных для анализа**
    
    Для надежных результатов нужно:
    - ≥ 1,000 строк для описательного анализа
    - ≥ 5,000 строк для моделей машинного обучения
    
    **Рекомендации:** Ослабьте фильтры
    """)
elif len(filtered_df) < 5000:
    st.sidebar.info("✅ Достаточно для описательного анализа")
else:
    st.sidebar.success("✅ Достаточно для ML моделей")

# Создаем производные колонки для анализа
if 'SALE DATE' in filtered_df.columns:
    filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.month
    filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
    
if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
    filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET'].replace(0, np.nan)
    
if 'YEAR BUILT' in filtered_df.columns:
    filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']

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

# Страница 1: Визуализация данных
elif page == "Визуализация данных":
    st.title("Визуализация данных о продажах недвижимости Нью-Йорка")
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего записей", f"{len(filtered_df):,}")
    
    with col2:
        if 'SALE PRICE' in filtered_df.columns:
            median_price = filtered_df['SALE PRICE'].median()
            st.metric("Медианная цена", f"${median_price:,.0f}")
    
    with col3:
        if 'SALE DATE' in filtered_df.columns:
            unique_months = filtered_df['SALE_MONTH'].nunique()
            st.metric("Месяцев данных", unique_months)
    
    with col4:
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
        default=all_columns_russian[:10] if len(all_columns_russian) > 10 else all_columns_russian
    )
    
    # Преобразуем выбранные русские названия обратно в английские
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
    
    # Экспорт данных
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
    
    # Предупреждение о данных
    st.info("""
    **ℹ️ Важная информация о данных:**
    - Данные о продажах охватывают только 12 месяцев (2016-2017)
    - Анализ сезонности возможен только внутри года
    - Прогнозные модели имеют ограниченную точность из-за малого объема исторических данных
    """)
    
    # Визуализации
    col1, col2 = st.columns(2)
    
    with col1:
        if 'SALE PRICE' in filtered_df.columns:
            # Используем логарифмическую шкалу для цен
            fig = px.histogram(
                filtered_df, 
                x='LOG_SALE_PRICE' if 'LOG_SALE_PRICE' in filtered_df.columns else 'SALE PRICE',
                nbins=50,
                title="Распределение цен на недвижимость (лог. шкала)",
                labels={'LOG_SALE_PRICE': 'Логарифм цены', 'SALE PRICE': 'Цена ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        if 'YEAR BUILT' in filtered_df.columns:
            valid_year_data = filtered_df[filtered_df['Год постройки'] > 0]
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
        if 'BOROUGH_NAME' in filtered_df.columns:
            borough_counts = filtered_df['BOROUGH_NAME'].value_counts()
            fig = px.pie(
                values=borough_counts.values,
                names=borough_counts.index,
                title="Распределение продаж по округам",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
        if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
            # Фильтруем выбросы для лучшей визуализации
            scatter_data = filtered_df[
                (filtered_df['GROSS SQUARE FEET'] > 0) & 
                (filtered_df['GROSS SQUARE FEET'] < filtered_df['GROSS SQUARE FEET'].quantile(0.95)) &
                (filtered_df['SALE PRICE'] < filtered_df['SALE PRICE'].quantile(0.95))
            ]
            
            fig = px.scatter(
                scatter_data,
                x='GROSS SQUARE FEET',
                y='SALE PRICE',
                title="Цена vs Общая площадь (без выбросов)",
                labels={
                    'GROSS SQUARE FEET': 'Площадь (кв. фут)',
                    'SALE PRICE': 'Цена ($)'
                },
                opacity=0.6,
                trendline="ols"
            )
            fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
    
    # Сезонность внутри года
    st.markdown("---")
    st.subheader("Сезонные паттерны внутри года")
    
    if 'SALE_MONTH' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
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
    
    # Предупреждение о данных
    st.info(f"Анализ на основе {len(filtered_df):,} записей за 12 месяцев")
    
    if len(filtered_df) < 100:
        st.error("⚠️ Слишком мало данных для анализа. Увеличьте объем данных через фильтры.")
    else:
        analysis_type = st.selectbox(
            "Выберите тип анализа:",
            ["Анализ по районам", "Анализ по типам зданий", "Стоимость квадратного фута", "Возраст vs Цена"]
        )
        
        if analysis_type == "Анализ по районам":
            st.subheader("Сравнение районов")
            
            if 'NEIGHBORHOOD' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
                # Используем более надежные метрики
                neighborhood_stats = filtered_df.groupby('NEIGHBORHOOD').agg({
                    'SALE PRICE': ['median', 'count', lambda x: x.quantile(0.75) / x.quantile(0.25)],
                    'GROSS SQUARE FEET': 'median',
                    'PRICE_PER_SQFT': 'median'
                }).round(2)
                
                neighborhood_stats.columns = ['Медианная цена', 'Количество продаж', 'Коэф. вариации', 
                                            'Медианная площадь', 'Медианная цена за кв.фут']
                
                # Добавляем цену за кв.фут если еще не добавлено
                if 'Цена за кв.фут' not in neighborhood_stats.columns:
                    neighborhood_stats['Цена за кв.фут'] = neighborhood_stats['Медианная цена'] / neighborhood_stats['Медианная площадь']
                
                # Фильтруем районы с достаточным количеством данных
                neighborhood_stats = neighborhood_stats[neighborhood_stats['Количество продаж'] >= 10]
                
                if len(neighborhood_stats) > 0:
                    # Сортируем по медианной цене
                    top_neighborhoods = neighborhood_stats.sort_values('Медианная цена', ascending=False).head(15)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            top_neighborhoods.reset_index(),
                            x='NEIGHBORHOOD',
                            y='Медианная цена',
                            title='Топ-15 районов по медианной цене',
                            color='Медианная цена',
                            hover_data=['Количество продаж', 'Цена за кв.фут']
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
                            size_max=40,
                            log_x=True,
                            log_y=True
                        )
                        fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Недостаточно данных по районам. Увеличьте объем выборки.")
        
        elif analysis_type == "Анализ по типам зданий":
            st.subheader("Анализ по типам недвижимости")
            
            if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
                building_stats = filtered_df.groupby('BUILDING CLASS CATEGORY').agg({
                    'SALE PRICE': ['median', 'count', 'std'],
                    'GROSS SQUARE FEET': 'median',
                    'TOTAL UNITS': 'median',
                    'PRICE_PER_SQFT': 'median'
                }).round(2)
                
                building_stats.columns = ['Медианная цена', 'Количество', 'Стд. отклонение', 
                                        'Медианная площадь', 'Медианное кол-во единиц', 'Медианная цена за кв.фут']
                
                # Фильтруем типы с достаточным количеством данных
                building_stats = building_stats[building_stats['Количество'] >= 5]
                
                if len(building_stats) > 0:
                    # Топ-10 типов по цене
                    top_buildings = building_stats.nlargest(10, 'Медианная цена')
                    
                    fig = px.bar(
                        top_buildings.reset_index(),
                        x='BUILDING CLASS CATEGORY',
                        y='Медианная цена',
                        title='Топ-10 самых дорогих типов недвижимости',
                        color='Медианная цена',
                        hover_data=['Количество', 'Медианная цена за кв.фут']
                    )
                    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Недостаточно данных по типам зданий.")
        
        elif analysis_type == "Стоимость квадратного фута":
            st.subheader("Анализ стоимости квадратного фута")
            
            if 'PRICE_PER_SQFT' in filtered_df.columns:
                # Удаляем выбросы более мягко
                if filtered_df['PRICE_PER_SQFT'].notna().any():
                    q1 = filtered_df['PRICE_PER_SQFT'].quantile(0.05)
                    q3 = filtered_df['PRICE_PER_SQFT'].quantile(0.95)
                    price_per_sqft_filtered = filtered_df[
                        (filtered_df['PRICE_PER_SQFT'] >= q1) & 
                        (filtered_df['PRICE_PER_SQFT'] <= q3)
                    ]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            price_per_sqft_filtered,
                            x='PRICE_PER_SQFT',
                            nbins=50,
                            title="Распределение цены за кв.фут",
                            labels={'PRICE_PER_SQFT': 'Цена за кв.фут ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'BOROUGH_NAME' in filtered_df.columns:
                            borough_price_sqft = price_per_sqft_filtered.groupby('BOROUGH_NAME')['PRICE_PER_SQFT'].median()
                            borough_price_sqft = borough_price_sqft.sort_values(ascending=False)
                            
                            fig = px.bar(
                                x=borough_price_sqft.index,
                                y=borough_price_sqft.values,
                                title='Средняя цена за кв.фут по округам',
                                labels={'x': 'Округ', 'y': 'Цена за кв.фут ($)'},
                                color=borough_price_sqft.values
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Возраст vs Цена":
            st.subheader("Влияние возраста здания на цену")
            
            if 'BUILDING_AGE' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
                # Более логичные возрастные категории
                age_bins = [0, 5, 10, 20, 50, 100, 200, 500]
                age_labels = ['0-5 лет', '6-10 лет', '11-20 лет', '21-50 лет', '51-100 лет', '101-200 лет', '200+ лет']
                
                filtered_df['AGE_CATEGORY'] = pd.cut(
                    filtered_df['BUILDING_AGE'],
                    bins=age_bins,
                    labels=age_labels,
                    right=False
                )
                
                age_stats = filtered_df.groupby('AGE_CATEGORY').agg({
                    'SALE PRICE': ['median', 'count'],
                    'PRICE_PER_SQFT': 'median',
                    'GROSS SQUARE FEET': 'median'
                }).reset_index()
                
                age_stats.columns = ['Возрастная категория', 'Медианная цена', 'Количество', 
                                    'Медианная цена за кв.фут', 'Медианная площадь']
                
                # Фильтруем категории с достаточным количеством данных
                age_stats = age_stats[age_stats['Количество'] >= 5]
                
                if len(age_stats) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            age_stats,
                            x='Возрастная категория',
                            y='Медианная цена',
                            title='Медианная цена по возрастным категориям',
                            color='Медианная цена',
                            hover_data=['Количество', 'Медианная цена за кв.фут']
                        )
                        fig.update_layout(yaxis_tickformat=',')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Фильтруем выбросы для scatter plot
                        scatter_data = filtered_df[
                            (filtered_df['BUILDING_AGE'] < filtered_df['BUILDING_AGE'].quantile(0.99)) &
                            (filtered_df['SALE PRICE'] < filtered_df['SALE PRICE'].quantile(0.99))
                        ]
                        
                        fig = px.scatter(
                            scatter_data,
                            x='BUILDING_AGE',
                            y='SALE PRICE',
                            trendline="lowess",
                            title='Зависимость цены от возраста здания',
                            labels={'BUILDING_AGE': 'Возраст здания (лет)', 'SALE PRICE': 'Цена ($)'},
                            opacity=0.3
                        )
                        fig.update_layout(yaxis_tickformat=',')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Недостаточно данных для анализа возраста.")

# Страница 3: Прогнозные модели
elif page == "Прогнозные модели":
    st.title("Прогнозные модели на основе данных за 12 месяцев")
    
    # Предупреждение об ограничениях
    st.warning("""
    ⚠️ **Важные ограничения:**
    1. Данные охватывают только 12 месяцев
    2. Не учитываются долгосрочные тренды и макроэкономические факторы
    3. Точность моделей ограничена объемом данных
    
    **Модели могут предсказывать только:**
    - Относительную стоимость на основе характеристик
    - Сезонные паттерны внутри года
    - Приблизительные ценовые категории
    """)
    
    if len(filtered_df) < 1000:
        st.error(f"""
        ⚠️ **Недостаточно данных для построения моделей**
        
        Текущий объем: {len(filtered_df):,} строк
        Требуется минимум: 1,000 строк
        
        **Рекомендации:**
        1. Ослабьте фильтры в сайдбаре
        2. Выберите "Все" для района и типа здания
        3. Расширьте диапазоны года постройки и цены
        """)
    else:
        model_type = st.selectbox(
            "Выберите модель:",
            ["Прогноз цены на основе характеристик", "Анализ сезонности", "Классификация по ценовым категориям"]
        )
        
        # Модель 1: Прогноз цены на основе характеристик
        if model_type == "Прогноз цены на основе характеристик":
            st.subheader("🎯 Прогноз цены на основе характеристик объекта")
            
            if len(filtered_df) < 2000:
                st.warning(f"Рекомендуется ≥ 2,000 строк для надежной модели. Доступно: {len(filtered_df):,}")
            
            # Подготовка данных для модели
            st.write("**Подготовка данных...**")
            
            # УЛУЧШЕННЫЙ набор признаков
            features = [
                'GROSS SQUARE FEET', 
                'BOROUGH', 
                'YEAR BUILT',
                'BUILDING_AGE',  # Добавлен возраст здания
                'TOTAL UNITS', 
                'BUILDING CLASS CATEGORY', 
                'LAND SQUARE FEET',
                'NEIGHBORHOOD',  # Добавлен район - важный признак!
            ]
            
            # Оставляем только существующие колонки
            features = [f for f in features if f in filtered_df.columns]
            
            # Создаем копию данных для модели
            model_df = filtered_df.copy()
            
            # ИМПУТАЦИЯ вместо удаления пропусков
            st.write("**Обработка пропущенных значений...**")
            
            # Разделяем на числовые и категориальные признаки
            numeric_features = [f for f in features if model_df[f].dtype in [np.int64, np.float64]]
            categorical_features = [f for f in features if model_df[f].dtype == 'object']
            
            # Импутация числовых признаков
            if numeric_features:
                imputer = SimpleImputer(strategy='median')
                model_df[numeric_features] = imputer.fit_transform(model_df[numeric_features])
            
            # Импутация категориальных признаков
            for cat_feature in categorical_features:
                if cat_feature in model_df.columns:
                    # Заполняем самым частым значением
                    most_frequent = model_df[cat_feature].mode()
                    if not most_frequent.empty:
                        model_df[cat_feature] = model_df[cat_feature].fillna(most_frequent.iloc[0])
                    else:
                        model_df[cat_feature] = model_df[cat_feature].fillna('Unknown')
            
            # Убедимся, что целевая переменная без пропусков
            if 'SALE PRICE' in model_df.columns:
                model_df = model_df[model_df['SALE PRICE'].notna()]
            
            if len(model_df) < 100:
                st.error(f"Недостаточно данных после обработки: {len(model_df)} строк")
            else:
                # Используем логарифмированную цену для лучшей сходимости
                model_df['LOG_SALE_PRICE'] = np.log1p(model_df['SALE PRICE'])
                
                X = model_df[features].copy()
                y = model_df['LOG_SALE_PRICE']
                
                # Кодируем категориальные переменные
                categorical_cols = X.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    # Для NEIGHBORHOOD используем частотное кодирование вместо One-Hot
                    if 'NEIGHBORHOOD' in categorical_cols:
                        neighborhood_freq = X['NEIGHBORHOOD'].value_counts(normalize=True)
                        X['NEIGHBORHOOD_FREQ'] = X['NEIGHBORHOOD'].map(neighborhood_freq)
                        X = X.drop('NEIGHBORHOOD', axis=1)
                        categorical_cols = categorical_cols.drop('NEIGHBORHOOD')
                    
                    # One-Hot для остальных категориальных
                    if len(categorical_cols) > 0:
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                
                # Масштабирование признаков
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Разделяем данные
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Обучаем модель с настройкой гиперпараметров
                st.write("**Обучение модели Random Forest...**")
                
                # Используем GridSearch для оптимизации
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
                
                # Для скорости используем только если достаточно данных
                if len(X_train) > 1000:
                    grid_search = GridSearchCV(
                        model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    st.write(f"**Лучшие параметры:** {grid_search.best_params_}")
                else:
                    model.fit(X_train, y_train)
                
                # Прогноз и оценка
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                y_test_original = np.expm1(y_test)
                
                # Метрики
                mae = mean_absolute_error(y_test_original, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
                r2 = r2_score(y_test_original, y_pred)
                
                # Относительные ошибки
                median_price = model_df['SALE PRICE'].median()
                mae_relative = (mae / median_price) * 100
                rmse_relative = (rmse / median_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"${mae:,.0f}", f"{mae_relative:.1f}% от медианы")
                with col2:
                    st.metric("RMSE", f"${rmse:,.0f}", f"{rmse_relative:.1f}% от медианы")
                with col3:
                    st.metric("R²", f"{r2:.3f}")
                
                # Интерпретация метрик
                st.info(f"""
                **Интерпретация метрик:**
                - **R² = {r2:.3f}**: Модель объясняет {r2*100:.1f}% дисперсии цен
                - **MAE = {mae_relative:.1f}%**: Средняя ошибка ±{mae_relative:.1f}% от медианной цены
                - **RMSE = {rmse_relative:.1f}%**: Учитывает большие ошибки
                
                **Контекст:** Для годовых данных R² > 0.5 считается хорошим результатом
                """)
                
                # Визуализация предсказаний
                fig = go.Figure()
                
                # Показываем только 100 случайных точек для наглядности
                np.random.seed(42)
                if len(y_test_original) > 100:
                    indices = np.random.choice(len(y_test_original), 100, replace=False)
                    y_test_sample = y_test_original.iloc[indices]
                    y_pred_sample = y_pred[indices]
                else:
                    y_test_sample = y_test_original
                    y_pred_sample = y_pred
                
                fig.add_trace(go.Scatter(
                    x=y_test_sample,
                    y=y_pred_sample,
                    mode='markers',
                    name='Предсказания',
                    marker=dict(size=8, opacity=0.6, color='blue'),
                    hovertemplate='Реальная: $%{x:,.0f}<br>Предсказанная: $%{y:,.0f}<extra></extra>'
                ))
                
                # Линия идеального предсказания
                max_val = max(y_test_sample.max(), y_pred_sample.max())
                min_val = min(y_test_sample.min(), y_pred_sample.min())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Идеальное предсказание',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title='Сравнение реальных и предсказанных цен',
                    xaxis_title='Реальная цена ($)',
                    yaxis_title='Предсказанная цена ($)',
                    xaxis_tickformat=',',
                    yaxis_tickformat=',',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Важность признаков
                st.subheader("📊 Важность признаков для предсказания цены")
                
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
                        title='Топ-15 важнейших признаков для предсказания цены',
                        color='Важность'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Интерактивный прогноз
                st.markdown("---")
                st.subheader("🔮 Интерактивный прогноз цены")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sqft = st.number_input(
                        "Общая площадь (кв. фут)",
                        min_value=100,
                        max_value=100000,
                        value=1000,
                        step=100,
                        key='sqft_interactive'
                    )
                    
                    # Получаем уникальные районы
                    if 'NEIGHBORHOOD' in model_df.columns:
                        neighborhoods = sorted(model_df['NEIGHBORHOOD'].dropna().unique())
                        neighborhood = st.selectbox(
                            "Район",
                            options=neighborhoods,
                            key='neighborhood_interactive'
                        )
                    
                    borough = st.selectbox(
                        "Округ",
                        options=sorted(model_df['BOROUGH'].unique()),
                        format_func=lambda x: {
                            1: 'Манхэттен',
                            2: 'Бруклин',
                            3: 'Квинс',
                            4: 'Бронкс',
                            5: 'Стэтен-Айленд'
                        }.get(x, x),
                        key='borough_interactive'
                    )
                
                with col2:
                    year_built = st.number_input(
                        "Год постройки",
                        min_value=1700,
                        max_value=datetime.now().year,
                        value=1980,
                        step=1,
                        key='year_interactive'
                    )
                    
                    total_units = st.number_input(
                        "Количество единиц",
                        min_value=1,
                        max_value=1000,
                        value=1,
                        step=1,
                        key='units_interactive'
                    )
                
                with col3:
                    land_sqft = st.number_input(
                        "Площадь земли (кв. фут)",
                        min_value=100,
                        max_value=1000000,
                        value=sqft,
                        step=100,
                        key='land_interactive'
                    )
                    
                    if 'BUILDING CLASS CATEGORY' in model_df.columns:
                        building_types = sorted(model_df['BUILDING CLASS CATEGORY'].unique())
                        building_type = st.selectbox(
                            "Тип здания",
                            options=building_types,
                            key='building_interactive'
                        )
                
                # Кнопка для прогноза
                if st.button("Сделать прогноз", key='predict_button'):
                    try:
                        # Создаем DataFrame с введенными данными
                        input_data = pd.DataFrame({
                            'GROSS SQUARE FEET': [sqft],
                            'BOROUGH': [borough],
                            'YEAR BUILT': [year_built],
                            'BUILDING_AGE': [datetime.now().year - year_built],
                            'TOTAL UNITS': [total_units],
                            'LAND SQUARE FEET': [land_sqft],
                            'BUILDING CLASS CATEGORY': [building_type],
                            'NEIGHBORHOOD': [neighborhood]
                        })
                        
                        # Частотное кодирование района
                        if 'NEIGHBORHOOD' in input_data.columns and 'NEIGHBORHOOD' in model_df.columns:
                            neighborhood_freq = model_df['NEIGHBORHOOD'].value_counts(normalize=True)
                            input_data['NEIGHBORHOOD_FREQ'] = input_data['NEIGHBORHOOD'].map(neighborhood_freq).fillna(0)
                            input_data = input_data.drop('NEIGHBORHOOD', axis=1)
                        
                        # Применяем те же преобразования
                        input_processed = pd.get_dummies(input_data, drop_first=True)
                        
                        # Выравниваем столбцы с тренировочными данными
                        for col in X.columns:
                            if col not in input_processed.columns:
                                input_processed[col] = 0
                        
                        input_processed = input_processed[X.columns]
                        
                        # Масштабируем
                        input_scaled = scaler.transform(input_processed)
                        
                        # Делаем прогноз
                        predicted_price_log = model.predict(input_scaled)[0]
                        predicted_price = np.expm1(predicted_price_log)
                        price_per_sqft = predicted_price / sqft if sqft > 0 else 0
                        
                        # Оцениваем доверительный интервал
                        # Используем стандартное отклонение предсказаний деревьев
                        tree_predictions = []
                        for tree in model.estimators_:
                            tree_pred_log = tree.predict(input_scaled)[0]
                            tree_predictions.append(np.expm1(tree_pred_log))
                        
                        mean_prediction = np.mean(tree_predictions)
                        std_prediction = np.std(tree_predictions)
                        
                        st.success(f"""
                        **🏠 Прогнозируемая цена: ${predicted_price:,.0f}**
                        
                        **📊 Статистика прогноза:**
                        - Цена за кв.фут: **${price_per_sqft:.2f}**
                        - Общая площадь: {sqft:,.0f} кв.фут
                        - Возраст здания: {datetime.now().year - year_built} лет
                        - Тип: {building_type}
                        - Район: {neighborhood}
                        
                        **🎯 Доверительный интервал (80%):**
                        - От **${max(0, mean_prediction - std_prediction):,.0f}**
                        - До **${mean_prediction + std_prediction:,.0f}**
                        
                        *Примечание: Точность ограничена 12 месяцами данных*
                        """)
                        
                    except Exception as e:
                        st.error(f"Ошибка при прогнозе: {str(e)}")
                        st.info("Попробуйте изменить параметры объекта")
        
        # Модель 2: Анализ сезонности (остается без изменений, так как это не ML модель)
        elif model_type == "Анализ сезонности":
            st.subheader("📅 Анализ сезонных паттернов")
            
            if 'SALE_MONTH' not in filtered_df.columns:
                st.error("В данных отсутствует информация о дате продажи.")
            else:
                # Анализ сезонности по месяцам
                monthly_analysis = filtered_df.groupby('SALE_MONTH').agg({
                    'SALE PRICE': ['median', 'count', 'std'],
                    'PRICE_PER_SQFT': 'median',
                    'GROSS SQUARE FEET': 'median'
                }).reset_index()
                
                monthly_analysis.columns = ['Месяц', 'Медианная цена', 'Количество продаж', 
                                           'Стд. отклонение', 'Медианная цена за кв.фут', 
                                           'Медианная площадь']
                
                # Нормализуем данные для сравнения
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
                    height=700,
                    showlegend=True,
                    title_text="Анализ сезонности продаж недвижимости (2016-2017)"
                )
                
                fig.update_xaxes(title_text="Месяц", row=1, col=1)
                fig.update_xaxes(title_text="Месяц", row=2, col=1)
                fig.update_yaxes(title_text="Цена ($)", tickformat=',', row=1, col=1)
                fig.update_yaxes(title_text="Нормализованное значение", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Количество продаж", row=2, col=1)
                fig.update_yaxes(title_text="Нормализованное значение", row=2, col=1, secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Предупреждение о данных
                st.warning("""
                ⚠️ **Важное ограничение:**
                Данные охватывают только 12 месяцев (2016-2017). 
                Сезонные паттерны могут быть специфичными для этого конкретного года 
                и не отражать многолетние тренды.
                """)
        
        # Модель 3: Классификация по ценовым категориям
        elif model_type == "Классификация по ценовым категориям":
            st.subheader("🏷️ Классификация объектов по ценовым категориям")
            
            if 'SALE PRICE' not in filtered_df.columns:
                st.error("В данных отсутствует информация о цене продажи.")
            elif len(filtered_df) < 500:
                st.error(f"Недостаточно данных для классификации. Требуется ≥ 500 строк, доступно: {len(filtered_df):,}")
            else:
                # Создаем целевые категории с более разумными границами
                classification_df = filtered_df.copy()
                
                # Используем квартили для более сбалансированных категорий
                price_25 = classification_df['SALE PRICE'].quantile(0.25)
                price_50 = classification_df['SALE PRICE'].quantile(0.50)  # медиана
                price_75 = classification_df['SALE PRICE'].quantile(0.75)
                
                # Создаем 4 категории
                price_bins = [0, price_25, price_50, price_75, classification_df['SALE PRICE'].max()]
                price_labels = ['Бюджет', 'Стандарт', 'Премиум', 'Элитный']
                
                classification_df['PRICE_CATEGORY'] = pd.cut(
                    classification_df['SALE PRICE'],
                    bins=price_bins,
                    labels=price_labels,
                    include_lowest=True
                )
                
                st.info(f"""
                **Границы категорий:**
                - **Бюджет:** до ${price_25:,.0f}
                - **Стандарт:** ${price_25:,.0f} - ${price_50:,.0f}
                - **Премиум:** ${price_50:,.0f} - ${price_75:,.0f}
                - **Элитный:** от ${price_75:,.0f}
                
                *Категории основаны на квартилях распределения цен*
                """)
                
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
                        title='Распределение объектов по ценовым категориям',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Характеристики по категориям
                    category_stats = classification_df.groupby('PRICE_CATEGORY').agg({
                        'SALE PRICE': ['median', 'min', 'max', 'count'],
                        'GROSS SQUARE FEET': 'median',
                        'YEAR BUILT': 'median',
                        'TOTAL UNITS': 'median',
                        'PRICE_PER_SQFT': 'median'
                    }).round(2)
                    
                    category_stats.columns = ['Медианная цена', 'Минимальная цена', 'Максимальная цена', 'Количество',
                                             'Медианная площадь', 'Медианный год постройки', 
                                             'Медианное кол-во единиц', 'Медианная цена за кв.фут']
                    
                    st.write("**Характеристики по категориям:**")
                    st.dataframe(
                        category_stats.style.format({
                            'Медианная цена': '${:,.0f}',
                            'Минимальная цена': '${:,.0f}',
                            'Максимальная цена': '${:,.0f}',
                            'Количество': '{:,.0f}',
                            'Медианная площадь': '{:,.0f}',
                            'Медианный год постройки': '{:.0f}',
                            'Медианное кол-во единиц': '{:.1f}',
                            'Медианная цена за кв.фут': '${:.2f}'
                        }),
                        use_container_width=True,
                        height=300
                    )
                
                # Обучение модели классификации
                st.markdown("---")
                st.subheader("🤖 Обучение модели классификации")
                
                # Улучшенный набор признаков
                features_class = [
                    'GROSS SQUARE FEET',
                    'BOROUGH',
                    'YEAR BUILT',
                    'BUILDING_AGE',
                    'TOTAL UNITS',
                    'LAND SQUARE FEET',
                    'BUILDING CLASS CATEGORY',
                    'NEIGHBORHOOD',  # Важный признак!
                ]
                
                # Оставляем только существующие колонки
                features_class = [f for f in features_class if f in classification_df.columns]
                
                st.write(f"**Используется {len(features_class)} признаков**")
                
                # Подготовка данных
                X_class = classification_df[features_class].copy()
                y_class = classification_df['PRICE_CATEGORY_ENCODED']
                
                # Импутация пропусков
                numeric_features_class = [f for f in features_class if X_class[f].dtype in [np.int64, np.float64]]
                categorical_features_class = [f for f in features_class if X_class[f].dtype == 'object']
                
                if numeric_features_class:
                    imputer = SimpleImputer(strategy='median')
                    X_class[numeric_features_class] = imputer.fit_transform(X_class[numeric_features_class])
                
                for cat_feature in categorical_features_class:
                    if cat_feature in X_class.columns:
                        most_frequent = X_class[cat_feature].mode()
                        if not most_frequent.empty:
                            X_class[cat_feature] = X_class[cat_feature].fillna(most_frequent.iloc[0])
                        else:
                            X_class[cat_feature] = X_class[cat_feature].fillna('Unknown')
                
                if len(X_class) < 100:
                    st.error(f"Недостаточно данных после обработки: {len(X_class)} строк")
                else:
                    # Кодирование категориальных переменных
                    categorical_cols_class = X_class.select_dtypes(include=['object']).columns
                    
                    # Частотное кодирование для NEIGHBORHOOD
                    if 'NEIGHBORHOOD' in categorical_cols_class:
                        neighborhood_freq = X_class['NEIGHBORHOOD'].value_counts(normalize=True)
                        X_class['NEIGHBORHOOD_FREQ'] = X_class['NEIGHBORHOOD'].map(neighborhood_freq)
                        X_class = X_class.drop('NEIGHBORHOOD', axis=1)
                        categorical_cols_class = categorical_cols_class.drop('NEIGHBORHOOD')
                    
                    # One-Hot Encoding для остальных категориальных
                    if len(categorical_cols_class) > 0:
                        X_class_encoded = pd.get_dummies(X_class, columns=categorical_cols_class, drop_first=True)
                    else:
                        X_class_encoded = X_class.copy()
                    
                    # Масштабирование
                    scaler_class = StandardScaler()
                    X_class_scaled = scaler_class.fit_transform(X_class_encoded)
                    
                    # Разделяем данные с стратификацией
                    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                        X_class_scaled, y_class, 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=y_class
                    )
                    
                    st.write(f"**Размер данных:** Обучающая выборка: {len(X_train_class):,}, Тестовая: {len(X_test_class):,}")
                    
                    # Обучение модели
                    st.write("**Обучение модели Random Forest Classifier...**")
                    
                    model_class = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    )
                    
                    model_class.fit(X_train_class, y_train_class)
                    
                    # Оценка модели
                    y_pred_class = model_class.predict(X_test_class)
                    y_pred_proba = model_class.predict_proba(X_test_class)
                    
                    # Метрики
                    accuracy = accuracy_score(y_test_class, y_pred_class)
                    precision = precision_score(y_test_class, y_pred_class, average='weighted')
                    recall = recall_score(y_test_class, y_pred_class, average='weighted')
                    f1 = f1_score(y_test_class, y_pred_class, average='weighted')
                    
                    st.subheader("📊 Оценка модели классификации")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Precision", f"{precision:.3f}")
                    with col3:
                        st.metric("Recall", f"{recall:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.3f}")
                    
                    # Интерпретация метрик
                    baseline_accuracy = category_counts.max() / len(classification_df)
                    
                    st.info(f"""
                    **Интерпретация метрик:**
                    - **Accuracy = {accuracy:.3f}**: Модель правильно классифицирует {accuracy*100:.1f}% объектов
                    - **Baseline = {baseline_accuracy:.3f}**: Точность наивного классификатора (всегда выбирать самую частую категорию)
                    - **Улучшение = {(accuracy - baseline_accuracy)*100:.1f}%**: На сколько модель лучше наивного подхода
                    - **F1-Score = {f1:.3f}**: Баланс между точностью и полнотой
                    
                    **Для 4-классовой классификации с годовыми данными это хороший результат**
                    """)
                    
                    # Матрица ошибок
                    st.subheader("📊 Матрица ошибок")
                    
                    cm = confusion_matrix(y_test_class, y_pred_class)
                    cm_df = pd.DataFrame(
                        cm,
                        index=le.classes_,
                        columns=le.classes_
                    )
                    
                    fig = px.imshow(
                        cm_df,
                        text_auto=True,
                        aspect="auto",
                        title="Матрица ошибок классификации",
                        color_continuous_scale='Blues',
                        labels=dict(x="Предсказанный класс", y="Истинный класс", color="Количество")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Анализ ошибок
                    error_mask = y_test_class != y_pred_class
                    if error_mask.any():
                        error_rate = error_mask.sum() / len(y_test_class)
                        st.write(f"**Общая ошибка классификации:** {error_rate:.1%}")
                        
                        # Наиболее частые ошибки
                        error_pairs = []
                        for i in range(len(y_test_class)):
                            if error_mask[i]:
                                true_label = le.inverse_transform([y_test_class.iloc[i]])[0]
                                pred_label = le.inverse_transform([y_pred_class[i]])[0]
                                error_pairs.append((true_label, pred_label))
                        
                        if error_pairs:
                            from collections import Counter
                            common_errors = Counter(error_pairs).most_common(5)
                            
                            st.write("**Наиболее частые ошибки:**")
                            for (true, pred), count in common_errors:
                                st.write(f"- {true} → {pred}: {count} случаев")
                    
                    # Важность признаков
                    st.subheader("📈 Важность признаков для классификации")
                    
                    if hasattr(model_class, 'feature_importances_'):
                        feature_importance_class = pd.DataFrame({
                            'Признак': X_class_encoded.columns,
                            'Важность': model_class.feature_importances_
                        }).sort_values('Важность', ascending=False).head(15)
                        
                        fig = px.bar(
                            feature_importance_class,
                            x='Важность',
                            y='Признак',
                            orientation='h',
                            title='Топ-15 важнейших признаков для классификации',
                            color='Важность'
                        )
                        st.plotly_chart(fig, use_container_width=True)

# Добавляем информацию о проекте в футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>📊 NYC Property Sales Dashboard | Анализ данных за 12 месяцев (2016-2017)</p>
    <p>⚠️ Ограничения: 12 месяцев данных недостаточно для надежных долгосрочных прогнозов</p>
</div>
""", unsafe_allow_html=True)
