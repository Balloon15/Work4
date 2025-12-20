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

# Словарь переводов названий колонок на русский (ТОЛЬКО для таблицы переводов)
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

# Функция для перевода названий колонок (используется только в Таблице переводов)
def translate_columns(df):
    translated_cols = []
    for col in df.columns:
        translated_cols.append(COLUMN_TRANSLATIONS.get(col, col))
    df.columns = translated_cols
    return df

# Функция для обратного перевода (используется только в Таблице переводов)
def reverse_translate_column(russian_name):
    for eng, rus in COLUMN_TRANSLATIONS.items():
        if rus == russian_name:
            return eng
    return russian_name

# Загрузка данных с очисткой выбросов
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
    
    # ОЧИСТКА ВЫБРОСОВ В ЦЕНАХ
    if 'SALE PRICE' in data.columns:
        # 1. Удаляем нулевые и отрицательные цены
        data = data[data['SALE PRICE'] > 0]
        
        # 2. Удаляем слишком низкие цены (< $10,000) - вероятно, опечатки
        data = data[data['SALE PRICE'] >= 1000]
        
        # 3. Удаляем экстремально высокие цены (> $500 миллионов)
        data = data[data['SALE PRICE'] <= 500_000_000]
        
        # 4. Статистическая очистка (IQR метод)
        q1 = data['SALE PRICE'].quantile(0.25)
        q3 = data['SALE PRICE'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        data = data[data['SALE PRICE'] <= upper_bound]
    
    # Очистка года постройки - РЕАЛИСТИЧНЫЕ границы
    if 'YEAR BUILT' in data.columns:
        # Удаляем нереалистично старые годы (до 1700) и будущие годы
        current_year = datetime.now().year
        data = data[(data['YEAR BUILT'] >= 1700) & (data['YEAR BUILT'] <= current_year)]
        # Удаляем нулевые и отрицательные значения
        data = data[data['YEAR BUILT'] > 0]
    
    # Очистка площади
    if 'GROSS SQUARE FEET' in data.columns:
        data = data[(data['GROSS SQUARE FEET'] > 0) & (data['GROSS SQUARE FEET'] <= 1000000)]
    
    return data

# Загружаем данные
df = load_data()

# Создаем навигацию
st.sidebar.title("NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Data Visualization", "Market Analysis", "Predictive Models", "Translation Table"]
)
# Добавляем фильтры в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("Data Filters")

# Фильтр по району - используем оригинальные английские названия
neighborhoods = ['All'] + sorted(df['NEIGHBORHOOD'].dropna().unique().tolist())
selected_neighborhood = st.sidebar.selectbox(
    'Neighborhood', 
    neighborhoods
)

# Фильтр по типу здания - используем оригинальные английские названия
building_classes = ['All'] + sorted(df['BUILDING CLASS CATEGORY'].dropna().unique().tolist())
selected_building_class = st.sidebar.selectbox(
    'Building Class Category', 
    building_classes
)

# Фильтр по году постройки (реалистичные границы)
if 'YEAR BUILT' in df.columns:
    valid_years = df[df['YEAR BUILT'] > 0]['YEAR BUILT']
    
    if not valid_years.empty:
        min_year = int(max(valid_years.min(), 1700))  # Не ранее 1700 года
        max_year = int(min(valid_years.max(), datetime.now().year))  # Не позже текущего года
        
        year_range = st.sidebar.slider(
            "Year Built",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (1800, 2023)

# Фильтр по цене с реалистичными границами
if 'SALE PRICE' in df.columns:
    realistic_min_price = 1000
    realistic_max_price = 50_000_000
    
    price_range = st.sidebar.slider(
        "Sale Price ($)",
        min_value=int(realistic_min_price),
        max_value=int(realistic_max_price),
        value=(int(realistic_min_price), int(realistic_max_price)),
        step=1000
    )

# Применяем фильтры
filtered_df = df.copy()

if selected_neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['NEIGHBORHOOD'] == selected_neighborhood]

if selected_building_class != 'All':
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

# Создаем производные колонки
if 'SALE DATE' in filtered_df.columns:
    filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.month
    filtered_df['SALE_YEAR'] = filtered_df['SALE DATE'].dt.year
    
if all(col in filtered_df.columns for col in ['SALE PRICE', 'GROSS SQUARE FEET']):
    filtered_df['PRICE_PER_SQFT'] = filtered_df['SALE PRICE'] / filtered_df['GROSS SQUARE FEET']
    
if 'YEAR BUILT' in filtered_df.columns:
    filtered_df['BUILDING_AGE'] = datetime.now().year - filtered_df['YEAR BUILT']

# Страница 4: Таблица переводов
if page == "Translation Table":
    st.title("Column Names Translation Table")
    
    translation_table = pd.DataFrame({
        'Original Name (English)': list(COLUMN_TRANSLATIONS.keys()),
        'Translation (Russian)': list(COLUMN_TRANSLATIONS.values())
    })
    
    st.dataframe(
        translation_table,
        use_container_width=True,
        height=600
    )
    
    st.markdown("---")

# Страница 1: Визуализация данных
elif page == "Data Visualization":
    st.title("NYC Property Sales Data Visualization")
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(filtered_df))
    
    with col2:
        if 'SALE PRICE' in filtered_df.columns:
            median_price = filtered_df['SALE PRICE'].median()
            st.metric("Median Price ($)", f"{median_price:,.0f}")
    
    with col3:
        if 'SALE DATE' in filtered_df.columns:
            unique_months = filtered_df['SALE_MONTH'].nunique()
            st.metric("Months of Data", unique_months)
    
    with col4:
        unique_neighborhoods = filtered_df['NEIGHBORHOOD'].nunique()
        st.metric("Number of Neighborhoods", unique_neighborhoods)

    st.markdown("---")
    
    # Таблица с данными
    st.subheader("Data Preview")
    
    # Выбор колонок для отображения - используем оригинальные английские названия
    all_columns = filtered_df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display:",
        all_columns,
        default=all_columns[:10] if len(all_columns) > 10 else all_columns
    )
    
    # Пагинация
    page_size = st.selectbox("Rows per page:", [10, 25, 50, 100])
    page_number = st.number_input("Page number:", min_value=1, value=1)
    
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    
    if selected_columns:
        # Отображаем таблицу с английскими названиями колонок
        display_df = filtered_df[selected_columns].iloc[start_idx:end_idx]
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    # Экспорт данных
    if selected_columns:
        export_df = filtered_df[selected_columns]
    else:
        export_df = filtered_df
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_nyc_property_sales.csv",
        mime="text/csv",
    )
    
    st.markdown("---")
    
    # Базовые статистики
    st.subheader("Basic Statistics")
    
    if st.checkbox("Show statistics for numeric columns"):
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            stats_df = filtered_df[numeric_cols].describe().T
            stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df.columns = ['Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max']
            
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
                title="Property Price Distribution",
                labels={'SALE PRICE': 'Sale Price ($)'}
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
                    title="Year Built Distribution",
                    labels={'YEAR BUILT': 'Year'}
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
                title="Sales Distribution by Borough",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
        if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
            fig = px.scatter(
                filtered_df,
                x='GROSS SQUARE FEET',
                y='SALE PRICE',
                title="Price vs Gross Square Feet",
                labels={
                    'GROSS SQUARE FEET': 'Area (sq ft)',
                    'SALE PRICE': 'Price ($)'
                },
                opacity=0.6
            )
            fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
    
    # Сезонность внутри года
    st.markdown("---")
    st.subheader("Seasonal Patterns Within Year")
    
    if 'SALE_MONTH' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
        monthly_stats = filtered_df.groupby('SALE_MONTH').agg({
            'SALE PRICE': ['median', 'count'],
            'GROSS SQUARE FEET': 'median'
        }).reset_index()
        
        monthly_stats.columns = ['Month', 'Median Price', 'Sales Count', 'Median Area']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                monthly_stats,
                x='Month',
                y='Sales Count',
                title='Sales Count by Month',
                color='Sales Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                monthly_stats,
                x='Month',
                y='Median Price',
                title='Median Price by Month',
                markers=True
            )
            fig.update_layout(yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)

# Страница 2: Анализ рынка
elif page == "Market Analysis":
    st.title("NYC Real Estate Market Analysis")
    
    analysis_type = st.selectbox(
        "Select analysis type:",
        ["Neighborhood Analysis", "Building Type Analysis", "Price per Sq Ft Analysis", "Age vs Price Analysis"]
    )
    
    if analysis_type == "Neighborhood Analysis":
        st.subheader("Neighborhood Comparison")
        
        if 'NEIGHBORHOOD' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
            # Top 15 neighborhoods by median price
            neighborhood_stats = filtered_df.groupby('NEIGHBORHOOD').agg({
                'SALE PRICE': ['median', 'count'],
                'GROSS SQUARE FEET': 'median'
            }).round(2)
            
            neighborhood_stats.columns = ['Median Price', 'Sales Count', 'Median Area']
            
            # Add price per sq ft
            neighborhood_stats['Price per Sq Ft'] = neighborhood_stats['Median Price'] / neighborhood_stats['Median Area']
            
            # Sort by median price
            top_neighborhoods = neighborhood_stats.sort_values('Median Price', ascending=False).head(15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    top_neighborhoods.reset_index(),
                    x='NEIGHBORHOOD',
                    y='Median Price',
                    title='Top 15 Neighborhoods by Median Price',
                    color='Median Price'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    neighborhood_stats.reset_index(),
                    x='Sales Count',
                    y='Median Price',
                    size='Sales Count',
                    color='Price per Sq Ft',
                    hover_name='NEIGHBORHOOD',
                    title='Price vs Sales Count Correlation',
                    size_max=40
                )
                fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Building Type Analysis":
        st.subheader("Building Type Analysis")
        
        if 'BUILDING CLASS CATEGORY' in filtered_df.columns:
            building_stats = filtered_df.groupby('BUILDING CLASS CATEGORY').agg({
                'SALE PRICE': ['median', 'count', 'std'],
                'GROSS SQUARE FEET': 'median',
                'TOTAL UNITS': 'median'
            }).round(2)
            
            building_stats.columns = ['Median Price', 'Count', 'Std Deviation', 
                                      'Median Area', 'Median Units']
            
            # Top 10 types by price
            top_buildings = building_stats.nlargest(10, 'Median Price')
            
            fig = px.bar(
                top_buildings.reset_index(),
                x='BUILDING CLASS CATEGORY',
                y='Median Price',
                title='Top 10 Most Expensive Building Types',
                color='Median Price'
            )
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Price per Sq Ft Analysis":
        st.subheader("Price per Square Foot Analysis")
        
        if 'PRICE_PER_SQFT' in filtered_df.columns:
            # Remove outliers in price per sq ft
            q1 = filtered_df['PRICE_PER_SQFT'].quantile(0.01)
            q3 = filtered_df['PRICE_PER_SQFT'].quantile(0.99)
            price_per_sqft_filtered = filtered_df[(filtered_df['PRICE_PER_SQFT'] >= q1) & 
                                                 (filtered_df['PRICE_PER_SQFT'] <= q3)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    price_per_sqft_filtered,
                    x='PRICE_PER_SQFT',
                    nbins=50,
                    title="Price per Sq Ft Distribution",
                    labels={'PRICE_PER_SQFT': 'Price per Sq Ft ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'BOROUGH' in filtered_df.columns:
                    borough_map = {
                        1: 'Manhattan',
                        2: 'Brooklyn', 
                        3: 'Queens',
                        4: 'Bronx',
                        5: 'Staten Island'
                    }
                    
                    temp_df = price_per_sqft_filtered.copy()
                    temp_df['BOROUGH_NAME_TEMP'] = temp_df['BOROUGH'].map(borough_map)
                    
                    borough_price_sqft = temp_df.groupby('BOROUGH_NAME_TEMP')['PRICE_PER_SQFT'].median().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=borough_price_sqft.index,
                        y=borough_price_sqft.values,
                        title='Average Price per Sq Ft by Borough',
                        labels={'x': 'Borough', 'y': 'Price per Sq Ft ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Age vs Price Analysis":
        st.subheader("Building Age vs Price Analysis")
        
        if 'BUILDING_AGE' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
            # Group by age categories
            age_bins = [0, 10, 25, 50, 100, 200, 500]
            age_labels = ['0-10 years', '11-25 years', '26-50 years', '51-100 years', '101-200 years', '200+ years']
            
            filtered_df['AGE_CATEGORY'] = pd.cut(
                filtered_df['BUILDING_AGE'],
                bins=age_bins,
                labels=age_labels,
                right=False
            )
            
            age_stats = filtered_df.groupby('AGE_CATEGORY').agg({
                'SALE PRICE': 'median',
                'PRICE_PER_SQFT': 'median',
                'GROSS SQUARE FEET': 'median'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    age_stats,
                    x='AGE_CATEGORY',
                    y='SALE PRICE',
                    title='Median Price by Age Category',
                    color='SALE PRICE'
                )
                fig.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    filtered_df,
                    x='BUILDING_AGE',
                    y='SALE PRICE',
                    trendline="lowess",
                    title='Price vs Building Age',
                    labels={'BUILDING_AGE': 'Building Age (years)', 'SALE PRICE': 'Price ($)'},
                    opacity=0.3
                )
                fig.update_layout(yaxis_tickformat=',')
                st.plotly_chart(fig, use_container_width=True)

# Страница 3: Прогнозные модели
elif page == "Predictive Models":
    st.title("Predictive Models Based on 12 Months Data")    
    
    model_type = st.selectbox(
        "Select model type:",
        ["Price Prediction Based on Features", "Seasonality Analysis", "Price Category Classification"]
    )
    
    # Модель 1: Прогноз цены на основе характеристик
    if model_type == "Price Prediction Based on Features":
        st.subheader("Property Price Prediction Based on Features")
        
        if len(filtered_df) < 100:
            st.error("Too little data to build a model. Filter less data.")
        else:
            # Подготовка данных для модели
            st.write("**Preparing data...**")
            
            # Выбираем релевантные признаки
            features = ['GROSS SQUARE FEET', 'BOROUGH', 'YEAR BUILT', 
                       'TOTAL UNITS', 'BUILDING CLASS CATEGORY', 'LAND SQUARE FEET']
            
            # Создаем копию данных для модели
            model_df = filtered_df.copy()
            
            # Удаляем пропуски
            for feature in features + ['SALE PRICE']:
                if feature in model_df.columns:
                    model_df = model_df.dropna(subset=[feature])
            
            if len(model_df) < 50:
                st.error("Not enough data after cleaning missing values.")
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
                st.write("**Training Random Forest model...**")
                model = RandomForestRegressor(
                    n_estimators=100,
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
                    st.metric("MAE (Mean Absolute Error)", f"${mae:,.0f}")
                with col2:
                    st.metric("RMSE (Root Mean Square Error)", f"${rmse:,.0f}")
                with col3:
                    st.metric("R² (Coefficient of Determination)", f"{r2:.3f}")
                
                # Визуализация предсказаний
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=y_test.values[:100],
                    y=y_pred[:100],
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                # Линия идеального предсказания
                max_val = max(y_test.max(), y_pred.max())
                min_val = min(y_test.min(), y_pred.min())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Actual vs Predicted Prices (first 100 samples)',
                    xaxis_title='Actual Price ($)',
                    yaxis_title='Predicted Price ($)',
                    xaxis_tickformat=',',
                    yaxis_tickformat=','
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Важность признаков
                st.subheader("Feature Importance for Price Prediction")
                
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features for Price Prediction',
                        color='Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Интерактивный прогноз
                st.markdown("---")
                st.subheader("Interactive Price Prediction")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sqft = st.number_input(
                        "Gross Square Feet",
                        min_value=100,
                        max_value=100000,
                        value=1000,
                        step=100
                    )
                    
                    borough = st.selectbox(
                        "Borough",
                        options=sorted(model_df['BOROUGH'].unique()),
                        format_func=lambda x: {
                            1: 'Manhattan',
                            2: 'Brooklyn',
                            3: 'Queens',
                            4: 'Bronx',
                            5: 'Staten Island'
                        }.get(x, x)
                    )
                
                with col2:
                    year_built = st.number_input(
                        "Year Built",
                        min_value=1700,
                        max_value=datetime.now().year,
                        value=1980,
                        step=1
                    )
                    
                    total_units = st.number_input(
                        "Total Units",
                        min_value=1,
                        max_value=1000,
                        value=1,
                        step=1
                    )
                
                with col3:
                    land_sqft = st.number_input(
                        "Land Square Feet",
                        min_value=100,
                        max_value=1000000,
                        value=sqft,
                        step=100
                    )
                    
                    # Получаем уникальные типы зданий
                    if 'BUILDING CLASS CATEGORY' in model_df.columns:
                        building_types = sorted(model_df['BUILDING CLASS CATEGORY'].unique())
                        building_type = st.selectbox(
                            "Building Type",
                            options=building_types
                        )
                
                # Кнопка для прогноза
                if st.button("Make Prediction"):
                    # Создаем DataFrame с введенными данными
                    input_data = pd.DataFrame({
                        'GROSS SQUARE FEET': [sqft],
                        'BOROUGH': [borough],
                        'YEAR BUILT': [year_built],
                        'TOTAL UNITS': [total_units],
                        'LAND SQUARE FEET': [land_sqft],
                        'BUILDING CLASS CATEGORY': [building_type]
                    })
                    
                    # Применяем те же преобразования
                    input_processed = pd.get_dummies(input_data, drop_first=True)
                    
                    # Выравниваем столбцы с тренировочными данными
                    for col in X.columns:
                        if col not in input_processed.columns:
                            input_processed[col] = 0
                    
                    input_processed = input_processed[X.columns]
                    
                    # Делаем прогноз
                    predicted_price = model.predict(input_processed)[0]
                    price_per_sqft = predicted_price / sqft if sqft > 0 else 0
                    
                    st.success(f"""
                    **Predicted Price: ${predicted_price:,.0f}**
                    
                    Details:
                    - Price per sq ft: ${price_per_sqft:.2f}
                    - Gross area: {sqft:,.0f} sq ft
                    - Building age: {datetime.now().year - year_built} years
                    - Type: {building_type}
                    """)
    
    # Модель 2: Анализ сезонности
    elif model_type == "Seasonality Analysis":
        st.subheader("Seasonal Pattern Analysis")
        
        if 'SALE_MONTH' not in filtered_df.columns:
            st.error("Data does not contain sale date information.")
        else:
            # Анализ сезонности по месяцам
            monthly_analysis = filtered_df.groupby('SALE_MONTH').agg({
                'SALE PRICE': ['median', 'count', 'std'],
                'PRICE_PER_SQFT': 'median',
                'GROSS SQUARE FEET': 'median'
            }).reset_index()
            
            monthly_analysis.columns = ['Month', 'Median Price', 'Sales Count', 
                                       'Std Deviation', 'Median Price per Sq Ft', 
                                       'Median Area']
            
            # Нормализуем данные для сравнения
            monthly_analysis['Normalized Price'] = monthly_analysis['Median Price'] / monthly_analysis['Median Price'].mean()
            monthly_analysis['Normalized Count'] = monthly_analysis['Sales Count'] / monthly_analysis['Sales Count'].mean()
            
            # Визуализация сезонности
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Seasonality', 'Sales Volume Seasonality'),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Bar(
                    x=monthly_analysis['Month'],
                    y=monthly_analysis['Median Price'],
                    name='Median Price',
                    marker_color='royalblue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_analysis['Month'],
                    y=monthly_analysis['Normalized Price'],
                    name='Normalized Price',
                    line=dict(color='red', width=3),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=monthly_analysis['Month'],
                    y=monthly_analysis['Sales Count'],
                    name='Sales Count',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_analysis['Month'],
                    y=monthly_analysis['Normalized Count'],
                    name='Normalized Count',
                    line=dict(color='orange', width=3),
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                title_text="Real Estate Sales Seasonality Analysis"
            )
            
            fig.update_xaxes(title_text="Month", row=1, col=1)
            fig.update_xaxes(title_text="Month", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", tickformat=',', row=1, col=1)
            fig.update_yaxes(title_text="Normalized Value", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Sales Count", row=2, col=1)
            fig.update_yaxes(title_text="Normalized Value", row=2, col=1, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Статистический анализ сезонности
            st.subheader("📊 Seasonality Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Самый дорогой месяц
                most_expensive_month = monthly_analysis.loc[monthly_analysis['Median Price'].idxmax()]
                st.metric(
                    "Most Expensive Month",
                    f"Month {int(most_expensive_month['Month'])}",
                    f"${most_expensive_month['Median Price']:,.0f}"
                )
                
                # Месяц с наибольшим количеством продаж
                busiest_month = monthly_analysis.loc[monthly_analysis['Sales Count'].idxmax()]
                st.metric(
                    "Busiest Month",
                    f"Month {int(busiest_month['Month'])}",
                    f"{int(busiest_month['Sales Count'])} sales"
                )
            
            with col2:
                # Самый дешевый месяц
                cheapest_month = monthly_analysis.loc[monthly_analysis['Median Price'].idxmin()]
                st.metric(
                    "Cheapest Month",
                    f"Month {int(cheapest_month['Month'])}",
                    f"${cheapest_month['Median Price']:,.0f}"
                )
                
                # Амплитуда цен
                price_amplitude = ((most_expensive_month['Median Price'] - cheapest_month['Median Price']) / 
                                  cheapest_month['Median Price'] * 100)
                st.metric(
                    "Seasonal Price Amplitude",
                    f"{price_amplitude:.1f}%",
                    f"from ${cheapest_month['Median Price']:,.0f} to ${most_expensive_month['Median Price']:,.0f}"
                )
            
            # Рекомендации по сезонности
            st.markdown("---")
            st.subheader("Seasonality Recommendations")
            
            recommendations = []
            
            if most_expensive_month['Month'] in [5, 6, 7]:  # Весна/лето
                recommendations.append("**Price peak** occurs in spring/summer months")
            elif most_expensive_month['Month'] in [11, 12, 1]:  # Зима
                recommendations.append("**High prices** observed in winter months")
            
            if cheapest_month['Month'] in [9, 10]:  # Осень
                recommendations.append("**Best time to buy** is autumn months")
            
            if busiest_month['Sales Count'] > monthly_analysis['Sales Count'].mean() * 1.3:
                recommendations.append("**Peak market activity** in certain months")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            
            # Детальная таблица
            st.markdown("---")
            st.subheader("Detailed Monthly Statistics")
            
            display_stats = monthly_analysis.copy()
            display_stats['Price per Sq Ft'] = display_stats['Median Price'] / display_stats['Median Area']
            
            st.dataframe(
                display_stats.style.format({
                    'Month': '{:.0f}',
                    'Median Price': '${:,.0f}',
                    'Sales Count': '{:,.0f}',
                    'Std Deviation': '${:,.0f}',
                    'Median Price per Sq Ft': '${:.2f}',
                    'Median Area': '{:,.0f}',
                    'Normalized Price': '{:.3f}',
                    'Normalized Count': '{:.3f}',
                    'Price per Sq Ft': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )
    
    # Модель 3: Классификация по ценовым категориям
    elif model_type == "Price Category Classification":
        st.subheader("Property Price Category Classification")
        
        if 'SALE PRICE' not in filtered_df.columns:
            st.error("Data does not contain sale price information.")
        else:
            # Создаем целевые категории
            classification_df = filtered_df.copy()
            
            # Определяем границы категорий
            price_33 = classification_df['SALE PRICE'].quantile(0.33)
            price_66 = classification_df['SALE PRICE'].quantile(0.66)
            
            classification_df['PRICE_CATEGORY'] = pd.cut(
                classification_df['SALE PRICE'],
                bins=[0, price_33, price_66, classification_df['SALE PRICE'].max()],
                labels=['Cheap', 'Medium', 'Expensive']
            )
            
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
                    title='Property Distribution by Price Category',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Характеристики по категориям
                category_stats = classification_df.groupby('PRICE_CATEGORY').agg({
                    'SALE PRICE': ['median', 'min', 'max'],
                    'GROSS SQUARE FEET': 'median',
                    'YEAR BUILT': 'median',
                    'TOTAL UNITS': 'median'
                }).round(2)
                
                category_stats.columns = ['Median Price', 'Min Price', 'Max Price',
                                         'Median Area', 'Median Year Built', 'Median Units']
                
                category_stats['Price per Sq Ft'] = category_stats['Median Price'] / category_stats['Median Area']
                
                st.write("**Category Characteristics:**")
                st.dataframe(
                    category_stats.style.format({
                        'Median Price': '${:,.0f}',
                        'Min Price': '${:,.0f}',
                        'Max Price': '${:,.0f}',
                        'Median Area': '{:,.0f}',
                        'Median Year Built': '{:.0f}',
                        'Median Units': '{:.1f}',
                        'Price per Sq Ft': '${:.2f}'
                    }),
                    use_container_width=True
                )
            
            # Обучение модели классификации
            st.markdown("---")
            st.subheader("Classification Model")
            
            # Выбираем признаки
            features_class = ['GROSS SQUARE FEET', 'BOROUGH', 'YEAR BUILT', 
                            'TOTAL UNITS', 'LAND SQUARE FEET', 'BUILDING CLASS CATEGORY']
            
            # Подготовка данных
            X_class = classification_df[features_class].copy()
            y_class = classification_df['PRICE_CATEGORY_ENCODED']
            
            # Удаляем пропуски
            X_class = X_class.dropna()
            y_class = y_class[X_class.index]
            
            if len(X_class) < 50:
                st.error("Not enough data for classification model training.")
            else:
                # Кодируем категориальные переменные
                categorical_cols_class = X_class.select_dtypes(include=['object']).columns
                if len(categorical_cols_class) > 0:
                    X_class_encoded = pd.get_dummies(X_class, columns=categorical_cols_class, drop_first=True)
                else:
                    X_class_encoded = X_class.copy()
                
                # Разделяем данные
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                    X_class_encoded, y_class, test_size=0.2, random_state=42, stratify=y_class
                )
                
                # Обучаем модель
                st.write("**Training Random Forest Classifier...**")
                model_class = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                
                model_class.fit(X_train_class, y_train_class)
                
                # Оценка модели
                y_pred_class = model_class.predict(X_test_class)
                y_pred_proba = model_class.predict_proba(X_test_class)
                
                # Метрики
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test_class, y_pred_class)
                precision = precision_score(y_test_class, y_pred_class, average='weighted')
                recall = recall_score(y_test_class, y_pred_class, average='weighted')
                f1 = f1_score(y_test_class, y_pred_class, average='weighted')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                with col2:
                    st.metric("Precision", f"{precision:.3f}")
                with col3:
                    st.metric("Recall", f"{recall:.3f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.3f}")                            
                
                # Важность признаков для классификации
                st.subheader("Feature Importance for Classification")
                
                if hasattr(model_class, 'feature_importances_'):
                    feature_importance_class = pd.DataFrame({
                        'Feature': X_class_encoded.columns,
                        'Importance': model_class.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_importance_class,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features for Classification',
                        color='Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Интерактивная классификация
                st.markdown("---")
                st.subheader("Interactive Property Classification")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    class_sqft = st.number_input(
                        "Gross Square Feet",
                        min_value=100,
                        max_value=100000,
                        value=1500,
                        step=100,
                        key='class_sqft'
                    )
                    
                    class_borough = st.selectbox(
                        "Borough",
                        options=sorted(classification_df['BOROUGH'].unique()),
                        format_func=lambda x: {
                            1: 'Manhattan',
                            2: 'Brooklyn',
                            3: 'Queens',
                            4: 'Bronx',
                            5: 'Staten Island'
                        }.get(x, x),
                        key='class_borough'
                    )
                    
                    class_year = st.number_input(
                        "Year Built",
                        min_value=1700,
                        max_value=datetime.now().year,
                        value=1990,
                        step=1,
                        key='class_year'
                    )
                
                with col2:
                    class_units = st.number_input(
                        "Total Units",
                        min_value=1,
                        max_value=1000,
                        value=2,
                        step=1,
                        key='class_units'
                    )
                    
                    class_land_sqft = st.number_input(
                        "Land Square Feet",
                        min_value=100,
                        max_value=1000000,
                        value=2000,
                        step=100,
                        key='class_land_sqft'
                    )
                    
                    if 'BUILDING CLASS CATEGORY' in classification_df.columns:
                        class_building_types = sorted(classification_df['BUILDING CLASS CATEGORY'].unique())
                        class_building_type = st.selectbox(
                            "Building Type",
                            options=class_building_types,
                            key='class_building_type'
                        )
                
                if st.button("Classify Property"):
                    # Создаем DataFrame с введенными данными
                    input_class_data = pd.DataFrame({
                        'GROSS SQUARE FEET': [class_sqft],
                        'BOROUGH': [class_borough],
                        'YEAR BUILT': [class_year],
                        'TOTAL UNITS': [class_units],
                        'LAND SQUARE FEET': [class_land_sqft],
                        'BUILDING CLASS CATEGORY': [class_building_type]
                    })
                    
                    # Применяем те же преобразования
                    input_class_processed = pd.get_dummies(input_class_data, drop_first=True)
                    
                    # Выравниваем столбцы
                    for col in X_class_encoded.columns:
                        if col not in input_class_processed.columns:
                            input_class_processed[col] = 0
                    
                    input_class_processed = input_class_processed[X_class_encoded.columns]
                    
                    # Делаем предсказание
                    predicted_class = model_class.predict(input_class_processed)[0]
                    predicted_proba = model_class.predict_proba(input_class_processed)[0]
                    
                    # Определяем ценовой диапазон для предсказанной категории
                    category_ranges = {
                        0: (0, price_33),
                        1: (price_33, price_66),
                        2: (price_66, classification_df['SALE PRICE'].max())
                    }
                    
                    min_price, max_price = category_ranges[predicted_class]
                    
                    # Отображаем результат
                    category_name = le.inverse_transform([predicted_class])[0]
                    
                    st.success(f"""
                    **Classification Result: {category_name}**
                    
                    Category probabilities:
                    - Cheap: {predicted_proba[0]*100:.1f}%
                    - Medium: {predicted_proba[1]*100:.1f}%
                    - Expensive: {predicted_proba[2]*100:.1f}%
                    
                    **Expected price range:**
                    - From ${min_price:,.0f} to ${max_price:,.0f}
                    - Average category price: ${category_stats.loc[category_name, 'Median Price']:,.0f}
                    
                    **Typical characteristics of "{category_name}" category:**
                    - Area: {category_stats.loc[category_name, 'Median Area']:,.0f} sq ft
                    - Year built: {int(category_stats.loc[category_name, 'Median Year Built'])}
                    - Price per sq ft: ${category_stats.loc[category_name, 'Price per Sq Ft']:.2f}
                    """)
                    
                    # Визуализация вероятностей
                    prob_df = pd.DataFrame({
                        'Category': le.classes_,
                        'Probability (%)': predicted_proba * 100
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Category',
                        y='Probability (%)',
                        title='Price Category Membership Probabilities',
                        color='Probability (%)',
                        text='Probability (%)'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
