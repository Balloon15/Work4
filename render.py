import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Настройка страницы
st.set_page_config(
    page_title="NYC Property Sales Dashboard",
    page_icon="🏙️",
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

# Обратный словарь для поиска
REVERSE_TRANSLATIONS = {v: k for k, v in COLUMN_TRANSLATIONS.items()}

# Функция для перевода названий колонок
def translate_columns(df):
    df = df.copy()
    df.columns = [COLUMN_TRANSLATIONS.get(col, col) for col in df.columns]
    return df

# Загрузка данных с кэшированием
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("nyc-rolling-sales.csv")
    except FileNotFoundError:
        # Если файл не найден, создаем пример данных для демонстрации
        st.warning("Файл данных не найден. Используются демонстрационные данные.")
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'BOROUGH': np.random.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], n_samples),
            'NEIGHBORHOOD': np.random.choice(['Upper East Side', 'Williamsburg', 'Astoria', 'Riverdale', 'St. George'], n_samples),
            'BUILDING CLASS CATEGORY': np.random.choice(['01 ONE FAMILY DWELLINGS', '02 TWO FAMILY DWELLINGS', '03 THREE FAMILY DWELLINGS'], n_samples),
            'SALE PRICE': np.random.randint(50000, 5000000, n_samples),
            'GROSS SQUARE FEET': np.random.randint(500, 5000, n_samples),
            'LAND SQUARE FEET': np.random.randint(1000, 10000, n_samples),
            'YEAR BUILT': np.random.randint(1900, 2020, n_samples),
            'RESIDENTIAL UNITS': np.random.randint(1, 10, n_samples),
            'COMMERCIAL UNITS': np.random.randint(0, 5, n_samples),
            'TOTAL UNITS': np.random.randint(1, 15, n_samples),
            'ZIP CODE': np.random.randint(10001, 11698, n_samples),
            'SALE DATE': pd.date_range('2016-01-01', periods=n_samples, freq='D'),
            'TAX CLASS AT PRESENT': np.random.choice(['1', '2', '3', '4'], n_samples),
            'ADDRESS': [f"{i} Main St" for i in range(1, n_samples + 1)],
        })
        data['TOTAL UNITS'] = data['RESIDENTIAL UNITS'] + data['COMMERCIAL UNITS']
    
    # Преобразование типов данных
    numeric_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 
                       'YEAR BUILT', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 
                       'TOTAL UNITS']
    
    for col in numeric_columns:
        if col in data.columns:
            # Очистка и преобразование числовых значений
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
                errors='coerce'
            )
    
    # Преобразуем дату
    if 'SALE DATE' in data.columns:
        data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce')
    
    # Очистка данных
    if 'SALE PRICE' in data.columns:
        # Удаляем выбросы и некорректные значения
        data = data[(data['SALE PRICE'] > 100) & (data['SALE PRICE'] < 1e9)]
    
    if 'YEAR BUILT' in data.columns:
        data = data[(data['YEAR BUILT'] > 1800) & (data['YEAR BUILT'] <= datetime.now().year)]
    
    return data.dropna(subset=['SALE PRICE']).reset_index(drop=True)

# Загружаем данные
df = load_data()

# Создаем словари для фильтров
neighborhoods_dict = {'Все': None}
neighborhoods_dict.update({n: n for n in sorted(df['NEIGHBORHOOD'].dropna().unique())})

building_classes_dict = {'Все': None}
building_classes_dict.update({b: b for b in sorted(df['BUILDING CLASS CATEGORY'].dropna().unique())})

# Создаем навигацию
st.sidebar.title("🏙️ NYC Property Sales Dashboard")
page = st.sidebar.radio(
    "Навигация",
    ["📊 Визуализация данных", "📈 Анализ", "📋 Таблица переводов"],
    label_visibility="collapsed"
)

# Добавляем фильтры в сайдбар
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Фильтры данных")

# Фильтр по району
selected_neighborhood_key = st.sidebar.selectbox(
    "Район",
    options=list(neighborhoods_dict.keys()),
    index=0
)
selected_neighborhood = neighborhoods_dict[selected_neighborhood_key]

# Фильтр по типу здания
selected_building_key = st.sidebar.selectbox(
    "Категория класса здания",
    options=list(building_classes_dict.keys()),
    index=0
)
selected_building_class = building_classes_dict[selected_building_key]

# Фильтр по году постройки
if 'YEAR BUILT' in df.columns:
    min_year = int(df['YEAR BUILT'].min())
    max_year = int(df['YEAR BUILT'].max())
    year_range = st.sidebar.slider(
        "Год постройки",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

# Фильтр по цене
if 'SALE PRICE' in df.columns:
    min_price = float(df['SALE PRICE'].quantile(0.01))  # Используем 1-й перцентиль для исключения выбросов
    max_price = float(df['SALE PRICE'].quantile(0.99))  # Используем 99-й перцентиль
    price_range = st.sidebar.slider(
        "Цена продажи ($)",
        min_value=float(min_price),
        max_value=float(max_price),
        value=(float(min_price), float(max_price))
    )

# Применяем фильтры
filtered_df = df.copy()

if selected_neighborhood:
    filtered_df = filtered_df[filtered_df['NEIGHBORHOOD'] == selected_neighborhood]

if selected_building_class:
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

# Создаем DataFrame с русскими названиями
filtered_df_ru = translate_columns(filtered_df.copy())

# Страница 3: Таблица переводов
if page == "📋 Таблица переводов":
    st.title("📋 Таблица переводов названий колонок")
    
    # Создаем таблицу с переводами
    translation_data = []
    for eng, rus in COLUMN_TRANSLATIONS.items():
        if eng in df.columns:
            sample_value = "✓" if eng in filtered_df.columns else "✗"
            translation_data.append({
                "Оригинальное название (англ.)": eng,
                "Перевод (рус.)": rus,
                "В данных": sample_value
            })
    
    translation_df = pd.DataFrame(translation_data)
    
    # Показываем статистику
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего колонок", len(COLUMN_TRANSLATIONS))
    with col2:
        st.metric("В текущих данных", len([c for c in COLUMN_TRANSLATIONS if c in filtered_df.columns]))
    with col3:
        st.metric("Переведено", len([c for c in filtered_df.columns if c in COLUMN_TRANSLATIONS]))
    
    st.markdown("---")
    
    # Отображаем таблицу
    st.dataframe(
        translation_df,
        use_container_width=True,
        height=600,
        column_config={
            "Оригинальное название (англ.)": st.column_config.TextColumn(width="large"),
            "Перевод (рус.)": st.column_config.TextColumn(width="large"),
            "В данных": st.column_config.TextColumn(width="small")
        }
    )

# Страница 1: Визуализация исходных данных
elif page == "📊 Визуализация данных":
    st.title("📊 Визуализация данных о продажах недвижимости")
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего записей", f"{len(filtered_df):,}")
    
    with col2:
        if 'SALE PRICE' in filtered_df.columns:
            avg_price = filtered_df['SALE PRICE'].mean()
            st.metric("Средняя цена", f"${avg_price:,.0f}")
    
    with col3:
        if 'GROSS SQUARE FEET' in filtered_df.columns:
            avg_sqft = filtered_df['GROSS SQUARE FEET'].mean()
            st.metric("Ср. площадь", f"{avg_sqft:,.0f} кв.фут")
    
    with col4:
        unique_neighborhoods = filtered_df['NEIGHBORHOOD'].nunique()
        st.metric("Районов", unique_neighborhoods)
    
    st.markdown("---")
    
    # Вкладки для разных типов визуализаций
    tab1, tab2, tab3 = st.tabs(["📈 Распределения", "🗺️ География", "📊 Статистика"])
    
    with tab1:
        # Графики распределений
        col1, col2 = st.columns(2)
        
        with col1:
            if 'SALE PRICE' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df_ru,
                    x='Цена продажи',
                    nbins=50,
                    title="Распределение цен на недвижимость",
                    labels={'Цена продажи': 'Цена ($)'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(xaxis_tickformat=',', showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'YEAR BUILT' in filtered_df.columns:
                fig = px.histogram(
                    filtered_df_ru,
                    x='Год постройки',
                    nbins=30,
                    title="Распределение по году постройки",
                    color_discrete_sequence=['#2ca02c']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        if 'SALE PRICE' in filtered_df.columns and 'GROSS SQUARE FEET' in filtered_df.columns:
            fig = px.scatter(
                filtered_df_ru,
                x='Общая площадь (кв. фут)',
                y='Цена продажи',
                title="Зависимость цены от площади",
                labels={
                    'Общая площадь (кв. фут)': 'Площадь (кв.фут)',
                    'Цена продажи': 'Цена ($)'
                },
                opacity=0.6,
                color_discrete_sequence=['#ff7f0e']
            )
            fig.update_layout(xaxis_tickformat=',', yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Географический анализ
        col1, col2 = st.columns(2)
        
        with col1:
            if 'BOROUGH' in filtered_df.columns:
                borough_counts = filtered_df['BOROUGH'].value_counts()
                fig = px.bar(
                    x=borough_counts.index,
                    y=borough_counts.values,
                    title="Распределение по городским округам",
                    labels={'x': 'Округ', 'y': 'Количество продаж'},
                    color=borough_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'NEIGHBORHOOD' in filtered_df.columns:
                top_neighborhoods = filtered_df['NEIGHBORHOOD'].value_counts().head(10)
                fig = px.pie(
                    values=top_neighborhoods.values,
                    names=top_neighborhoods.index,
                    title="Топ 10 районов по количеству продаж",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Статистика и таблица
        st.subheader("Базовая статистика")
        
        if st.checkbox("Показать статистики", value=True):
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                stats_df = filtered_df[numeric_cols].describe().T
                stats_df = stats_df[['count', 'mean', 'std', 'min', '50%', 'max']]
                stats_df.columns = ['Кол-во', 'Среднее', 'Стд. откл.', 'Мин.', 'Медиана', 'Макс.']
                stats_df.index = [COLUMN_TRANSLATIONS.get(col, col) for col in numeric_cols]
                
                st.dataframe(
                    stats_df.style.format("{:,.2f}"),
                    use_container_width=True,
                    height=400
                )
        
        st.subheader("Просмотр данных")
        
        # Выбор колонок для отображения
        available_columns = filtered_df_ru.columns.tolist()
        selected_columns = st.multiselect(
            "Выберите колонки:",
            available_columns,
            default=available_columns[:min(8, len(available_columns))]
        )
        
        if selected_columns:
            # Пагинация
            page_size = st.selectbox("Строк на странице:", [10, 25, 50, 100], index=0)
            total_pages = max(1, len(filtered_df_ru) // page_size + 1)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_number = st.number_input(
                    "Страница:",
                    min_value=1,
                    max_value=total_pages,
                    value=1
                )
            
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            
            display_df = filtered_df_ru[selected_columns].iloc[start_idx:end_idx]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Экспорт данных
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать все данные (CSV)",
                data=csv,
                file_name="nyc_property_sales.csv",
                mime="text/csv",
                use_container_width=True
            )

# Страница 2: Результаты анализа
else:
    st.title("📈 Анализ продаж недвижимости")
    
    # Информация о выбранных данных
    with st.expander("📋 Информация о выборке", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Записей в выборке", len(filtered_df))
        with col2:
            st.metric("Доля от всех данных", f"{(len(filtered_df)/len(df)*100):.1f}%")
        with col3:
            if 'SALE PRICE' in filtered_df.columns:
                st.metric("Общий объем продаж", f"${filtered_df['SALE PRICE'].sum():,.0f}")
    
    # Анализ трендов
    st.subheader("📅 Анализ трендов")
    
    if 'SALE DATE' in filtered_df.columns:
        # Анализ по месяцам
        filtered_df['SALE_MONTH'] = filtered_df['SALE DATE'].dt.to_period('M').astype(str)
        monthly_stats = filtered_df.groupby('SALE_MONTH').agg({
            'SALE PRICE': ['count', 'mean', 'median']
        }).round(2).reset_index()
        
        monthly_stats.columns = ['Месяц', 'Кол-во продаж', 'Средняя цена', 'Медианная цена']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                monthly_stats,
                x='Месяц',
                y='Кол-во продаж',
                title="Динамика количества продаж",
                markers=True,
                line_shape='spline'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                monthly_stats,
                x='Месяц',
                y='Медианная цена',
                title="Динамика медианной цены",
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(yaxis_tickformat=',')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Анализ районов
    st.subheader("🏘️ Анализ по районам")
    
    if 'NEIGHBORHOOD' in filtered_df.columns and 'SALE PRICE' in filtered_df.columns:
        neighborhood_analysis = filtered_df.groupby('NEIGHBORHOOD').agg({
            'SALE PRICE': ['count', 'mean', 'median', 'std'],
            'GROSS SQUARE FEET': 'mean'
        }).round(2).reset_index()
        
        neighborhood_analysis.columns = [
            'Район', 'Кол-во продаж', 'Средняя цена', 
            'Медианная цена', 'Стд. откл.', 'Средняя площадь'
        ]
        
        neighborhood_analysis['Цена за кв.фут'] = (
            neighborhood_analysis['Средняя цена'] / neighborhood_analysis['Средняя площадь']
        ).round(2)
        
        # Сортировка и выбор топ N
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox(
                "Сортировать по:",
                ['Кол-во продаж', 'Средняя цена', 'Медианная цена', 'Цена за кв.фут']
            )
        with col2:
            top_n = st.slider("Показать топ:", 5, 25, 10)
        
        sorted_df = neighborhood_analysis.sort_values(sort_by, ascending=False).head(top_n)
        
        # График и таблица
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                sorted_df,
                x='Район',
                y=sort_by,
                title=f"Топ {top_n} районов по {sort_by.lower()}",
                color=sort_by,
                color_continuous_scale='thermal'
            )
            fig.update_xaxes(tickangle=45)
            if 'цена' in sort_by.lower():
                fig.update_layout(yaxis_tickformat=',')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                sorted_df.style.format({
                    'Средняя цена': '${:,.0f}',
                    'Медианная цена': '${:,.0f}',
                    'Стд. откл.': '${:,.0f}',
                    'Средняя площадь': '{:,.0f}',
                    'Цена за кв.фут': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )
    
    # Корреляционный анализ
    st.subheader("🔗 Корреляционный анализ")
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = filtered_df[numeric_cols].corr()
        
        # Переводим названия для отображения
        numeric_cols_ru = [COLUMN_TRANSLATIONS.get(col, col) for col in numeric_cols]
        corr_matrix.index = numeric_cols_ru
        corr_matrix.columns = numeric_cols_ru
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Корреляционная матрица числовых признаков",
            color_continuous_scale='RdBu',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Находим самые сильные корреляции
        st.write("**Самые сильные корреляции:**")
        corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_pairs.append({
                    'Признак 1': corr_matrix.index[i],
                    'Признак 2': corr_matrix.columns[j],
                    'Корреляция': abs(corr_matrix.iloc[i, j])
                })
        
        corr_df = pd.DataFrame(corr_pairs).sort_values('Корреляция', ascending=False).head(10)
        st.dataframe(corr_df, use_container_width=True)
    
    # Ценовые сегменты
    st.subheader("💰 Анализ ценовых сегментов")
    
    if 'SALE PRICE' in filtered_df.columns:
        # Определяем ценовые категории
        price_quantiles = filtered_df['SALE PRICE'].quantile([0.25, 0.5, 0.75, 0.9])
        
        price_bins = [0, price_quantiles[0.25], price_quantiles[0.5], 
                     price_quantiles[0.75], price_quantiles[0.9], float('inf')]
        
        price_labels = [
            f'Низкая (<${price_quantiles[0.25]:,.0f})',
            f'Средняя-низкая (${price_quantiles[0.25]:,.0f}-${price_quantiles[0.5]:,.0f})',
            f'Средняя (${price_quantiles[0.5]:,.0f}-${price_quantiles[0.75]:,.0f})',
            f'Средняя-высокая (${price_quantiles[0.75]:,.0f}-${price_quantiles[0.9]:,.0f})',
            f'Высокая (>${price_quantiles[0.9]:,.0f})'
        ]
        
        filtered_df['PRICE_SEGMENT'] = pd.cut(
            filtered_df['SALE PRICE'],
            bins=price_bins,
            labels=price_labels,
            include_lowest=True
        )
        
        # Анализ по сегментам
        segment_analysis = filtered_df.groupby('PRICE_SEGMENT').agg({
            'SALE PRICE': ['count', 'mean', 'median'],
            'GROSS SQUARE FEET': 'mean',
            'YEAR BUILT': 'mean'
        }).round(2).reset_index()
        
        segment_analysis.columns = [
            'Ценовой сегмент', 'Кол-во', 'Средняя цена', 
            'Медианная цена', 'Ср. площадь', 'Ср. год постройки'
        ]
        
        segment_analysis['Цена за кв.фут'] = (
            segment_analysis['Средняя цена'] / segment_analysis['Ср. площадь']
        ).round(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                segment_analysis,
                x='Ценовой сегмент',
                y='Кол-во',
                title="Распределение по ценовым сегментам",
                color='Кол-во',
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                segment_analysis.style.format({
                    'Средняя цена': '${:,.0f}',
                    'Медианная цена': '${:,.0f}',
                    'Ср. площадь': '{:,.0f}',
                    'Ср. год постройки': '{:.0f}',
                    'Цена за кв.фут': '${:.2f}'
                }),
                use_container_width=True
            )
    
    # Инсайты и рекомендации
    st.subheader("💡 Ключевые инсайты")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("##### 📊 Статистические инсайты:")
        
        insights = []
        
        if 'SALE PRICE' in filtered_df.columns:
            # Коэффициент вариации цен
            cv_price = (filtered_df['SALE PRICE'].std() / filtered_df['SALE PRICE'].mean()) * 100
            insights.append(f"**Волатильность цен**: {cv_price:.1f}% (коэффициент вариации)")
            
            # Распределение по квартилям
            q1, q3 = filtered_df['SALE PRICE'].quantile([0.25, 0.75])
            iqr = q3 - q1
            insights.append(f"**Межквартильный размах**: ${iqr:,.0f}")
        
        if 'YEAR BUILT' in filtered_df.columns:
            recent_buildings = filtered_df[filtered_df['YEAR BUILT'] > 2000]
            if len(recent_buildings) > 0:
                pct_recent = len(recent_buildings) / len(filtered_df) * 100
                insights.append(f"**Новые постройки**: {pct_recent:.1f}% зданий построены после 2000 года")
        
        for insight in insights:
            st.write(f"• {insight}")
    
    with insight_col2:
        st.markdown("##### 🎯 Рекомендации для анализа:")
        
        recommendations = [
            "**Для инвесторов**: Сфокусируйтесь на районах с высокой ценой за кв.фут",
            "**Для застройщиков**: Проанализируйте спрос в разных ценовых сегментах",
            "**Для аналитиков**: Изучите сезонность для прогнозирования цен",
            "**Для риелторов**: Обратите внимание на корреляции между характеристиками объектов"
        ]
        
        for rec in recommendations:
            st.write(rec)

# Футер
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ О дашборде")
st.sidebar.info("""
**NYC Property Sales Dashboard**  
Визуализация и анализ данных  
о продажах недвижимости в Нью-Йорке

**Данные**: NYC Rolling Sales Dataset
""")

# Кнопка сброса фильтров
if st.sidebar.button("🔄 Сбросить фильтры", use_container_width=True):
    st.rerun()

# Статус
st.sidebar.markdown(f"*Данные загружены: {len(df):,} записей*")
st.sidebar.markdown(f"*Отфильтровано: {len(filtered_df):,} записей*")
