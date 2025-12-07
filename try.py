import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ====== Загрузка и очистка данных ======
@st.cache_data
def load_data():
    # Загружаем CSV вручную с правильным разделителем и именами колонок (по структуре файла)
    # Судя по данным, у вас 21 колонка
    cols = [
        "id", "borough", "neighborhood", "building_class_category", "tax_class_at_present",
        "block", "lot", "ease_ment", "building_class_at_present", "address", "apartment_number",
        "zip_code", "residential_units", "commercial_units", "total_units", "land_square_feet",
        "gross_square_feet", "year_built", "tax_class_at_time_of_sale", "building_class_at_time_of_sale",
        "sale_price", "sale_date"
    ]
    df = pd.read_csv("nyc-rolling-sales.csv", names=cols, skiprows=1, low_memory=False)

    # Очистка: приведение типов
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    df['land_square_feet'] = pd.to_numeric(df['land_square_feet'], errors='coerce')
    df['gross_square_feet'] = pd.to_numeric(df['gross_square_feet'], errors='coerce')
    df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce')
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')

    # Фильтрация некорректных/нулевых цен (часто — сделки без реальной стоимости)
    df = df[(df['sale_price'] > 10000) & (df['sale_price'] < 100_000_000)]
    # Удаление явно некорректных площадей
    df = df[
        (df['land_square_feet'] > 100) |
        (df['gross_square_feet'] > 100) |
        (df['land_square_feet'].isna()) |
        (df['gross_square_feet'].isna())
    ]
    return df

df = load_data()
df_original = df.copy()

# ====== Настройка страницы ======
st.set_page_config(page_title="NYC Real Estate Dashboard", layout="wide")
st.title("🏙️ NYC Rolling Sales Dashboard")

# Навигация
page = st.sidebar.radio("🧭 Navigation", ["📊 Raw Data Visualization", "🔬 Analysis Results"])

# ====== Страница 1: Raw Data Visualization ======
if page == "📊 Raw Data Visualization":
    st.header("🔍 Raw Data Overview")

    # --- KPI карточки ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Avg Sale Price", f"${df['sale_price'].mean():,.0f}")
    col3.metric("Median Sale Price", f"${df['sale_price'].median():,.0f}")
    col4.metric("Missing Price", f"{df['sale_price'].isnull().sum()}")

    # --- Фильтры (в раскрывающемся блоке) ---
    with st.expander("⚙️ Filters"):
        col_a, col_b, col_c, col_d = st.columns(4)
        boroughs = ["All"] + sorted(df["borough"].dropna().unique().tolist())
        selected_borough = col_a.selectbox("Borough", boroughs)
        min_price = int(df["sale_price"].quantile(0.01))
        max_price = int(df["sale_price"].quantile(0.99))
        price_range = col_b.slider("Sale Price Range ($)", min_price, max_price, (min_price, max_price))
        
        min_year = int(df["year_built"].min())
        max_year = int(df["year_built"].max())
        year_range = col_c.slider("Year Built", min_year, max_year, (1900, max_year))
        
        # Фильтрация
        filtered_df = df[
            (df["sale_price"] >= price_range[0]) &
            (df["sale_price"] <= price_range[1]) &
            (df["year_built"] >= year_range[0]) &
            (df["year_built"] <= year_range[1])
        ]
        if selected_borough != "All":
            filtered_df = filtered_df[filtered_df["borough"] == selected_borough]

    # --- Таблица данных ---
    st.subheader("📋 Sample Data")
    st.dataframe(
        filtered_df.head(20).reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    # --- Распределения — два ряда по 2 графика ---
    st.subheader("📈 Feature Distributions")

    tab_num, tab_cat = st.tabs(["🔢 Numerical", "🔤 Categorical"])

    with tab_num:
        cols_num = ['sale_price', 'gross_square_feet', 'year_built', 'total_units']
        figs = []
        for i, col in enumerate(cols_num):
            if col in filtered_df.columns and filtered_df[col].dtype in ['int64', 'float64']:
                fig = px.histogram(
                    filtered_df, x=col, nbins=50,
                    marginal="box",
                    title=f"Distribution of {col.replace('_', ' ').title()}"
                )
                fig.update_layout(height=350)
                figs.append(fig)
        for i in range(0, len(figs), 2):
            cols_plot = st.columns(2)
            for j, fig in enumerate(figs[i:i+2]):
                with cols_plot[j]:
                    st.plotly_chart(fig, use_container_width=True)

    with tab_cat:
        cols_cat = ['borough', 'neighborhood', 'building_class_category']
        for col in cols_cat:
            if col in filtered_df.columns:
                top_n = filtered_df[col].value_counts().nlargest(10)
                fig = px.bar(
                    x=top_n.index, y=top_n.values,
                    labels={'x': col, 'y': 'Count'},
                    title=f"Top 10 {col.replace('_', ' ').title()}",
                    color=top_n.values
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Корреляционная матрица ---
    st.subheader("🔗 Correlation Heatmap")
    num_cols = ['sale_price', 'land_square_feet', 'gross_square_feet', 'year_built', 'total_units']
    corr_df = filtered_df[num_cols].corr()
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix (Numeric Features)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- Scatter plot matrix (выбор двух признаков) ---
    st.subheader("🔍 Scatter Plot (Interactive)")
    col_x = st.selectbox("X-axis", num_cols, index=1)
    col_y = st.selectbox("Y-axis", num_cols, index=0)
    color_by = st.selectbox("Color by", ["None", "borough", "building_class_category"])

    scatter_fig = px.scatter(
        filtered_df,
        x=col_x,
        y=col_y,
        color=None if color_by == "None" else color_by,
        hover_data=["address", "neighborhood"],
        title=f"{col_y} vs {col_x}",
        trendline="ols"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# ====== Страница 2: Analysis Results ======
elif page == "🔬 Analysis Results":
    st.header("🧪 Advanced Analysis & Modeling")

    # Подготовка данных для модели
    df_model = df_original.copy()
    df_model = df_model.dropna(subset=['sale_price', 'gross_square_feet', 'year_built', 'total_units'])
    df_model = df_model[(df_model['gross_square_feet'] > 0) & (df_model['year_built'] > 1800)]

    features = ['gross_square_feet', 'year_built', 'total_units']
    X = df_model[features]
    y = df_model['sale_price']

    # --- 1. Кластеризация (KMeans) ---
    st.subheader("🏘️ Clustering: Property Segments")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_model['cluster'] = clusters

    # Визуализация кластеров
    fig_cluster = px.scatter_3d(
        df_model,
        x='gross_square_feet',
        y='year_built',
        z='total_units',
        color='cluster',
        symbol='cluster',
        opacity=0.7,
        title="Property Clusters (3D)",
        hover_data=['sale_price', 'neighborhood']
    )
    fig_cluster.update_layout(height=500)
    st.plotly_chart(fig_cluster, use_container_width=True)

    # KPI по кластерам
    cluster_stats = df_model.groupby('cluster')['sale_price'].agg(['mean', 'median', 'count']).round()
    cluster_stats['mean'] = cluster_stats['mean'].apply(lambda x: f"${x:,.0f}")
    cluster_stats['median'] = cluster_stats['median'].apply(lambda x: f"${x:,.0f}")
    st.write("**Cluster Insights**")
    st.dataframe(cluster_stats, use_container_width=True)

    # --- 2. Прогнозирование цены (Random Forest) ---
    st.subheader("💰 Price Prediction (Regression)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # KPI
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("MAE ($)", f"${mae:,.0f}")

    # Scatter предсказаний
    pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
    fig_pred = px.scatter(
        pred_df,
        x='Actual',
        y='Predicted',
        trendline="ols",
        title="Predicted vs Actual Sale Price",
        labels={"Actual": "Actual Price ($)", "Predicted": "Predicted Price ($)"},
        opacity=0.5
    )
    fig_pred.add_shape(
        type="line", line=dict(dash='dash', color="gray"),
        x0=pred_df['Actual'].min(), x1=pred_df['Actual'].max(),
        y0=pred_df['Actual'].min(), y1=pred_df['Actual'].max()
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Feature Importance
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    fig_fi = px.bar(fi_df, x='importance', y='feature', orientation='h',
                    title="Feature Importance (Random Forest)")
    st.plotly_chart(fig_fi, use_container_width=True)

    # --- 3. Геовизуализация (Folium) ---
    st.subheader("📍 Geographic Distribution (Sample of 1000 sales)")

    # Подготовим координаты (примечание: в датасете нет lat/lon — сымитируем через zip → средние координаты по районам)
    # Для демо — возьмём средние координаты по borough (упрощённо)
    borough_coords = {
        1: (40.7505, -73.9934),  # Manhattan
        2: (40.8448, -73.8648),  # Bronx
        3: (40.6501, -73.9496),  # Brooklyn
        4: (40.7282, -73.7949),  # Queens
        5: (40.5795, -74.1502),  # Staten Island
    }

    df_map = df_model.sample(n=min(1000, len(df_model)), random_state=42).copy()
    df_map['lat'] = df_map['borough'].map(lambda b: borough_coords.get(b, (40.7, -74.0))[0])
    df_map['lon'] = df_map['borough'].map(lambda b: borough_coords.get(b, (40.7, -74.0))[1])
    # Немного "размажем" точки, чтобы не все в одной точке
    np.random.seed(42)
    df_map['lat'] += np.random.normal(0, 0.005, len(df_map))
    df_map['lon'] += np.random.normal(0, 0.005, len(df_map))

    # Folium карта
    m = folium.Map(location=[40.7, -73.9], zoom_start=10, tiles="CartoDB positron")
    for _, row in df_map.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2 + np.log1p(row['sale_price']) / 10,
            color="steelblue",
            fill=True,
            fill_opacity=0.6,
            popup=f"${row['sale_price']:,.0f}<br>{row['neighborhood']}<br>{row['address']}"
        ).add_to(m)

    st_folium(m, width=1000, height=500)

    # --- Интерактивный insight по фильтрам ---
    st.subheader("💡 Dynamic Insights")
    selected_cluster = st.selectbox("Select Cluster to Analyze", sorted(df_model['cluster'].unique()))
    cluster_data = df_model[df_model['cluster'] == selected_cluster]
    overall_median = df_model['sale_price'].median()
    cluster_median = cluster_data['sale_price'].median()
    diff_pct = (cluster_median - overall_median) / overall_median * 100

    st.info(f"""
    🔹 **Cluster {selected_cluster}** has **{len(cluster_data):,} properties**.  
    🔹 Median sale price: **${cluster_median:,.0f}**  
    🔹 This is **{diff_pct:+.1f}%** {'above' if diff_pct > 0 else 'below'} the citywide median (${overall_median:,.0f}).
    """)

# ====== Footer ======
st.sidebar.markdown("---")
st.sidebar.caption("💡 Built with Streamlit • NYC Rolling Sales Dataset")
st.sidebar.caption("✅ Ready to scale: add time-series, SHAP, export to PDF/CSV")
