import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Загрузка данных
@st.cache
def load_data():
    data = pd.read_csv('nyc-rolling-sales.csv')
    return data

# Функция для очистки данных
def clean_data(df):
    # Преобразование SALE PRICE в числовой формат
    df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'].str.replace(',', '').str.strip(), errors='coerce')
    df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Генерация данных для кластеризации
def generate_cluster_data(df):
    # Используем только числовые данные для кластеризации
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X = df[numeric_columns].values
    return X

# Функция для кластеризации и построения графика
def plot_clusters(X):
    kmeans = KMeans(n_clusters=4)
    y_kmeans = kmeans.fit_predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('Кластеры с центроидами')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.grid()
    st.pyplot(plt)

# Функция для генерации данных временного ряда и построения графика
def plot_time_series(df):
    df['YEAR'] = df['SALE DATE'].dt.year
    df['MONTH'] = df['SALE DATE'].dt.month
    monthly_sales = df.groupby(['YEAR', 'MONTH'])['SALE PRICE'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales['MONTH'], monthly_sales['SALE PRICE'], marker='o')
    plt.title('Средняя цена продажи по месяцам')
    plt.xlabel('Месяц')
    plt.ylabel('Средняя цена продажи')
    plt.grid()
    st.pyplot(plt)

# Функция для генерации данных для классификации и построения confusion matrix и ROC curve
def plot_classification(df):
    # Пример: предсказание, будет ли цена продажи выше медианы
    df['TARGET'] = (df['SALE PRICE'] > df['SALE PRICE'].median()).astype(int)
    X = df[['LAND SQUARE FEET', 'GROSS SQUARE FEET']]
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    st.pyplot(plt)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ложноположительная ставка')
    plt.ylabel('Истинноположительная ставка')
    plt.title('ROC Кривая')
    plt.legend(loc="lower right")
    st.pyplot(plt)

# Основной код приложения
st.sidebar.title('Навигация')
page = st.sidebar.radio('Выберите страницу:', ['Кластеры', 'Временной ряд', 'Классификация'])

# Загрузка и очистка данных
data = load_data()
data = clean_data(data)

if page == 'Кластеры':
    st.title('Кластеры в scatter plots с центроидами')
    X = generate_cluster_data(data)
    plot_clusters(X)

elif page == 'Временной ряд':
    st.title('Средняя цена продажи по месяцам')
    plot_time_series(data)

elif page == 'Классификация':
    st.title('Confusion Matrix и ROC кривая')
    plot_classification(data)

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
