import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Функция для генерации данных для кластеризации
def generate_cluster_data():
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return X, y

# Функция для кластеризации и построения графика
def plot_clusters(X, y):
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
def plot_time_series():
    np.random.seed(0)
    time = np.arange(100)
    data = 0.5 * time + np.random.normal(size=time.shape)

    X = time.reshape(-1, 1)
    y = data
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.plot(time, data, label='Данные', color='blue')
    plt.plot(time, predictions, label='Линия тренда', color='red', linestyle='--')
    plt.title('Временной ряд с линией тренда')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Функция для генерации данных для классификации и построения confusion matrix и ROC curve
def plot_classification():
    X_class, y_class = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=0)

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

if page == 'Кластеры':
    st.title('Кластеры в scatter plots с центроидами')
    X, y = generate_cluster_data()
    plot_clusters(X, y)

elif page == 'Временной ряд':
    st.title('Линия тренда для временного ряда')
    plot_time_series()

elif page == 'Классификация':
    st.title('Confusion Matrix и ROC кривая')
    plot_classification()

# Запуск приложения
if __name__ == '__main__':
    st.write("Запустите это приложение с помощью `streamlit run app.py`")
