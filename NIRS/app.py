import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Прогноз", layout="centered")
st.title("Предсказание ухода сотрудника")
st.markdown("Выберите значения признаков и настройте количество деревьев модели Bagging для предсказания вероятности ухода сотрудника.")

# --- Ввод признаков пользователя ---
age = st.slider("Возраст", 18, 60, 30)
joining_year = st.slider("Год прихода в компанию", 2010, 2025, 2017)
experience = st.slider("Опыт в текущей области (лет)", 0, 20, 3)
payment_tier = st.selectbox("Уровень оплаты", [1, 2, 3])
gender = st.selectbox("Пол", ["Мужчина", "Женщина"])
education = st.selectbox("Образование", ["Bachelor", "Master", "PHD"])
city = st.selectbox("Город", ["Bangalore", "Pune", "New Delhi"])
ever_benched = st.selectbox("Был ли на \"скамейке\"?", ["Нет", "Да"])

# --- Гиперпараметры модели ---
n_estimators = st.slider("Число деревьев (n_estimators)", 10, 1000, 100, step=10)

# --- Кодирование категориальных признаков ---
gender_encoded = 1 if gender == "Мужчина" else 0
ever_benched_encoded = 1 if ever_benched == "Да" else 0
education_encoded = [int(education == e) for e in ["Bachelor", "Master", "PHD"]]
city_encoded = [int(city == c) for c in ["Bangalore", "Pune", "New Delhi"]]

# --- Масштабирование числовых признаков ---
scaler = StandardScaler()
scaled_vals = scaler.fit_transform([[age, joining_year]])[0]
age_scaled, joining_year_scaled = scaled_vals

# --- Формирование входного массива ---
X_input = np.array([
    age_scaled, joining_year_scaled, experience, payment_tier,
    gender_encoded, ever_benched_encoded,
    *education_encoded, *city_encoded
]).reshape(1, -1)

# --- Загрузка предварительно обученной модели ---
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=n_estimators,
    max_samples=0.6,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    random_state=42
)

# Загрузка или обучение модели на предобработанных данных (здесь должен быть ваш X_train, y_train)
# Замените следующую часть на загрузку своих данных
import pandas as pd
X_train = pd.read_csv("X_train_processed.csv")  # Предобработанный X
y_train = pd.read_csv("y_train.csv").values.ravel()
model.fit(X_train, y_train)

# --- Предсказание ---
y_pred = model.predict(X_input)[0]
y_prob = model.predict_proba(X_input)[0][1]

# --- Результаты ---
st.subheader("Результат предсказания")
st.write(f"**Вероятность ухода:** {y_prob:.2%}")
st.success("Сотрудник, вероятно, останется." if y_pred == 0 else "Сотрудник, скорее всего, уйдёт.")
