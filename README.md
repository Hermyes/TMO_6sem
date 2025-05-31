# TMO_6sem
Репозиторий по "Технологиям машинного обучения" 3 курс МГТУ

## [Лабораторная работа №1: Разведочный анализ данных](./lab1)

**Тема:** Исследование и визуализация данных  
**Датасет:** [Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file?resource=download)

### Краткое описание

В данной лабораторной работе произведён разведочный анализ данных на основе датасета о пассажирах Титаника. Работа включает:

- Импорт и первичную обработку данных
- Анализ структуры данных (колонки, типы, пропуски)
- Визуализации: scatter plot, histogram, pairplot, boxplot, violinplot
- Расчёт и визуализация корреляционной матрицы

### Основные выводы:

- Пол пассажира сильно влияет на выживаемость
- Умеренная отрицательная корреляция между классом билета и стоимостью
- Слабая корреляция между возрастом и другими признаками

### Используемые библиотеки:

```python
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
%matplotlib inline  
sns.set(style="ticks")
```

## [Лабораторная работа №2: Предобработка данных](./lab2)

**Тема:** Обработка пропусков, кодирование категориальных признаков и масштабирование  
**Датасет:** Housing

### Краткое описание

В данной лабораторной работе выполняется предобработка данных, включающая:

- Обнаружение и заполнение пропусков
- Кодирование категориальных признаков с помощью one-hot encoding
- Масштабирование числовых признаков с использованием MinMaxScaler

### Основные этапы обработки:

- В колонке `total_bedrooms` обнаружены пропущенные значения, заменённые на среднее
- Признак `ocean_proximity` преобразован с помощью one-hot кодирования
- Данные масштабированы в диапазон от 0 до 1

### Используемые библиотеки:

```python
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MinMaxScaler  
%matplotlib inline  
sns.set(style="ticks")
```

## [Лабораторная работа №3: Подготовка выборки и подбор гиперпараметров](./lab3)

**Тема:** Метод ближайших соседей, разбиение выборки, кросс-валидация и подбор параметров  
**Датасет:** Iris

### Краткое описание

В данной лабораторной работе проводится обучение модели k-ближайших соседей с подбором оптимального значения параметра `K` с помощью кросс-валидации и методов GridSearchCV и RandomizedSearchCV. Основные шаги:

- Загрузка и предварительный анализ датасета Iris
- Разделение выборки на обучающую и тестовую с использованием `train_test_split`
- Обучение модели `KNeighborsClassifier` с различными значениями `K`
- Подбор оптимального параметра `K` через GridSearchCV (KFold) и RandomizedSearchCV (StratifiedKFold)
- Оценка точности модели по метрикам классификации

### Результаты:

- **Точность модели с k=50:** 0.97
- **Лучший `k` (GridSearchCV):** 12 → точность 0.967
- **Лучший `k` (RandomizedSearchCV):** 14 → точность 0.958
- **Оптимальная модель (по GridSearchCV):** Точность на тесте 1.00

### Используемые библиотеки:

```python
import pandas as pd  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, classification_report
```

## [Лабораторная работа №4: Сравнение моделей классификации](./lab4)

**Тема:** Логистическая регрессия, SVM и дерево решений на задаче классификации  
**Датасет:** Crop Recommendation

### Краткое описание

В данной лабораторной работе реализовано обучение трёх моделей классификации (логистическая регрессия, метод опорных векторов и дерево решений) для определения подходящей сельскохозяйственной культуры на основе агрохимических и климатических данных. Работа включает:

- Первичный анализ и предобработка данных (`LabelEncoder`, `StandardScaler`)
- Деление на обучающую и тестовую выборки с помощью `train_test_split`
- Обучение моделей `LogisticRegression`, `SVC`, `DecisionTreeClassifier`
- Сравнение моделей по метрикам **Accuracy** и **F1-score**
- Визуализация важности признаков для дерева решений
- Построение схемы дерева и вывод логики решений в текстовом виде

### Результаты:

| Модель               | Accuracy | F1 Score |
|----------------------|----------|----------|
| Logistic Regression  | 0.973    | 0.972    |
| SVM                  | 0.984    | 0.984    |
| Decision Tree        | 0.980    | 0.979    |

### Используемые библиотеки:

```python
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler, LabelEncoder  
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text  
from sklearn.metrics import accuracy_score, f1_score, classification_report  
```

## [Лабораторная работа №5: Ансамблевые методы классификации](./lab5)

**Тема:** Сравнение ансамблевых моделей (бэггинг, бустинг)  
**Датасет:** Fake News Detection

### Краткое описание

В данной лабораторной работе реализована задача бинарной классификации новостных статей на настоящие и фейковые с использованием ансамблевых моделей. Работа включает:

- Предобработку текста (`clean_text`), заполнение пропусков
- Создание объединённого текстового признака
- Векторизацию текста с помощью `TfidfVectorizer`
- Обучение моделей: `BaggingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`
- Сравнение моделей по метрике Accuracy

### Результаты:

| Модель                 | Accuracy |
|------------------------|----------|
| AdaBoost               | 0.512    |
| Extra Trees            | 0.509    |
| Bagging (DecisionTree) | 0.508    |
| Gradient Boosting      | 0.506    |
| Random Forest          | 0.499    |

### Используемые библиотеки:

```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import re  

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,  
                              AdaBoostClassifier, GradientBoostingClassifier)  
from sklearn.metrics import accuracy_score  
```

## [Лабораторная работа №6: Модели регрессии — стекинг и нейронные сети](./lab6)

**Тема:** Сравнение моделей стекинга и MLP в задаче регрессии  
**Датасет:** California Housing

### Краткое описание

В данной лабораторной работе решается задача регрессии — предсказание медианной стоимости жилья на основе географических и социально-экономических признаков. Рассмотрены и сравниваются два подхода:

- Модель стекинга (`StackingRegressor`) с базовыми моделями: `LinearRegression`, `RandomForestRegressor`, `KNeighborsRegressor` и мета-моделью `GradientBoostingRegressor`
- Нейронная сеть — `MLPRegressor` с двумя скрытыми слоями

### Этапы:

- Загрузка датасета California Housing
- Масштабирование признаков с помощью `StandardScaler`
- Разделение выборки на обучающую и тестовую
- Обучение и тестирование моделей стекинга и MLP
- Расчёт метрик: **MSE** и **R²**

### Используемые библиотеки:

```python
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  

from sklearn.datasets import fetch_california_housing  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, r2_score  

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor  
from sklearn.linear_model import LinearRegression  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.neural_network import MLPRegressor  
```

## [НИР: Разработка и оценка моделей машинного обучения](./NIRS)

**Тема:** Предсказание ухода сотрудника на основе HR-данных  
**Датасет:** Employee.csv

### Краткое описание

В рамках научно-исследовательской работы была решена задача бинарной классификации — предсказание вероятности ухода сотрудника из компании. Проект охватывает весь цикл работы с данными: от разведочного анализа и подготовки до построения ансамблевых моделей и их внедрения в веб-приложение.

**Основные этапы:**

- Разведочный анализ данных (EDA)
- Кодирование категориальных и масштабирование числовых признаков
- Обучение моделей: `LogisticRegression`, `KNN`, `DecisionTree`, `RandomForest`, `XGBoost`, `Bagging`, `Stacking`
- Подбор гиперпараметров (GridSearchCV)
- Выбор финальной модели по метрикам: Accuracy, F1-score, ROC AUC
- Реализация интерактивного веб-приложения с возможностью настройки модели `BaggingClassifier`

### Итоговые метрики лучших моделей:

| Модель             | Accuracy | F1 Score | ROC AUC |
|--------------------|----------|----------|---------|
| Bagging (настроен) | 0.858    | 0.784    | 0.874   |
| XGBoost            | 0.865    | 0.785    | 0.892   |

Модель `BaggingClassifier` показала наилучший баланс между точностью, полнотой и устойчивостью. Она используется в веб-приложении.

### 🖥 Веб-приложение

Создано с помощью [Streamlit](https://streamlit.io/). Пользователь может:

- Ввести параметры сотрудника (возраст, город, опыт, образование и др.)
- Настроить число деревьев в ансамбле (`n_estimators`)
- Получить вероятность ухода сотрудника
- Визуально оценить влияние гиперпараметров на результат

Файл запуска: `app.py`  
Модель: `bagging_model.pkl`  
Тренировочные данные: `X_train_processed.csv`, `y_train.csv`

### Используемые библиотеки:

```python
import pandas as pd  
import numpy as np  
import streamlit as st  
from sklearn.ensemble import BaggingClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.preprocessing import StandardScaler  
```
