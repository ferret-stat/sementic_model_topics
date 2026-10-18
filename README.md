Semantic Model & Topic Analysis (sementic_model_topics)
Этот проект предназначен для предварительной обработки текста, анализа тональности (Sentiment Analysis) и тематического моделирования (Topic Modeling) с использованием алгоритма K-Means и моделей машинного обучения.
📁 Структура проекта
sementic_model_topics-main/
├── data/
│ ├── datasets_example.csv # Пример датасета для обучения/тестирования (содержит текст и тональность)
│ └── stopswords_example.txt # Список стоп-слов для предобработки текста
├── notebooks/
│ ├── datasets_metrics.ipynb # Исследование метрик и датасетов
│ ├── kmeans_optimized.ipynb # Оптимизированное тематическое моделирование на базе K-Means
│ ├── kmeans_optimized_v2.ipynb# Вторая версия / эксперименты с K-Means
│ └── sentiment_model.ipynb # Эксперименты и обучение модели анализа тональности
├── src/
│ ├── **init**.py
│ ├── preprocessing.py # Модуль для очистки и токенизации текста
│ ├── sentiment_model.py # Модуль для работы с моделью тональности
│ └── topic_model.py # Модуль для тематического моделирования
├── .gitignore # Конфигурация исключений Git
└── README.md # Документация проекта

⚙️ Основной функционал
Предобработка текста (src/preprocessing.py):
Очистка от спецсимволов и шума.
Удаление стоп-слов (с использованием data/stopswords_example.txt).
Подготовка текста к векторизации.
Анализ тональности (src/sentiment_model.py):
Классификация текстов по эмоциональной окраске (позитивный, негативный, нейтральный).
Обучение и тестирование в notebooks/sentiment_model.ipynb.
Тематическое моделирование (src/topic_model.py):
Кластеризация текстов с помощью алгоритма K-Means.
Подбор оптимального количества кластеров и оценка качества (см. notebooks/kmeans_optimized.ipynb).
🚀 Быстрый старт

1. Требования
   Убедитесь, что у вас установлен Python версии 3.10 или выше.
2. Установка зависимостей
   Создайте виртуальное окружение и установите необходимые библиотеки:

# Создание виртуального окружения

python -m venv venv

# Активация окружения (Linux/macOS)

source venv/bin/activate

# Активация окружения (Windows)

# venv\Scripts\activate

# Установка основных библиотек (при отсутствии requirements.txt)

pip install pandas numpy scikit-learn nltk jupyter

📊 Использование
Запуск Jupyter Notebooks
Для анализа данных и проведения экспериментов перейдите в папку notebooks/:
jupyter notebook notebooks/

sentiment_model.ipynb — запуск обучения модели классификации тональности.
kmeans_optimized.ipynb — кластеризация документов и извлечение ключевых тем.
Использование модулей Python
Вы можете импортировать логику из папки src/ в свои скрипты:
from src.preprocessing import preprocess_text
from src.sentiment_model import SentimentModel
from src.topic_model import TopicModel

# Пример очистки текста

raw_text = "Пример текста для анализа и кластеризации!"
clean_text = preprocess_text(raw_text)

print(f"Очищенный текст: {clean_text}")
