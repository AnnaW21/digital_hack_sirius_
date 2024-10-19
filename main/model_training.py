import pickle
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from collections import Counter
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

# Функция для извлечения ключевых фраз (1-3 слова)
def extract_key_phrases(text):
    doc = nlp(text)
    key_phrases = []
    # Ищем фразы от 1 до 3 слов
    for n in range(1, 4):  # n - количество слов в фразе
        for i in range(len(doc) - n + 1):
            phrase = ' '.join([token.lemma_ for token in doc[i:i+n] if not token.is_stop and not token.is_punct])
            if len(phrase.split()) > 0:
                key_phrases.append(phrase)
    return key_phrases

# Функция для удаления именованных сущностей
def remove_named_entities(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        # Удаляем только именованные сущности (PER, ORG, GPE и т.д.)
        if token.ent_type_ not in ["PER", "ORG", "GPE", "LOC"]:
            filtered_tokens.append(token.text)
        else:
            filtered_tokens.append(" ")

    return " ".join(filtered_tokens)

def preprocessing(file):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Список стоп-слов для русского языка
    stopwords_rus = stopwords.words('russian')

    try:
        df = pd.read_excel(file)
    except:
        try:
            df = pd.read_csv(file)
        except:
            try:
                df = pd.read_json(file)
            except:
                pass

    # объединение всех столбцов, кроме последнего в один
    columns_to_combine = df.iloc[:, :3]  # Здесь берутся все столбцы, если нужно выбрать конкретные, можно указать их индексы

    # Объединение данных всех столбцов в один
    combined_column = pd.concat([columns_to_combine[col] for col in columns_to_combine.columns], ignore_index=True)

    # Создание нового DataFrame с объединённым столбцом
    df = pd.DataFrame({'combined': combined_column})
    df = df.dropna()

    # Предобработка данных датафрейма
    df_new = []
    df_key_phrases = []
    for i in range(df.shape[0]):
        # Приводим текст к нижнему регистру
        string = str(df.iloc[i][0])
        string = string.lower()
        # print("string", string)

        # Уберём неинформативные данные (оставим только текст)
        string = re.sub("([^0-9А-ЯЁа-яё \t])|(\w+:\/\/\S+)", " ", string)
        # print(f"удаление неинформативных данных\n: {string}")
        if string != '':

            # удаление имен собственных
            string = remove_named_entities(string)

            # Извлечение ключевых фраз из очищенных текстов
            df_key_phrases.append(extract_key_phrases(string))

            # Токенизируем текст
            string = word_tokenize(string)
            # print(f"Токенизация текста\n: {string}")

            # Удалим стоп-слова
            string_withoutstop = [word for word in string if word not in stopwords_rus]
            # print(f"Удаление стоп-слов\n: {string_withoutstop}")

            # Лемматизируем (приведем к исходной форме) слова
            string = [WordNetLemmatizer().lemmatize(word) for word in string_withoutstop]
            # print(f"Лемматизация текста\n: {string}")

            df_new.append(string)

        else:
            pass
    df_new = pd.Series(df_new)
    df_key_phrases = pd.Series(df_key_phrases)

    # ????? !!!
    for i in range(df_key_phrases.shape[0]):
        print(type(df_key_phrases.iloc[i]))
        df_key_phrases.iloc[i] = str(df_key_phrases.iloc[i])

    # Векторизация данных (ключевых фраз !!!!!)
    # Векторизация текстов с помощью TF-IDF на основе ключевых фраз
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_key_phrases)

    return tfidf_matrix, df_key_phrases

# Предобработка данных
X_vectorized, X = preprocessing("dataset.xlsx")
# print(X)

# Кластеризация с помощью K-means
# df_clusterization = pd.DataFrame(columns=['df_key_phrases', 'kmeans_cluster'])
# kmeans = KMeans(n_clusters=15, random_state=42)
# df_clusterization['df_key_phrases'] = X
# df_clusterization['kmeans_cluster'] = kmeans.fit_predict(X_vectorized)
# clusters = kmeans.labels_.tolist()

# Обучение модели
print(X_vectorized.shape)
X_vectorized = X_vectorized[:, :1003]
print(X_vectorized.shape)
model = KMeans(n_clusters=15, random_state=42).fit(X_vectorized)

# Сохранение модели
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
