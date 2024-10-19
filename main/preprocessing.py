import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

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
    nltk.download('punkt_tab')
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
                df = pd.DataFrame()

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
        # print(type(df_key_phrases.iloc[i]))
        df_key_phrases.iloc[i] = str(df_key_phrases.iloc[i])

    # Векторизация данных (ключевых фраз !!!!!)
    # Векторизация текстов с помощью TF-IDF на основе ключевых фраз
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_key_phrases)
    return tfidf_matrix
    # # Предобработка данных
    # for i in range(df.shape[0]):
    #     # Приводим текст к нижнему регистру
    #     string = str(df.iloc[i][0])
    #     string = string.lower()
    #     # print("string", string)

    #     # Уберём неинформативные данные (оставим только текст)
    #     string = re.sub("([^0-9А-ЯЁа-яё \t])|(\w+:\/\/\S+)", "", string)
    #     # print(f"удаление неинформативных данных\n: {string}")

    #     # Токенизируем текст
    #     string = word_tokenize(string)
    #     # print(f"Токенизация текста\n: {string}")

    #     # Удалим стоп-слова
    #     string_withoutstop = [word for word in string if word not in stopwords_rus]
    #     # print(f"Удаление стоп-слов\n: {string_withoutstop}")

    #     # Лемматизируем (приведем к исходной форме) слова
    #     string = [WordNetLemmatizer().lemmatize(word) for word in string_withoutstop]
    #     # print(f"Лемматизация текста\n: {string}")

    #     df.iloc[i, 0] = str(string)

    #     # удаление имен собственных

    # # Векторизуем
    # vectorizer = CountVectorizer(ngram_range=(1, 3))
    
    # X = vectorizer.fit_transform(df.iloc[:][0])
    # return X

def predict_question(X, model):
    # Используем модель для предсказания ключевых слов
    # key_terms - предсказанные кластеры для каждого ответа 
    key_terms = pd.DataFrame(model.predict(X))
    return key_terms

def get_dict_of_key_terms(key_terms):
    # Создаём ранжированный словарь
    ranged_dict = dict(Counter(key_terms[0].values))
    list_ley_terms = []
    for key in ranged_dict.keys():
        list_ley_terms.append(key)
    return list_ley_terms

def get_frequencies(key_terms):
    # Создаём ранжированный словарь
    ranged_dict = dict(Counter(key_terms[0].values))
    frequency = []
    for key in ranged_dict.keys():
        percent = (ranged_dict.get(key) / sum(ranged_dict.values())) * 100
        frequency.append(percent)
    return frequency
