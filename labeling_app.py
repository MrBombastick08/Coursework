import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import spacy
import os

# --- spaCy setup ---
# Перед запуском этого скрипта, убедитесь, что вы установили spaCy и русскую модель:
# pip install spacy
# python -m spacy download ru_core_news_sm

try:
    nlp = spacy.load("ru_core_news_sm")
except OSError:
    print("Ошибка: Модель spaCy 'ru_core_news_sm' не найдена.")
    print("Пожалуйста, установите ее, выполнив команду: python -m spacy download ru_core_news_sm")
    exit()

def preprocess_text(text):
    # Убедимся, что на вход подается строка
    text = str(text)
    # Обработка текста с помощью spaCy
    doc = nlp(text)
    # Лемматизация, удаление стоп-слов и пунктуации
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return ' '.join(tokens)

def auto_label_comments(train_file='train.csv', test_file='test.csv', output_file='test_labeled.csv'):
    print(f"Загрузка данных из {train_file} и {test_file}...")
    try:
        df_train = pd.read_csv(train_file)
    except FileNotFoundError:
        print(f"Ошибка: {train_file} не найден.")
        return

    try:
        df_test = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Ошибка: {test_file} не найден.")
        return

    if 'comment' not in df_train.columns or 'toxic' not in df_train.columns:
        print(f"Ошибка: {train_file} не содержит нужных колонок ('comment', 'toxic').")
        return
    if 'comment' not in df_test.columns:
        print(f"Ошибка: {test_file} не содержит колонку 'comment'.")
        return
    
    # Обработка пропущенных значений в колонке 'comment'
    df_train['comment'] = df_train['comment'].fillna('')
    df_test['comment'] = df_test['comment'].fillna('')

    print("Предобработка текста (может занять некоторое время)...")
    df_train['processed_comment'] = df_train['comment'].apply(preprocess_text)
    df_test['processed_comment'] = df_test['comment'].apply(preprocess_text)

    print("Обучение модели...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(df_train['processed_comment'])
    y_train = df_train['toxic']

    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    print("Модель обучена успешно.")

    print("Автоматическая разметка комментариев test.csv...")
    X_test_tfidf = vectorizer.transform(df_test['processed_comment'])
    predictions = model.predict(X_test_tfidf)
    df_test['toxic'] = predictions

    df_test[['comment', 'toxic']].to_csv(output_file, index=False, encoding='utf-8')
    print(f"Автоматическая разметка завершена. Результаты сохранены в {output_file}")

if __name__ == '__main__':
    auto_label_comments()

