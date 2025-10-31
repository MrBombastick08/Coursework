import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import threading

model = None

# Train loader
def load_train():
    global train_path
    train_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if train_path:
        messagebox.showinfo("Успешно", "Train загружен!")

# Train model
def train_model():
    global model, train_path
    if not train_path:
        messagebox.showerror("Ошибка", "Сначала загрузите train!")
        return

    df = pd.read_csv(train_path)
    X = df['comment']
    y = df['toxic']

    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=2000))
    ])

    model.fit(X, y)

    joblib.dump(model, "model_saved.joblib")
    messagebox.showinfo("Готово", "Модель обучена и сохранена!")

# Test labeling
def label_test():
    global model
    if model is None:
        messagebox.showerror("Ошибка", "Сперва обучите модель!")
        return

    test_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not test_path:
        return

    df = pd.read_csv(test_path)

    progress.start()

    def process():
        df['toxic'] = model.predict(df['comment'])
        save_path = "test_labeled.csv"
        df.to_csv(save_path, index=False)
        progress.stop()
        messagebox.showinfo("Успех", f"Размечено и сохранено в {save_path}")

    threading.Thread(target=process).start()
# UI
root = tk.Tk()
root.title("Разметка комментариев")
root.geometry("400x260")
root.resizable(False, False)

btn_load_train = tk.Button(root, text="Загрузить TRAIN", font=("Arial", 12), command=load_train)
btn_load_train.pack(pady=10)

btn_train = tk.Button(root, text="Обучить модель", font=("Arial", 12), command=train_model)
btn_train.pack(pady=10)

btn_label = tk.Button(root, text="Разметить TEST", font=("Arial", 12), command=label_test)
btn_label.pack(pady=10)

root.mainloop()
