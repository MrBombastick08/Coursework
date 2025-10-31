import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

class LabelingApp:
    def __init__(self, root):
        self.root = root
        root.title("Auto Labeling App")
        root.geometry("420x220")
        root.resizable(False, False)  # Окно фиксированного размера

        # Состояние
        self.model = None
        self.train_path = None

        # Кнопки
        self.btn_load = tk.Button(root, text="Загрузить TRAIN", width=30, command=self.load_train)
        self.btn_load.pack(pady=(20, 8))

        self.btn_train = tk.Button(root, text="Обучить модель", width=30, command=self.train_model_thread)
        self.btn_train.pack(pady=8)

        self.btn_label = tk.Button(root, text="Разметить TEST", width=30, command=self.label_test_thread)
        self.btn_label.pack(pady=8)

        # Небольшой лейбл-статус
        self.status_var = tk.StringVar(value="Статус: ожидание действий")
        self.status_label = tk.Label(root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=(12,0))
    
    # Загрузка train файла
    def load_train(self):
        path = filedialog.askopenfilename(title="Выберите train CSV", filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except Exception as e:
            messagebox.showerror("Ошибка чтения файла", f"Не удалось прочитать файл:\n{e}")
            return

        # Проверка колонок
        if 'comment' not in df.columns or 'toxic' not in df.columns:
            messagebox.showerror("Неверный формат", "Файл train должен содержать столбцы: 'comment' и 'toxic'")
            return

        # Сохраним путь и покажем короткую инфу
        self.train_path = path
        cnt = len(df)
        pos = int((df['toxic']==0).sum())
        neg = int((df['toxic']==1).sum())
        self.status_var.set(f"Train загружен: {os.path.basename(path)} ({cnt} строк, 0:{pos}, 1:{neg})")
        messagebox.showinfo("Train загружен", f"Файл загружен: {os.path.basename(path)}\nСтрок: {cnt}\n0: {pos}, 1: {neg}")

    # Обучение (в отдельном потоке)
    def train_model_thread(self):
        # Запуск в отдельном потоке, чтобы не блокировать UI
        thread = threading.Thread(target=self.train_model, daemon=True)
        thread.start()

    def train_model(self):
        if not self.train_path:
            messagebox.showerror("Ошибка", "Сначала загрузите train (кнопка 'Загрузить TRAIN').")
            return

        # Блокируем кнопки на время обучения
        self._set_buttons_state("disabled")
        self.status_var.set("Статус: обучение модели...")

        try:
            df = pd.read_csv(self.train_path, encoding='utf-8')
            X = df['comment'].fillna('').astype(str)
            y = df['toxic']

            #TF-IDF + LogisticRegression
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', LogisticRegression(max_iter=1000))
            ])

            pipeline.fit(X, y)
            self.model = pipeline

            # Сохраняем модель рядом с train файлом
            save_path = os.path.join(os.path.dirname(self.train_path), "model_saved.joblib")
            joblib.dump(self.model, save_path)

            self.status_var.set(f"Статус: модель обучена и сохранена ({os.path.basename(save_path)})")
            messagebox.showinfo("Успех", f"Модель обучена и сохранена:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка обучения", str(e))
            self.status_var.set("Статус: ошибка при обучении")
        finally:
            self._set_buttons_state("normal")

    # Разметка теста (в отдельном потоке)
    def label_test_thread(self):
        thread = threading.Thread(target=self.label_test, daemon=True)
        thread.start()

    def label_test(self):
        # Проверки
        if self.model is None:
            # Попробуем загрузить модель из папки train, если train_path есть
            if self.train_path:
                alt_path = os.path.join(os.path.dirname(self.train_path), "model_saved.joblib")
                if os.path.exists(alt_path):
                    try:
                        self.model = joblib.load(alt_path)
                    except Exception:
                        self.model = None

            if self.model is None:
                messagebox.showerror("Ошибка", "Модель не обучена. Сначала обучите модель (кнопка 'Обучить модель').")
                return

        test_path = filedialog.askopenfilename(title="Выберите test CSV", filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if not test_path:
            return

        try:
            df_test = pd.read_csv(test_path, encoding='utf-8')
        except Exception as e:
            messagebox.showerror("Ошибка чтения файла", f"Не удалось прочитать test-файл:\n{e}")
            return

        if 'comment' not in df_test.columns:
            messagebox.showerror("Неверный формат", "Файл test должен содержать столбец: 'comment'")
            return

        # Блокируем кнопки на время прогнозирования
        self._set_buttons_state("disabled")
        self.status_var.set("Статус: разметка теста...")

        try:
            X_test = df_test['comment'].fillna('').astype(str)
            df_test['toxic'] = self.model.predict(X_test)

            # Сохраняем файл (предлагаем путь для сохранения)
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="test_labeled.csv",
                                                     filetypes=[("CSV files","*.csv"), ("All files","*.*")])
            if not save_path:
                # если пользователь отменил сохранение, всё же предложим сохранить рядом с test
                fallback = os.path.splitext(test_path)[0] + "_labeled.csv"
                df_test.to_csv(fallback, index=False, encoding='utf-8')
                self.status_var.set(f"Статус: сохранено (fallback) {os.path.basename(fallback)}")
                messagebox.showinfo("Сохранено", f"Файл сохранён: {fallback}")
            else:
                df_test.to_csv(save_path, index=False, encoding='utf-8')
                self.status_var.set(f"Статус: разметка завершена и сохранена ({os.path.basename(save_path)})")
                messagebox.showinfo("Готово", f"Разметка завершена. Файл сохранён:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Ошибка разметки", str(e))
            self.status_var.set("Статус: ошибка при разметке")
        finally:
            self._set_buttons_state("normal")

    # Утилиты
    def _set_buttons_state(self, state: str):
        """Установить состояние всех кнопок: 'normal' или 'disabled'"""
        try:
            self.btn_load.config(state=state)
            self.btn_train.config(state=state)
            self.btn_label.config(state=state)
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
