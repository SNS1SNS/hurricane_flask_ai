import sqlite3
import customtkinter as ctk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

# Настройка базы данных
def setup_database():
    conn = sqlite3.connect("hurricane_app.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

# Функция для регистрации пользователя
def register_user(username, password):
    conn = sqlite3.connect("hurricane_app.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        messagebox.showinfo("Success", "Registration successful!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Username already exists.")
    conn.close()

# Функция для входа пользователя
def login_user(username, password):
    conn = sqlite3.connect("hurricane_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Загрузка и обработка данных
def load_and_train_model():
    df = pd.read_csv("Hurricane_Data.csv")
    label_encoder = LabelEncoder()
    df['CycloneCategoryEncoded'] = label_encoder.fit_transform(df['CycloneCategory'])
    features = [
        "AtmosphericPressure_hPa", "AirTemperature_C", "Humidity_%",
        "WindSpeed_kmh", "Precipitation_mm", "SeaSurfaceTemperature_C",
        "OceanHeatContent_J", "WaveHeight_m", "LightningFrequency_Hz",
        "Longitude", "Latitude"
    ]
    target = "CycloneCategoryEncoded"
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, label_encoder

# Главное приложение
class HurricaneApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Hurricane Prediction App")
        self.root.geometry("500x700")
        self.root.configure(fg_color="#f4f4f9")
        self.model, self.scaler, self.label_encoder = load_and_train_model()
        self.show_login_page()

    def show_login_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # Заголовок
        header = ctk.CTkLabel(self.root, text="Login Form", font=("Helvetica", 20, "bold"))
        header.pack(pady=30)

        # Логин форма
        frame = ctk.CTkFrame(self.root, corner_radius=10)
        frame.pack(pady=20, padx=20)

        ctk.CTkLabel(frame, text="Username", font=("Helvetica", 14)).grid(row=0, column=0, pady=10, padx=10)
        self.login_username = ctk.CTkEntry(frame, width=200, font=("Helvetica", 14))
        self.login_username.grid(row=0, column=1, pady=10, padx=10)

        ctk.CTkLabel(frame, text="Password", font=("Helvetica", 14)).grid(row=1, column=0, pady=10, padx=10)
        self.login_password = ctk.CTkEntry(frame, show="*", width=200, font=("Helvetica", 14))
        self.login_password.grid(row=1, column=1, pady=10, padx=10)

        ctk.CTkButton(frame, text="Log in", command=self.login, width=150).grid(row=2, columnspan=2, pady=20)
        ctk.CTkButton(frame, text="Register", command=self.show_registration_page, width=150, fg_color="#6c757d").grid(row=3, columnspan=2, pady=10)

    def show_registration_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # Заголовок
        header = ctk.CTkLabel(self.root, text="Register Form", font=("Helvetica", 20, "bold"))
        header.pack(pady=30)

        # Регистрация форма
        frame = ctk.CTkFrame(self.root, corner_radius=10)
        frame.pack(pady=20, padx=20)

        ctk.CTkLabel(frame, text="Username", font=("Helvetica", 14)).grid(row=0, column=0, pady=10, padx=10)
        self.reg_username = ctk.CTkEntry(frame, width=200, font=("Helvetica", 14))
        self.reg_username.grid(row=0, column=1, pady=10, padx=10)

        ctk.CTkLabel(frame, text="Password", font=("Helvetica", 14)).grid(row=1, column=0, pady=10, padx=10)
        self.reg_password = ctk.CTkEntry(frame, show="*", width=200, font=("Helvetica", 14))
        self.reg_password.grid(row=1, column=1, pady=10, padx=10)

        ctk.CTkButton(frame, text="Register", command=self.register, width=150).grid(row=2, columnspan=2, pady=20)
        ctk.CTkButton(frame, text="Back to Login", command=self.show_login_page, width=150, fg_color="#6c757d").grid(row=3, columnspan=2, pady=10)

    def show_main_page(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        # Заголовок
        header = ctk.CTkLabel(self.root, text="Hurricane Prediction", font=("Helvetica", 20, "bold"))
        header.pack(pady=30)

        frame = ctk.CTkFrame(self.root, corner_radius=10)
        frame.pack(pady=20, padx=20)

        self.input_features = {}
        for i, feature in enumerate([
            "AtmosphericPressure_hPa", "AirTemperature_C", "Humidity_%",
            "WindSpeed_kmh", "Precipitation_mm", "SeaSurfaceTemperature_C",
            "OceanHeatContent_J", "WaveHeight_m", "LightningFrequency_Hz",
            "Longitude", "Latitude"
        ]):
            ctk.CTkLabel(frame, text=feature, font=("Helvetica", 14)).grid(row=i, column=0, pady=5, padx=10)
            self.input_features[feature] = ctk.CTkEntry(frame, width=200, font=("Helvetica", 14))
            self.input_features[feature].grid(row=i, column=1, pady=5, padx=10)

        ctk.CTkButton(frame, text="Predict", command=self.predict, width=150).grid(row=len(self.input_features), columnspan=2, pady=20)

    def login(self):
        username = self.login_username.get()
        password = self.login_password.get()
        if login_user(username, password):
            messagebox.showinfo("Success", "Login successful!")
            self.show_main_page()
        else:
            messagebox.showerror("Error", "Invalid username or password.")

    def register(self):
        username = self.reg_username.get()
        password = self.reg_password.get()
        register_user(username, password)

    def predict(self):
        try:
            features = [float(self.input_features[feature].get()) for feature in self.input_features]
            scaled_features = self.scaler.transform([features])
            prediction = self.model.predict(scaled_features)
            category = self.label_encoder.inverse_transform(prediction)
            messagebox.showinfo("Prediction", f"Predicted Hurricane Category: {category[0]}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def run(self):
        self.root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    setup_database()
    app = HurricaneApp()
    app.run()
