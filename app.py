import os
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, send_file
import pyodbc
import hashlib
import re
import folium
from prophet import Prophet
from folium.plugins import PolyLineTextPath

from hurricane.predication import scaler, model, label_encoder
from hurricane.show import generate_analysis_graphs

app = Flask(__name__)

app.secret_key = 'secret_key'




def get_db_connection():
    try:
        conn = pyodbc.connect(
            r"DRIVER={ODBC Driver 17 for SQL Server};"
            r"SERVER=localhost\MSSQLSERVER01;"
            r"DATABASE=pythonlogin;"
            r"Trusted_Connection=yes;"
        )
        print("Успешное подключение!")
        return conn
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return None  # Возвращаем None в случае ошибки

@app.route('/')
def index():
    return redirect(url_for('login'))  # Перенаправление на маршрут login

@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        hashed_password = hashlib.sha1(password.encode()).hexdigest()

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM accounts WHERE username = ? AND password = ?", (username, hashed_password))
        account = cursor.fetchone()

        if account:
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            return redirect(url_for('show_analysis_page'))
        else:
            msg = 'Неправильное имя пользователя или пароль!'

    return render_template('index.html', msg=msg)

@app.route('/pythonlogin/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))
@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
        return render_template('home.html')
    return redirect(url_for('login'))
@app.route('/dash')
def dash():
    return render_template('dashboard.html')

@app.route('/analysis', methods=['POST', 'GET'])
def show_analysis_page():
    try:
        generate_analysis_graphs()  # Генерация графиков
        return render_template('analysis.html')
    except Exception as e:
        print(f"Ошибка при генерации графиков: {e}")
        return "Произошла ошибка при генерации графиков. Проверьте данные.", 500


@app.route('/import', methods=['GET', 'POST'])
def import_data():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            upload_dir = 'uploaded'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)  # Создание директории, если её нет

            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            session['uploaded_file'] = file_path  # Сохраняем путь для дальнейшей обработки
            return redirect(url_for('show_uploaded_data'))
        else:
            return "Invalid file format. Only CSV files are allowed.", 400
    return render_template('import.html')
@app.route('/uploaded_data', methods=['GET'])
def show_uploaded_data():
    try:
        # Получение пути к загруженному файлу
        file_path = session.get('uploaded_file')
        if not file_path:
            return "No file uploaded.", 400

        # Чтение данных CSV
        df = pd.read_csv(file_path)

        # Удаление символов переноса строки
        df = df.replace(r'\n', ' ', regex=True)

        # Проверяем результат
        print(df.head())  # Отладочный вывод, можно удалить

        # Отображение таблицы
        return render_template(
            'uploaded_data.html',
            tables=[df.to_html(classes='data table-striped', index=False)],
            titles=df.columns.values
        )

    except Exception as e:
        # Логирование ошибки для отладки
        print(f"Error during file processing: {e}")
        return {"error": f"Error during file processing: {e}"}, 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        # Процесс обработки данных и предсказаний
        return {"prediction": "Category_A"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/predict_uploaded', methods=['GET', 'POST'])
def predict_uploaded_data():
    if request.method == 'POST':
        try:
            file_path = session.get('uploaded_file')
            if not file_path:
                return "No file uploaded for predictions.", 400

            # Чтение данных
            df = pd.read_csv(file_path)
            df = df.applymap(lambda x: x.replace("\n", "") if isinstance(x, str) else x)
            df.dropna(inplace=True)  # Удаляем строки с пустыми значениями
            features = df[[
                "AtmosphericPressure_hPa", "AirTemperature_C", "Humidity_%",
                "WindSpeed_kmh", "Precipitation_mm", "SeaSurfaceTemperature_C",
                "OceanHeatContent_J", "WaveHeight_m", "LightningFrequency_Hz",
                "Longitude", "Latitude"
            ]]
            scaled_features = scaler.transform(features)

            # Предсказания
            predictions = model.predict(scaled_features)
            df['Predicted_Category'] = label_encoder.inverse_transform(predictions)

            # Сохранение результата
            result_file = f"processed/predictions_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            df.to_csv(result_file, index=False)
            session['result_file'] = result_file

            return render_template('predicted_results.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

        except Exception as e:
            return {"error": f"Error during prediction: {e}"}, 400
    else:
        return "Invalid request method. Please use POST.", 405
@app.route('/download_predictions', methods=['GET'])
def download_predictions():
    result_file = session.get('result_file')
    if not result_file:
        return "No predictions available for download.", 400
    return send_file(result_file, as_attachment=True)

@app.route('/generate_predictions', methods=['POST'])
def generate_predictions():
    try:
        # Загружаем данные для предсказаний
        file_path = "Hurricane_Data.csv"
        df = pd.read_csv(file_path)

        # Фильтрация будущих данных
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        future_data = df[df['Timestamp'] > pd.Timestamp.now()]

        if future_data.empty:
            return "No future data available for predictions.", 400

        # Предсказания
        predictions = []
        for _, row in future_data.iterrows():
            input_features = np.array([[row["AtmosphericPressure_hPa"], row["AirTemperature_C"],
                                        row["Humidity_%"], row["WindSpeed_kmh"], row["Precipitation_mm"],
                                        row["SeaSurfaceTemperature_C"], row["OceanHeatContent_J"],
                                        row["WaveHeight_m"], row["LightningFrequency_Hz"],
                                        row["Longitude"], row["Latitude"]]])
            scaled_features = scaler.transform(input_features)
            predicted_category = model.predict(scaled_features)
            predicted_label = label_encoder.inverse_transform(predicted_category)[0]
            predictions.append({
                "Timestamp": row["Timestamp"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "PredictedCategory": predicted_label
            })

        # Создание DataFrame с предсказаниями
        predictions_df = pd.DataFrame(predictions)

        # Сохранение в файл
        result_file = "predictions_results.csv"
        predictions_df.to_csv(result_file, index=False)

        # Сохранение пути к файлу в сессии
        session['result_file'] = result_file

        return "Predictions generated successfully. You can download them now.", 200
    except Exception as e:
        return {"error": f"Error during prediction generation: {e}"}, 500

@app.route('/generate_map', methods=['GET'])
def generate_map_with_ml_predictions():
    """
    Генерирует карту с предсказаниями, используя данные из Hurricane_Data.csv и машинное обучение.
    """
    try:
        # Чтение данных из CSV
        file_path = "Hurricane_Data.csv"
        df = pd.read_csv(file_path)

        # Предобработка данных: удаляем строки с пустыми значениями
        df.dropna(inplace=True)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Текущее время
        now = datetime.now()

        # Фильтрация будущих данных
        future_data = df[df['Timestamp'] > now]

        if future_data.empty:
            return "Нет данных для будущих временных точек.", 400

        # Выбор 4 случайных строк из будущих данных
        sample_data = future_data.sample(4)

        # Список предсказаний для карты
        predictions = []

        # Подготовка данных для предсказания
        for _, row in sample_data.iterrows():
            input_features = np.array([[
                row["AtmosphericPressure_hPa"],
                row["AirTemperature_C"],
                row["Humidity_%"],
                row["WindSpeed_kmh"],
                row["Precipitation_mm"],
                row["SeaSurfaceTemperature_C"],
                row["OceanHeatContent_J"],
                row["WaveHeight_m"],
                row["LightningFrequency_Hz"],
                row["Longitude"],
                row["Latitude"]
            ]])

            # Масштабирование данных
            scaled_features = scaler.transform(input_features)

            # Прогнозирование категории
            predicted_category = model.predict(scaled_features)
            predicted_label = label_encoder.inverse_transform(predicted_category)[0]

            # Добавление предсказания и информации о местоположении
            predictions.append({
                "lat": row["Latitude"],
                "lon": row["Longitude"],
                "popup": f"Predicted Category: {predicted_label} ({row['Timestamp']})"
            })

        # Создание карты
        map_file = 'templates/hurricane_map.html'
        hurricane_map = folium.Map(location=[20, -80], zoom_start=5)

        # Добавление маркеров на карту
        for pred in predictions:
            folium.Marker(
                location=[pred["lat"], pred["lon"]],
                popup=pred["popup"],
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(hurricane_map)

        # Сохранение карты
        hurricane_map.save(map_file)

        # Возврат файла карты
        return send_file(map_file)

    except Exception as e:
        return {"error": f"Error during map generation: {e}"}, 400

@app.route('/predict', methods=['POST'])
def predict_hurricane():
    """
    Предсказывает мощность урагана и возвращает координаты для отображения на карте.
    """
    # Получение данных из запроса (например, JSON)
    try:
        input_data = request.json  # Ожидается JSON с ключами, соответствующими признакам
        data = np.array([[
            input_data["AtmosphericPressure_hPa"],
            input_data["AirTemperature_C"],
            input_data["Humidity_%"],
            input_data["WindSpeed_kmh"],
            input_data["Precipitation_mm"],
            input_data["SeaSurfaceTemperature_C"],
            input_data["OceanHeatContent_J"],
            input_data["WaveHeight_m"],
            input_data["LightningFrequency_Hz"],
            input_data["Longitude"],
            input_data["Latitude"],
        ]])

        # Масштабирование данных
        scaled_data = scaler.transform(data)

        # Прогнозирование
        predicted_category = model.predict(scaled_data)
        predicted_label = label_encoder.inverse_transform(predicted_category)

        # Возврат координат и мощности
        result = {
            "predicted_category": predicted_label[0],
            "coordinates": {"lat": input_data["Latitude"], "lon": input_data["Longitude"]}
        }

        return result, 200  # HTTP 200 OK

    except Exception as e:
        return {"error": f"Error during prediction: {e}"}, 400  # HTTP 400 Bad Request

@app.route('/home', methods=['GET'])
def home():
    if 'loggedin' in session and 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/pythonlogin/profile')
def profile():
    # Check if the user is logged in
    if 'loggedin' in session:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM accounts WHERE id = ?', (session['id'],))
        account = cursor.fetchone()

        if account:
            columns = [column[0] for column in cursor.description]  # Имена столбцов
            account = dict(zip(columns, account))

        conn.close()

        return render_template('profile.html', account=account)

    return redirect(url_for('login'))
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    msg = ''
    try:
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']

            print(f"Форма: {username}, {email}")

            # Проверка корректности данных
            if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = 'Неверный формат email!'
            elif not re.match(r'[A-Za-z0-9]+', username):
                msg = 'Имя пользователя должно содержать только буквы и цифры!'
            elif not username or not password or not email:
                msg = 'Пожалуйста, заполните все поля!'
            else:
                hashed_password = hashlib.sha1(password.encode()).hexdigest()

                conn = get_db_connection()
                if conn is None:
                    return "Ошибка подключения к базе данных", 500

                cursor = conn.cursor()
                cursor.execute('SELECT * FROM accounts WHERE username = ? OR email = ?', (username, email))
                account = cursor.fetchone()

                if account:
                    msg = 'Учетная запись уже существует!'
                else:
                    cursor.execute('INSERT INTO accounts (username, password, email) VALUES (?, ?, ?)',
                                   (username, hashed_password, email))
                    conn.commit()
                    msg = 'Вы успешно зарегистрировались!'
                    return redirect(url_for('login'))
    except Exception as e:
        print(f"Ошибка: {e}")
        msg = f"Ошибка: {e}"
    finally:
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except:
            pass
    return render_template('register.html', msg=msg)




@app.route('/hurricane_paths')
def visualize_hurricane_paths():
    df = pd.read_csv('Hurricane_Data.csv')
    hurricane_map = folium.Map(location=[20, -80], zoom_start=5)

    for category in df['CycloneCategory'].unique():
        subset = df[df['CycloneCategory'] == category]
        coordinates = list(zip(subset['Latitude'], subset['Longitude']))
        folium.PolyLine(coordinates, color='blue', weight=2.5, opacity=0.8).add_to(hurricane_map)

    hurricane_map.save('templates/hurricane_paths.html')
    return render_template('hurricane_paths.html')
import matplotlib.pyplot as plt

@app.route('/hurricane_intensity')
def hurricane_intensity():
    df = pd.read_csv('Hurricane_Data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['Timestamp'], df['WindSpeed_kmh'], label='Wind Speed (km/h)', color='red')
    plt.xlabel('Time')
    plt.ylabel('Wind Speed')
    plt.title('Hurricane Intensity Over Time')
    plt.legend()
    plt.savefig('static/intensity_plot.png')
    return render_template('intensity.html', image='static/intensity_plot.png')
from sklearn.ensemble import IsolationForest

@app.route('/detect_anomalies')
def detect_anomalies():
    df = pd.read_csv('Hurricane_Data.csv')
    model = IsolationForest(contamination=0.05)
    df['Anomaly'] = model.fit_predict(df[['WindSpeed_kmh', 'AtmosphericPressure_hPa']])
    anomalies = df[df['Anomaly'] == -1]
    return render_template('anomalies.html', tables=[anomalies.to_html(classes='data')])

@app.route('/forecast_trajectory')
def forecast_trajectory():
    df = pd.read_csv('Hurricane_Data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    model = Prophet()
    df.rename(columns={'Timestamp': 'ds', 'Longitude': 'y'}, inplace=True)
    model.fit(df[['ds', 'y']])
    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)
    return render_template('forecast.html', tables=[forecast.to_html(classes='data')])


# @app.route('/interactive_map')
# def interactive_map():
#     try:
#         df = pd.read_csv('Hurricane_Data.csv')
#
#         df['CycloneCategory'] = pd.to_numeric(df['CycloneCategory'], errors='coerce')
#
#         hurricane_map = folium.Map(location=[20, -80], zoom_start=5)
#
#         for _, row in df.iterrows():
#             if not pd.isna(row['CycloneCategory']):
#                 folium.Marker(
#                     location=[row['Latitude'], row['Longitude']],
#                     popup=f"Category: {row['CycloneCategory']}<br>Wind Speed: {row['WindSpeed_kmh']} km/h",
#                     icon=folium.Icon(color="red" if row['CycloneCategory'] > 3 else "blue")
#                 ).add_to(hurricane_map)
#
#         hurricane_map.save('templates/interactive_map.html')
#         return render_template('interactive_map.html')
#
#     except Exception as e:
#         return {"error": f"Error generating interactive map: {e}"}, 500
@app.route('/interactive_map')
def interactive_map():
    try:
        # Чтение данных
        file_path = 'Hurricane_Data.csv'
        if not os.path.exists(file_path):
            return {"error": "Data file not found."}, 404

        df = pd.read_csv(file_path)

        # Преобразование категории в числовой формат
        df['CycloneCategory'] = df['CycloneCategory'].str.extract('(\d+)').astype(float)

        # Создание карты
        hurricane_map = folium.Map(location=[20, -80], zoom_start=10)

        # Добавление маркеров на карту
        data_added = False
        for _, row in df.iterrows():
            if not pd.isna(row['CycloneCategory']):
                data_added = True
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Category: {int(row['CycloneCategory'])}<br>Wind Speed: {row['WindSpeed_kmh']} km/h",
                    icon=folium.Icon(color="red" if row['CycloneCategory'] > 3 else "blue")
                ).add_to(hurricane_map)

        if not data_added:
            folium.Marker(location=[20, -80], popup="No valid data available").add_to(hurricane_map)

        # Сохранение карты
        hurricane_map.save('templates/interactive_map.html')
        return render_template('interactive_map.html')

    except Exception as e:
        return {"error": f"Error generating interactive map: {e}"}, 500


@app.route('/historical_trends')
def historical_trends():
    df = pd.read_csv('Hurricane_Data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    trends = df.groupby('Year')['WindSpeed_kmh'].mean().reset_index()
    return render_template('trends.html', tables=[trends.to_html(classes='data')])


if __name__ == '__main__':
    app.run(debug=True)
    connection_test = get_db_connection()
    if connection_test:
        connection_test.close()
