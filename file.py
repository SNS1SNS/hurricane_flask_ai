import pandas as pd
import pyodbc

# Пример загрузки данных в базу данных
def load_data_to_db(csv_file, conn):
    data = pd.read_csv(csv_file)
    cursor = conn.cursor()
    for _, row in data.iterrows():
        cursor.execute('''
            INSERT INTO hurricane_data (name, date, latitude, longitude, intensity, trajectory)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', row['name'], row['date'], row['latitude'], row['longitude'], row['intensity'], row['trajectory'])
    conn.commit()
