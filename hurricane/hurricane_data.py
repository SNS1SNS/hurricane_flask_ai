import pandas as pd
import random
import datetime

# Запрос количества записей у пользователя
num_records = int(input("Введите количество записей для генерации: "))

# Генерация данных для прогноза ураганов
data = {
    "Timestamp": [datetime.datetime(2024, 12, 4, 0, 0) + datetime.timedelta(hours=i) for i in range(num_records)],  # Временные метки
    "AtmosphericPressure_hPa": [random.uniform(900, 1013) for _ in range(num_records)],  # Атмосферное давление
    "AirTemperature_C": [random.uniform(25, 35) for _ in range(num_records)],  # Температура воздуха
    "Humidity_%": [random.uniform(60, 100) for _ in range(num_records)],  # Влажность
    "WindSpeed_kmh": [random.uniform(50, 200) for _ in range(num_records)],  # Скорость ветра
    "Precipitation_mm": [random.uniform(0, 200) for _ in range(num_records)],  # Осадки
    "SeaSurfaceTemperature_C": [random.uniform(26, 31) for _ in range(num_records)],  # Температура поверхности моря
    "OceanHeatContent_J": [random.uniform(1e8, 5e8) for _ in range(num_records)],  # Тепловой запас океана
    "WaveHeight_m": [random.uniform(0.5, 5) for _ in range(num_records)],  # Высота волн
    "LightningFrequency_Hz": [random.uniform(0, 10) for _ in range(num_records)],  # Частота молний
    "Longitude": [random.uniform(-100, -60) for _ in range(num_records)],  # Долгота
    "Latitude": [random.uniform(10, 30) for _ in range(num_records)],  # Широта
    "CycloneCategory": [random.choice(["Tropical Depression", "Tropical Storm", "Hurricane Category 1",
                                       "Hurricane Category 2", "Hurricane Category 3",
                                       "Hurricane Category 4", "Hurricane Category 5"]) for _ in range(num_records)]  # Категория урагана
}

# Создание DataFrame
df = pd.DataFrame(data)

# Сохранение в CSV файл
file_path = "Hurricane_Data.csv"
df.to_csv(file_path, index=False)

print(f"CSV файл создан: {file_path}")
