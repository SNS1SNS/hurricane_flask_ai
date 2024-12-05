import matplotlib.pyplot as plt
import pandas as pd

def generate_analysis_graphs():
    # Загрузка данных
    df = pd.read_csv("hurricane/Hurricane_Data.csv")


    plt.figure(figsize=(8, 6))
    df['CycloneCategory'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Cyclone Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.savefig('static/images/cyclone_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df['AirTemperature_C'], df['WindSpeed_kmh'], color='red', alpha=0.5)
    plt.title('Air Temperature vs Wind Speed')
    plt.xlabel('Air Temperature (°C)')
    plt.ylabel('Wind Speed (km/h)')
    plt.savefig('static/images/temp_vs_wind_speed.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(df['Precipitation_mm'], bins=20, color='green', alpha=0.7)
    plt.title('Precipitation Distribution')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Frequency')
    plt.savefig('static/images/precipitation_histogram.png')
    plt.close()
