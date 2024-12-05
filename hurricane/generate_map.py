from tkinter import messagebox

import folium
import pandas as pd
def generate_map(self):
    df = pd.read_csv("Hurricane_Data.csv")
    hurricane_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=5)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Category: {row['CycloneCategory']}, Wind Speed: {row['WindSpeed_kmh']} km/h"
        ).add_to(hurricane_map)

    hurricane_map.save("hurricane_map.html")
    messagebox.showinfo("Map Generated", "Map has been saved as hurricane_map.html. Open it in your browser.")
