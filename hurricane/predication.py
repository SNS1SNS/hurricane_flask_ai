import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "Hurricane_Data.csv"
df = pd.read_csv(file_path)



df.dropna(inplace=True)


label_encoder = LabelEncoder()
df['CycloneCategoryEncoded'] = label_encoder.fit_transform(df['CycloneCategory'])


features = [
    "AtmosphericPressure_hPa",
    "AirTemperature_C",
    "Humidity_%",
    "WindSpeed_kmh",
    "Precipitation_mm",
    "SeaSurfaceTemperature_C",
    "OceanHeatContent_J",
    "WaveHeight_m",
    "LightningFrequency_Hz",
    "Longitude",
    "Latitude"
]
target = "CycloneCategoryEncoded"

X = df[features]
y = df[target]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()

new_data = np.array([[950, 30, 80, 150, 50, 28, 3e8, 3, 5, -80, 20]])  # Ввод новых данных
new_data = scaler.transform(new_data)
predicted_category = model.predict(new_data)
print("Predicted Cyclone Category:", label_encoder.inverse_transform(predicted_category))
