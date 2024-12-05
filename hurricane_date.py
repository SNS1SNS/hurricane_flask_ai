import pyodbc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
conn = pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=localhost\MSSQLSERVER01;"
        r"DATABASE=pythonlogin;"
        r"Trusted_Connection=yes;"
    )
def get_hurricane_data(conn):
    query = "SELECT latitude, longitude, intensity FROM hurricane_data"
    return pd.read_sql(query, conn)

# Обучение модели
data = get_hurricane_data(conn)
X = data[['latitude', 'longitude']]
y = data['intensity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
predictions = model.predict(X_test)
