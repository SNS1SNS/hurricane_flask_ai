import pyodbc

try:
    conn = pyodbc.connect(
        r"DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=localhost\MSSQLSERVER01;"
        r"DATABASE=pythonlogin;"
        r"Trusted_Connection=yes;"
    )
    print("Соединение с базой данных установлено.")
    QUERY = "SELECT * FROM accounts;"
    cursor = conn.cursor()
    cursor.execute(QUERY)
    for row in cursor.fetchall():
        print(row)
except pyodbc.Error as db_err:
    print(f"Ошибка базы данных: {db_err}")
except Exception as e:
    print(f"Общая ошибка: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
