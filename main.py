from flask import Flask, request, render_template, redirect, url_for, session
import pyodbc
import re
import hashlib

app = Flask(__name__)

app.config['SQL_SERVER'] = 'DESKTOP-UHS08AP\MSSQLSERVER01'  # Имя сервера
app.config['SQL_DATABASE'] = 'pythonlogin'  # Имя базы данных
app.config['SQL_DRIVER'] = '{ODBC Driver 17 for SQL Server}'  # Драйвер ODBC

# Функция для подключения к базе данных
def get_db_connection():
    return pyodbc.connect(
        f"DRIVER={app.config['SQL_DRIVER']};"
        f"SERVER={app.config['SQL_SERVER']};"
        f"DATABASE={app.config['SQL_DATABASE']};"
        f"Trusted_Connection=yes;"
    )
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    return render_template('index.html', msg='')

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
    elif request.method == 'POST':
        msg = 'Please fill out the form!'
    return render_template('register.html', msg=msg)
@app.route('/pythonlogin/home')
def home():
    # Check if the user is logged in
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))