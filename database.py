import pyodbc
server = 'localhost\SQLEXPRESS'
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE=master;Trusted_Connection=yes;"

conn=pyodbc.connect(conn_str,autocommit=True)
cursor=conn.cursor()
exists=cursor.execute("select name from sys.databases where name='nova'").fetchone()
if not exists:
    cursor.execute("CREATE DATABASE nova")
    cursor.execute("USE nova")
    print(">> Created nova database.")
else:
    print("Database 'nova' already exists.")
conn.close()

conn_nova=pyodbc.connect(f"DRIVER={{ODBC DRIVER 17 FOR SQL SERVER}};SERVER={server};DATABASE=nova;Trusted_Connection=yes;")

def get_connection():
    return pyodbc.connect(
        f"DRIVER={{ODBC DRIVER 17 FOR SQL SERVER}};"
        f"SERVER={server};"
        f"DATABASE=nova;"
        f"Trusted_Connection=yes;")


