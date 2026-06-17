import sqlite3
from typing import List, Any

def generate_connection(db_path: str, check_same_thread=True) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=check_same_thread)

def generate_cursor(conn: sqlite3.Connection) -> sqlite3.Cursor:
    return conn.cursor()

def execute_sql(cur: sqlite3.Cursor, sql: str) -> None:
    cur.execute(sql)

def find_all(cur: sqlite3.Cursor, sql: str) -> List[Any]:
    return cur.execute(sql).fetchall()