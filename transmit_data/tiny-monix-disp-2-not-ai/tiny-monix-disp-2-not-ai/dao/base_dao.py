import sqlite3
from typing import List, TypeVar, Generic
from pydantic import BaseModel
from utils.sqlite_utils import execute_sql, find_all, generate_connection, generate_cursor


global_conn = generate_connection("data.db", False)
global_cur = generate_cursor(global_conn)



# 定义一个泛型类型变量
T = TypeVar('T', bound=BaseModel)
class BaseDao(Generic[T]):
    """
    python与sqlite交互的封装类
    table_name: sqlite数据库的表名
    table_class: model类的类型
    """
    table_name: str
    table_class: type[T]
    def __init__(self, table_name: str, table_class: type[T], conn: sqlite3.Connection, cur: sqlite3.Cursor):
        self.table_name = table_name
        self.table_class = table_class
        self.conn = conn
        self.cur = cur
        pass

    def find_all(self) -> List[T]:
        """
        查找所有对象
        """
        columns = [column[1] for column in find_all(self.cur, f"PRAGMA table_info({self.table_name})")]  # 列名列表
        datas = find_all(self.cur, f"select * from {self.table_name};")
        datas = [self.table_class(**{columns[i]: data[i] for i in range(len(data)) }) for data in datas]
        return datas
    
    def find_by_id(self, id: int) -> T:
        """
        查找对应id的对象
        """
        columns = [column[1] for column in find_all(self.cur, f"PRAGMA table_info({self.table_name})")]  # 列名列表
        datas = find_all(self.cur, f"select * from {self.table_name} where id = {id};")
        datas = [self.table_class(**{columns[i]: data[i] for i in range(len(data)) }) for data in datas]
        if len(datas) > 0:
            return datas[0]
        return None
    
    def save(self, data: T) -> None:
        """
        插入一条monitor
        """
        monitor_dict = data.model_dump()
        keys = monitor_dict.keys()
        need_insert_keys = []
        need_insert_values = []
        for key in keys:
            value = monitor_dict.get(key)
            if value != None:
                need_insert_keys.append(key)
                if type(value) == bool:   
                    if value == True:
                        need_insert_values.append("1")
                    else:
                        need_insert_values.append("0")
                elif type(value) == str:  
                    need_insert_values.append("'" + value + "'")
                else:       
                    need_insert_values.append(str(value))
        sql = f"INSERT OR REPLACE INTO {self.table_name} (" + ", ".join(need_insert_keys) + ") VALUES (" + ", ".join(need_insert_values) + ")"
        execute_sql(self.cur, sql)
        self.conn.commit()

    def save_all(self, datas: List[T]) -> None:
        """
        插入多条monitor
        """
        for data in datas:
            monitor_dict = data.model_dump()
            keys = monitor_dict.keys()
            need_insert_keys = []
            need_insert_values = []
            for key in keys:
                value = monitor_dict.get(key)
                if value != None:
                    need_insert_keys.append(key)
                    if type(value) == bool:   
                        if value == True:
                            need_insert_values.append("1")
                        else:
                            need_insert_values.append("0")
                    elif type(value) == str:  
                        need_insert_values.append("'" + value + "'")
                    else:       
                        need_insert_values.append(str(value))
            sql = f"INSERT OR REPLACE INTO {self.table_name} (" + ", ".join(need_insert_keys) + ") VALUES (" + ", ".join(need_insert_values) + ")"
            execute_sql(self.cur, sql)
        self.conn.commit()



