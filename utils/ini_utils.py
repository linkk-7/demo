"""
ini配置文件的读写工具函数
"""
import configparser

def load_config(path: str) -> configparser.ConfigParser:
    # 创建 ConfigParser 对象
    config = configparser.ConfigParser()
    # 读取 config.ini 文件
    config.read(path, encoding='utf-8') 
    return config
