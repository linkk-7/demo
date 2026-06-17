
from influxdb import InfluxDBClient
from tcp_server import start_tcp_server
import configparser
import os
import threading
import signal  # 添加导入
import sys     # 添加导入
from dh_request import send_login_req,send_get_device_tree,send_get_all_points
from tcp_client import TcpClient
from schedule import get_previous_minute_data

# 标记程序是否运行中
running = True
serverSocket = None  # 提前声明以便退出时关闭

def signal_handler(sig, frame):
    global running, serverSocket
    print("收到退出信号，清理资源...")
    running = False
    if serverSocket:
        serverSocket.close()
    if client:
        client.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  # 捕捉 Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 捕捉 kill

# 初始化客户端（只执行一次）
client = InfluxDBClient(host='localhost', port=8086, database='dh')
if 'dh' not in [db['name'] for db in client.get_list_database()]:
    client.create_database('dh')
    
import tcp_server
tcp_server.client = client
tcp_server.running = running

import tcp_client
tcp_client.client = client


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(os.path.join("config", "config.ini"), encoding="utf-8")
    
   

    tcp_client = TcpClient()
    tcp_client.run()