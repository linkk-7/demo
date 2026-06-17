import socket
import threading
import time

# 服务端配置
HOST = '127.0.0.1'  # 监听的 IP 地址
PORT = 12345  # 监听的端口号
TIMEOUT = 5  # 定时发送消息的时间间隔（秒）
MESSAGE = b"""
{
    "type": "calibration",
    "param_id": 7
}
"""  # 定时发送的特定消息

# 存储所有已连接的客户端套接字
connected_clients = []

# 处理客户端连接的函数
def handle_client(client_socket, client_address):
    print(f'New connection from {client_address}')
    connected_clients.append(client_socket)
    try:
        while True:
            # 接收客户端发送的消息
            data = client_socket.recv(1024)
            if not data:
                break
            print(f'Received from {client_address}: {data.decode()}')
    except Exception as e:
        print(f'Error handling client {client_address}: {e}')
    finally:
        # 客户端断开连接时，从列表中移除该客户端套接字
        connected_clients.remove(client_socket)
        client_socket.close()
        print(f'Connection with {client_address} closed')

# 定时发送消息的函数
def send_timed_messages():
    while True:
        time.sleep(TIMEOUT)
        for client in connected_clients:
            try:
                # 向每个已连接的客户端发送特定消息
                client.sendall(MESSAGE)
            except Exception as e:
                print(f'Error sending message to client: {e}')

# 创建 TCP 套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print(f'Server listening on {HOST}:{PORT}')

# 启动定时发送消息的线程
timer_thread = threading.Thread(target=send_timed_messages)
timer_thread.daemon = True
timer_thread.start()

try:
    while True:
        # 接受客户端连接
        client_socket, client_address = server_socket.accept()
        # 为每个客户端创建一个新的线程来处理连接
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_thread.start()
except KeyboardInterrupt:
    print('Server shutting down...')
finally:
    # 关闭服务端套接字
    server_socket.close()