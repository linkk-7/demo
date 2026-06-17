import random
import socket
import time
import threading
from configparser import ConfigParser
from utils.byte_utils import get_length_prefix_bytes
from utils.ini_utils import load_config
import traceback  # 添加导入
import signal  # 添加导入
import sys     # 添加导入


class TcpClient():
    count: int = 0
    send_thread: threading.Thread = None  #发送线程
    send_thread2: threading.Thread = None  #第二个发送线程
    send_work_status: bool = True         #发送线程的工作状态    
    need_send_thread2: bool = False      #是否需要第二个发送线程
    config: ConfigParser
    client_socket: socket.socket      #tcp socket client
    def __init__(self, config_path: str = "./config/socket.ini") -> None:
        """
        初始化客户端，并设置信号处理
        """
        print("config_path", config_path)
        config = load_config(config_path)
        self.config = config

        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        """
        捕获 Ctrl+C 信号并优雅退出
        """
        print("\n捕获到中断信号，正在退出...")
        self.send_work_status = False  # 停止发送线程
        if self.client_socket:
            self.client_socket.close()  # 关闭 socket
        sys.exit(0)  # 退出程序

    def run(self) -> None:
        """
        启动客户端
        """
        # 定义服务器的地址和端口
        server_host = self.config.get('server', 'host')       # 服务器 IP 地址
        server_port = self.config.getint('server', 'port')    # 服务器端口
        while True:
            # 创建一个 TCP/IP socket
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # 尝试连接到服务器
                self.client_socket.connect((server_host, server_port))
                print(f"已连接到服务器 {server_host}:{server_port}")


                # 发送注册包到服务器上
                message = f"RG{self.config.get('server', 'imei')}MONIX"
                self.client_socket.sendall(get_length_prefix_bytes(message.encode('utf-8')))
                print(f"已发送消息: {message}")



                if self.config.getint('server', 'need_auto_send') == 1:  #需要自动推送
                    self.send_thread = threading.Thread(target=self.send_messages, args=())
                    self.send_thread.daemon = True
                    self.send_thread.start()
                    if self.need_send_thread2:
                        self.send_thread2 = threading.Thread(target=self.send_messages2, args=())
                        self.send_thread2.daemon = True
                        self.send_thread2.start()   

                # 启动一个线程来发送心跳包
                time.sleep(1)
                heartbeat_thread = threading.Thread(target=self.send_heartbeat, args=())
                heartbeat_thread.daemon = True
                heartbeat_thread.start()                     

                # 启动一个线程来接收回调数据
                recv_thread = threading.Thread(target=self.receive_messages, args=())
                recv_thread.daemon = True
                recv_thread.start()
            
                # 等待接收消息线程结束（服务器断开连接时会退出）
                recv_thread.join()


            except (ConnectionRefusedError, socket.error) as e:
                print(f"连接失败: {e}")
            finally:
                # 关闭连接并等待 1 分钟重试
                self.client_socket.close()
                print(f"连接已关闭，{self.config.getint('time', 'reconnect_time')}秒后重试...")
                time.sleep(self.config.getint("time", 'reconnect_time'))  # 重试连接


    def start_send_thread(self):
        """
        开启发送线程
        """
        self.send_work_status = True
        if not self.send_thread or not self.send_thread.is_alive():
            self.send_thread = threading.Thread(target=self.send_messages, args=())
            self.send_thread.daemon = True
            self.send_thread.start()
        if self.need_send_thread2 and (not self.send_thread2 or not self.send_thread2.is_alive()):
            self.send_thread2 = threading.Thread(target=self.send_messages2, args=())
            self.send_thread2.daemon = True
            self.send_thread2.start()

    def stop_send_thread(self):
        """
        停止发送线程
        """  
        self.send_work_status = False     

    def send_heartbeat(self):
        """每隔 固定时间 发送一次心跳包 并且检查发送线程，如果采集箱处于工作状态且发送线程断开，则自动启动发送线程"""
        try:
            self.count += 1
            while True:
                heartbeat_message = f"HB{self.config.get('server', 'imei')}MONIX"
                self.client_socket.sendall(get_length_prefix_bytes(heartbeat_message.encode('utf-8')))
                print(f"已发送心跳包{self.count}: {heartbeat_message}")
                #check send_message thread
                if self.config.getint('server', 'need_auto_send') == 1:
                    if (self.send_thread is not None and not self.send_thread.is_alive()) or (self.need_send_thread2 and self.send_thread2 is not None and not self.send_thread2.is_alive()):
                        self.start_send_thread()                     
                time.sleep(self.config.getint('time', 'heartbeat_time')) 


        except Exception as e:
            print(f"发送心跳包失败: {e}")
            return
        
    def send_messages(self):
        """每隔 固定时间 推送一次"""
        try:
            self.count += 1
            while True:
                if self.send_work_status:
                    ##########################主动推送消息逻辑start##################################           
                    self.handle_send_messages()
                    ###########################主动推送消息逻辑end################################### 
                time.sleep(self.config.getint('time', 'send_time')) 
        except Exception as e:
            print(f"主动推送消息失败: {e}")
        finally:
            print("推送线程终止!")

    def send_messages2(self):
        """每隔 固定时间 推送一次"""
        try:
            self.count += 1
            while True:
                if self.send_work_status:
                    ##########################主动推送消息逻辑start##################################           
                    self.handle_send_messages2()
                    ###########################主动推送消息逻辑end################################### 
                time.sleep(self.config.getint('time', 'send_time')) 
        except Exception as e:
            print(f"主动推送消息失败2: {e}")
        finally:
            print("推送线程终止!")
        
    def receive_messages(self):
        """接收并处理来自服务器的消息"""
        try:
            while True:
                data = self.client_socket.recv(1024)
                if not data:
                    print("服务器断开连接")
                    break
                ##########################处理回调消息逻辑start##################################
                self.handle_recv_messages(data)
                ###########################处理回调消息逻辑end###################################        
        except Exception as e:
            print(f"接收消息时出错: {e}")
            traceback.print_exc()  # 打印完整的错误堆栈信息
        finally:
            print("接收线程终止!")

    def handle_send_messages2(self):
        """
        发送给服务器的消息的具体逻辑
       
        '''''''''''''''''
        '继承后需要被重写'
        '''''''''''''''''

        """
        print("执行发送逻辑")

    def handle_send_messages(self):
        """
        发送给服务器的消息的具体逻辑
       
        '''''''''''''''''
        '继承后需要被重写'
        '''''''''''''''''

        """
        print("执行发送逻辑")
        res_bytes = str(random.random()).encode('utf-8')
        print(f"定时推送消息 deform1, 长度: {len(res_bytes)}")
        self.client_socket.sendall(get_length_prefix_bytes(res_bytes))

    def handle_recv_messages(self, data: bytes):
        """
        接收并处理来自服务器的消息的具体逻辑
       
        '''''''''''''''''
        '继承后需要被重写'
        '''''''''''''''''

        """
        print(f"接收到来自服务器的消息: {data}")
        print(f"接收到来自服务器的消息解码: {data.hex()} {data.hex()[0:2]}")

        res_bytes = ("RG" + self.config.get('server', 'imei')  + "MONIX" + str(data.hex()[0:2]) + "+" +str(random.random())).encode('utf-8')

        self.client_socket.sendall(get_length_prefix_bytes(res_bytes))


if __name__ == "__main__":
    tcp_client = TcpClient()
    tcp_client.run()
