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
import json
import os
import math,copy
client = None
import os
from datetime import datetime

import os
from datetime import datetime
#########################获得最新的上传地址##############################
import os
from datetime import datetime, timezone

def get_latest_json_file(output_folder='./output', prefix='displacement_result_', timestamp_format='%Y-%m-%d_%H-%M-%S'):
    """
    通用函数：根据文件名前缀和时间戳格式，获取最新的 JSON 文件路径。
    
    :param output_folder: 文件夹路径
    :param prefix: 文件前缀，如 'displacement_result_' 或 'metadata_'
    :param timestamp_format: 时间戳格式，如 '%Y-%m-%d_%H-%M-%S' 或 '%Y%m%d_%H%M%S'
    :return: 最新的 JSON 文件完整路径（str）或 None
    """
    if not os.path.exists(output_folder):
        print(f"文件夹 {output_folder} 不存在!")
        return None

    files = [f for f in os.listdir(output_folder) if f.startswith(prefix) and f.endswith('.json')]
    if not files:
        print(f"没有找到以 {prefix} 开头的 JSON 文件！")
        return None

    latest_file = None
    latest_time = None

    for f in files:
        try:
            timestamp_str = f[len(prefix):-len('.json')]
            file_time = datetime.strptime(timestamp_str, timestamp_format)
        except Exception as e:
            print(f"跳过格式错误文件: {f}，错误：{e}")
            continue

        if latest_time is None or file_time > latest_time:
            latest_time = file_time
            latest_file = f

    if latest_file:
        full_path = os.path.join(output_folder, latest_file)
        print(f"找到最新文件: {full_path}")
        return full_path
    else:
        print("找不到有效文件")
        return None
def get_latest_displacement_result(output_folder='./output'):
    return get_latest_json_file(
        output_folder=output_folder,
        prefix='displacement_result_',
        timestamp_format='%Y-%m-%d_%H-%M-%S'
    )


##############################################################################

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
    
    #################################新增数据上传###################################
    # def handle_send_messages(self, json_path: str = None):
    #     """
    #     按要求的格式发送：
    #     UD{imei}MONIX+{
    #         "固定通道号": {
    #             "t": ...,
    #             "s": ...,
    #             "p": [...]
    #         }
    #     }
    #     """
    #     print("执行发送逻辑")
    #
    #     if json_path is None:
    #         print("没有提供 JSON 文件路径!")
    #         return
    #
    #     try:
    #         with open(json_path, 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #     except Exception as e:
    #         print(f"读取 JSON 文件失败: {e}")
    #         return
    #
    #     if not self.check_data_integrity(data):
    #         print("数据完整性检查失败，上传中止！")
    #         return
    #
    #     try:
    #         if not data:
    #             print("没有可发送的数据！")
    #             return
    #
    #         # 只发送最新一帧
    #         record = data[-1]
    #
    #         # 固定通道号：这里先写 1
    #         # 后面如果你们约定成别的，比如 101、5，就改这里
    #         channel_id = "1"
    #
    #         # 外层只保留一个固定通道号
    #         send_obj = {
    #             channel_id: {
    #                 "t": record.get("t"),
    #                 "s": record.get("s"),
    #                 "p": record.get("p", [])
    #             }
    #         }
    #
    #         json_str = json.dumps(send_obj, ensure_ascii=False, separators=(",", ":"))
    #         msg = f"UD{self.config.get('server', 'imei')}MONIX+{json_str}"
    #         res_bytes = msg.encode("utf-8")
    #
    #         self.client_socket.sendall(get_length_prefix_bytes(res_bytes))
    #
    #         print(f"已发送最新帧，原始t={record.get('t')}")
    #         print(f"发送内容: {msg}")
    #
    #     except Exception as e:
    #         print(f"发送最新帧失败: {e}")


    def handle_send_messages(self, json_path: str = None):
        """
        按要求的格式发送：
        UD{imei}MONIX+{
            "固定通道号": {
                "t": ...,
                "s": ...,
                "p": [...]
            }
        }

        调试模式下：
        - 每次发送自动更新 t
        - 基于最后一帧构造虚拟变化位移
        """
        print("执行发送逻辑")

        if json_path is None:
            print("没有提供 JSON 文件路径!")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"读取 JSON 文件失败: {e}")
            return

        if not self.check_data_integrity(data):
            print("数据完整性检查失败，上传中止！")
            return

        try:
            if not data:
                print("没有可发送的数据！")
                return

            # ========= 取最后一帧作为模板 =========
            base_record = data[-1]
            record = copy.deepcopy(base_record)

            # ========= 初始化发送计数器 =========
            if not hasattr(self, "_virtual_send_count"):
                self._virtual_send_count = 0
            self._virtual_send_count += 1

            k = self._virtual_send_count

            # ========= 更新时间戳（毫秒） =========
            current_t_ms = int(time.time() * 1000)
            record["t"] = current_t_ms

            # ========= 强制状态正常 =========
            record["s"] = 1

            # ========= 构造虚拟变化数据 =========
            # 这里用正弦变化，曲线更像真实监测数据
            # 你可以自己调幅值
            virtual_dx_mm = 5.0 * math.sin(0.2 * k)  # x方向 ±5 mm
            virtual_dy_mm = 3.0 * math.sin(0.2 * k + 1)  # y方向 ±3 mm
            virtual_dz_mm = 8.0 * math.sin(0.2 * k + 2)  # z方向 ±8 mm

            # 如果 p 不为空，就把每个点的位移改掉
            p_list = record.get("p", [])
            new_p_list = []

            for pt in p_list:
                if not isinstance(pt, list):
                    pt = list(pt)

                # 双目协议格式：
                # [左点ID, 右点ID, 通道, x像素移动, y像素移动, z像素移动,
                #  x位移, y位移, z位移, 匹配置信度, 跟踪置信度]
                if len(pt) >= 11:
                    new_pt = pt[:]

                    # 像素位移也给一点小变化，避免完全静止
                    new_pt[3] = round(0.5 * math.sin(0.2 * k), 4)
                    new_pt[4] = round(0.3 * math.sin(0.2 * k + 1), 4)
                    new_pt[5] = round(0.8 * math.sin(0.2 * k + 2), 4)

                    # mm位移改成虚拟变化值
                    new_pt[6] = round(virtual_dx_mm, 6)
                    new_pt[7] = round(virtual_dy_mm, 6)
                    new_pt[8] = round(virtual_dz_mm, 6)

                    # 置信度固定高一点，保证能被协议筛出来
                    new_pt[9] = 1.0
                    new_pt[10] = 1.0

                    new_p_list.append(new_pt)
                else:
                    new_p_list.append(pt)

            record["p"] = new_p_list

            # ========= 固定通道号 =========
            channel_id = "1"

            send_obj = {
                channel_id: {
                    "t": record.get("t"),
                    "s": record.get("s"),
                    "p": record.get("p", [])
                }
            }

            json_str = json.dumps(send_obj, ensure_ascii=False, separators=(",", ":"))
            msg = f"UD{self.config.get('server', 'imei')}MONIX+{json_str}"
            res_bytes = msg.encode("utf-8")

            self.client_socket.sendall(get_length_prefix_bytes(res_bytes))

            print(f"已发送虚拟变化帧，第{k}次，t={record.get('t')}")
            print(f"虚拟位移: x={virtual_dx_mm:.3f} mm, y={virtual_dy_mm:.3f} mm, z={virtual_dz_mm:.3f} mm")
            print(f"发送内容: {msg}")

        except Exception as e:
            print(f"发送虚拟变化帧失败: {e}")


    def upload_metadata(self, output_folder: str = './output') -> None:
        """
        上传 metadata 文件：上传 metadata_+时间.json 文件
        :param output_folder: 输出文件夹路径，默认 './output'
        """
        # 获取 metadata 文件路径
        metadata_file = get_latest_json_file(output_folder=output_folder)
        
        if metadata_file:
            print(f"正在上传 {metadata_file} ...")
            self.handle_send_messages(json_path=metadata_file)
        else:
            print("没有找到有效的 metadata JSON 文件！")



    def send_messages(self):
        """每隔 固定时间推送一次"""
        try:
            while True:
                '''if self.send_work_status:
                    ##########################主动推送消息逻辑start##################################           
                    # 先上传 metadata 文件
                    self.upload_metadata(output_folder='./output')'''

                # 上传 displacement_result 文件（使用最新文件）
                latest_displacement_file = get_latest_displacement_result(output_folder='./output')
                if latest_displacement_file:
                    self.handle_send_messages(json_path=latest_displacement_file)
                else:
                    print("没有找到有效的 displacement_result JSON 文件！")

                    ###########################主动推送消息逻辑end################################### 
                time.sleep(self.config.getint('time', 'send_time'))  # 上传间隔
        except Exception as e:
            print(f"主动推送消息失败: {e}")
        finally:
            print("推送线程终止!")
####################检测我的最后一个数据是否可行###################
    def check_data_integrity(self, data):
        """
        检查 displacement_result JSON 数据完整性。
        你当前双目项目中：
        每条 frame 格式为 {"t": ..., "s": ..., "p": [...]}
        每个点格式为：
        [左点ID, 右点ID, 通道, x像素移动, y像素移动, z像素移动,
         x位移, y位移, z位移, 匹配置信度, 跟踪置信度]
        """
        if not data:
            print("没有数据！")
            return False

        if not isinstance(data, list):
            print("displacement_result 顶层不是 list！")
            return False

        last_record = data[-1]

        if not isinstance(last_record, dict):
            print("最后一条记录不是 dict！")
            return False

        if not isinstance(last_record.get('t'), (float, int)):
            print("时间戳 t 格式不正确！")
            return False

        if last_record.get('s') not in [1, 0, -1]:
            print("状态 s 格式不正确！")
            return False

        p = last_record.get('p')
        if not isinstance(p, list):
            print("p 不是 list！")
            return False

        for point in p:
            if not isinstance(point, list):
                print(f"点数据不是 list: {point}")
                return False

            if len(point) != 11:
                print(f"点数据 {point} 长度不正确，应为 11 个元素！")
                return False

            # 前 3 个应为 int：左点ID、右点ID、通道
            if not isinstance(point[0], int):
                print(f"左点ID 格式不正确: {point[0]}")
                return False
            if not isinstance(point[1], int):
                print(f"右点ID 格式不正确: {point[1]}")
                return False
            if not isinstance(point[2], int):
                print(f"通道格式不正确: {point[2]}")
                return False

            # 后 8 个应为数值
            if not all(isinstance(x, (float, int)) for x in point[3:]):
                print(f"点数据数值格式不正确: {point}")
                return False

        print("数据完整性检查通过！")
        return True




    ###############################################################################

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
        


    def send_messages2(self):
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
            print(f"主动推送消息失败2: {e}")
        finally:
            print("推送线程终止!")
        
    def receive_messages(self):
        """接收并处理来自服务器的消息"""
        try:
            while True:
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        print("⚠️ 服务器断开连接")
                        break

                    ##########################处理回调消息逻辑start##################################
                    self.handle_recv_messages(data)
                    ###########################处理回调消息逻辑end################################### 

                except ConnectionResetError:
                    print("❌ 连接被服务器重置 (ConnectionResetError)")
                    break
                except ConnectionAbortedError:
                    print("❌ 连接被本地软件中止 (ConnectionAbortedError)")
                    break
                except OSError as e:
                    if e.winerror == 10053:
                        print("❌ WinError 10053: 本地软件中止了一个已建立的连接（可能是防火墙或程序问题）")
                    else:
                        print(f"❌ OSError: {e}")
                    break
                except Exception as e:
                    print(f"接收数据时遇到未知异常: {e}")
                    traceback.print_exc()
                    break  # 根据需求，也可以考虑继续而不是 break

        finally:
            print("🔚 接收线程终止！尝试关闭 socket")
            try:
                self.client_socket.close()
            except:
                pass

    def handle_recv_messages(self, data: bytes):
        """
        接收并处理来自服务器的消息的具体逻辑
       
        '''''''''''''''''
        '继承后需要被重写'
        '''''''''''''''''

        """
        print(f"接收到来自服务器的消息: {data}")
        print(f"接收到来自服务器的消息解码: {data.hex()} {data.hex()[0:2]}")

        res_bytes = ("UD" + self.config.get('server', 'imei')  + "MONIX" + str(data.hex()[0:2]) + "+" +str(random.random())).encode('utf-8')

        self.client_socket.sendall(get_length_prefix_bytes(res_bytes))


if __name__ == "__main__":
    tcp_client = TcpClient()
    tcp_client.run()