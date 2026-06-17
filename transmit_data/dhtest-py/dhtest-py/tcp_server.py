import socket
import struct
import threading
from datetime import datetime, timedelta

client = None
running = True
serverSocket = None  # 保持这个

def start_tcp_server(host, port,pointMap):
    try:
        print("start")
        global serverSocket
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.bind((host, port))
        serverSocket.listen(5)

        while running:
            try:
                serverSocket.settimeout(1)  # 每1秒检查一次是否应退出
                clientSocket, clientAddress = serverSocket.accept()
                # print("host",host,"port",port)
                threading.Thread(target=handle_client, args=(clientSocket,pointMap), daemon=True).start()
            except socket.timeout:
                # print("TCP服务器超时",host,port)
                continue
            except Exception as e:
                print("TCP服务器异常：", e)
            
    except Exception as e:
        print("TCP server start error:", e)
        return
    
def handle_client(clientSocket,pointMap):
    with clientSocket:
        try:
            while running:
                #print("runing")
                rev_tcp(clientSocket,pointMap)
        except Exception as e:
            print("客户端处理错误：", e)

def recv_n_bytes(conn, n):
    """确保从 socket 中读取 n 字节"""
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise Exception("连接断开")
        data += chunk
    return data

def rev_tcp(conn,pointMap):
    header = recv_n_bytes(conn, 16)
    # print("header",header)
    # if len(header) < 16:
        # return
    # 包头标识 2字节
    packetSign = header[:2].hex()

    # 包体长度 4字节
    packetLengthHex = header[2:6].hex()
    packetByte = bytes.fromhex(packetLengthHex)
    packetLength = int.from_bytes(packetByte, byteorder="little")

    # 版本号 1字节
    versionHex = header[6:7].hex()

    # 命令码 1字节
    cmdHex = header[7:8].hex()
    cmdByte = bytes.fromhex(cmdHex)
    cmd = int.from_bytes(cmdByte, byteorder="little")

    # 预留字段 8 字节
    remainHex = header[8:16].hex()

    body = recv_n_bytes(conn, packetLength)
    if len(body) < packetLength:
        raise Exception("接收包体失败")

    # 4. 判断是否处理
    if cmd != 2:
        # print(f"跳过命令码 {cmd},{packetLength} 的报文")
        return
    
    # print("body",body)

    resolve_tcp(body,pointMap)
    #end_time = datetime.utcnow()
    #print("一次波形数据",end_time)


def resolve_tcp(body,pointMap):
    # 包体的前4个字节是测点数量
    pointsNumHex = body[:4].hex()
    pointsNumByte = bytes.fromhex(pointsNumHex)
    pointsNum = int.from_bytes(pointsNumByte, byteorder="little")

    # 这里包含m个波形数据
    waveBody = body[4:]

    # 上一次的结束位置
    lastEnd = 0
    influx_data = []
    # 从0-m
    for i in range(pointsNum):
        # 前4个字节是测点id
        pointIdHex = waveBody[lastEnd:lastEnd + 4].hex()
        lastEnd += 4
        pointIdByte = bytes.fromhex(pointIdHex)
        pointId = int.from_bytes(pointIdByte, byteorder="little")
        # 测点外部编码 16字节
        pointOutCodeHex = waveBody[lastEnd:lastEnd + 16].hex()
        lastEnd += 16
        # 采样频率 4字节
        frequencyByte = waveBody[lastEnd:lastEnd + 4]
        lastEnd += 4
        frequency = struct.unpack('<f', frequencyByte)[0]  # 小端 float
        # 采样时间 8字节
        timeBytes = waveBody[lastEnd:lastEnd + 8]
        lastEnd += 8
        timeInNs = struct.unpack('<q', timeBytes)[0]  # 小端 int64（带符号）
        time = timeInNs // 10000
        # 波形数据个数n 4字节
        waveNumHex = waveBody[lastEnd:lastEnd + 4].hex()
        lastEnd += 4
        waveNumByte = bytes.fromhex(waveNumHex)
        waveNum = int.from_bytes(waveNumByte, byteorder="little")
        # 波形类型 2字节
        waveTypeHex = waveBody[lastEnd:lastEnd + 2].hex()
        lastEnd += 2
        waveTypeByte = bytes.fromhex(waveTypeHex)
        waveType = int.from_bytes(waveTypeByte, byteorder="little")

        channel = pointMap[pointId]
        # print("channel",channel)

        for j in range(waveNum):
            # YF 1维float
            calcRes = {}
            if waveType == 0:
                calcRes = calc_wave_data_yf(waveBody, lastEnd,j,time,frequency)
                lastEnd += 4
            elif waveType == 1:
                # YD 1维double
                calcRes = calc_wave_data_yd(waveBody, lastEnd,j,time,frequency)
                lastEnd += 8
            elif waveType == 2:
                # XYF 二维 时间/频率
                calcRes = calc_wave_data_xyf(waveBody, lastEnd)
                lastEnd += 8
            elif waveType == 3:
                # XYD 二维 时间/频率
                calcRes = calc_wave_data_xyd(waveBody, lastEnd)
                lastEnd += 8
            else:
                # complex 复数
                calcRes = calc_wave_data_complex(waveBody, lastEnd,j,time,frequency)
                lastEnd += 8
            # print("存储消息",calcRes,waveType)
            if(channel == None or channel == ""):
                # print("channel")
                continue
            #print("data",{"channel": channel,"time": calcRes["time"],"value": calcRes["value"]})
            influx_data.append({
                "channel": channel,
                "time": calcRes["time"],
                "value": calcRes["value"]
            })
        store_data_to_influx({"channel": channel,"time": calcRes["time"],"value": calcRes["value"]})


# 处理YF 数据
def calc_wave_data_yf(waveBody, lastEnd,index,time,frequency):
    # 小端 float 解码
    waveDataFloat = struct.unpack('<f', waveBody[lastEnd:lastEnd + 4])[0]
    lastEnd += 4
    # 计算采样时间
    sampTime = round(time + index * (1 / frequency))
    return {
        "time": sampTime,
        "value": waveDataFloat
    }

# 处理YD
def calc_wave_data_yd(waveBody, lastEnd,index,time,frequency):
    # 小端 double
    waveDataDouble = struct.unpack('<d', waveBody[lastEnd:lastEnd + 8])[0]
    lastEnd += 8
    # 计算采样时间
    sampTime = round(time + index * (1 / frequency))
    return {
        "time": sampTime,
        "value": waveDataDouble
    }

def calc_wave_data_xyf(waveBody, lastEnd):
    # 二维x 小端float 时间， 小端float 数值
    x = struct.unpack('<f', waveBody[lastEnd:lastEnd + 4])[0]
    lastEnd += 4
    y = struct.unpack('<f', waveBody[lastEnd:lastEnd + 4])[0]
    lastEnd += 4
    return {
        "time": x,
        "value": y
    }
    
def calc_wave_data_xyd(waveBody, lastEnd):
    # 二维x 小端double 时间， 小端double 数值
    x = struct.unpack('<d', waveBody[lastEnd:lastEnd + 8])[0]
    lastEnd += 4
    y = struct.unpack('<d', waveBody[lastEnd:lastEnd + 8])[0]
    lastEnd += 4
    return {
        "time": x,
        "value": y
    }

def calc_wave_data_complex(waveBody, lastEnd,index,time,frequency):
    # 小端 double
    realPart = struct.unpack('<f', waveBody[lastEnd:lastEnd + 4])[0]
    lastEnd += 4
    imagPart = struct.unpack('<f', waveBody[lastEnd:lastEnd + 4])[0]
    lastEnd += 4
    # 计算采样时间
    sampTime = round(time + index * (1 / frequency))
    return {
        "time": sampTime,
        "value": complex(realPart, imagPart)
    }

def store_data_to_influx(data):
    # 如果是单条字典数据，自动转换为列表
    if isinstance(data, dict):
        data = [data]

    try:
        json_body = []
        for point in data:
            # 将毫秒时间戳转换为 ISO 时间
            timestamp = datetime.utcfromtimestamp(point["time"] / 1000).isoformat() + "Z"
            # print(timestamp)
            json_body.append({
                "measurement": "wave_data",
                "time": timestamp,
                "tags": {
                    "channel": str(point["channel"])
                },
                "fields": {
                    "value": float(point["value"])
                }
            })

        # 写入数据
        client.write_points(json_body)
    except Exception as e:
        print(f"Failed to write data to InfluxDB: {e}")