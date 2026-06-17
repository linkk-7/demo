import requests

# 发送登录请求
def send_login_req():
    url = "http://192.168.0.195:18001/webapi/api/Login/Login"
    reqData = {
        "UserName": "admin",
        "Password": "phmpassword",
        "UseToken": False,
        "LoginMethold": 0
    }
    try:
        response = requests.post(url, json=reqData)
        response.raise_for_status()
        data = response.json()
        # print(data)

        accessToken = data["Result"]["AccessToken"]
        refreshToken = data["Result"]["RefreshToken"]
        token = data["Result"]["Token"]
        return accessToken, refreshToken, token
    except requests.exceptions.RequestException as e:
        print(e)


# 获取设备树
def send_get_device_tree(accessToken, token):
    url = "http://192.168.0.195:18001/webapi/api/Tree/tree"
    reqHeaders = {
        "Authorization": "Bearer " + accessToken,
        "Content-Type": "application/json",
        "UserToken": token
    }
    try:
        response = requests.get(url, headers=reqHeaders)
        response.raise_for_status()
        data = response.json()
        # print("device", data)
        return data
    except requests.exceptions.RequestException as e:
        print(e)

# 获取设备下所有测点
def send_get_all_points(accessToken, token, deviceId):
    url = "http://192.168.0.195:18001/webapi/api/Point/simple-points?DeviceId=" + deviceId
    reqHeaders = {
        "Authorization": "Bearer " + accessToken,
        "Content-Type": "application/json",
        "UserToken": token
    }
    try:
        response = requests.get(url, headers=reqHeaders)
        response.raise_for_status()
        data = response.json()
        # print("device", data)
        return data
    except requests.exceptions.RequestException as e:
        print(e)




# 获取所有连接
def send_get_AllConnections(accessToken, token):
    url = "http://192.168.0.195:18001/webapi/api/Tcp/GetAllConnections"
    reqHeaders = {
        "Authorization": "Bearer " + accessToken,
        "Content-Type": "application/json",
        "UserToken": token
    }
    try:
        response = requests.get(url, headers=reqHeaders)
        response.raise_for_status()
        data = response.json()
        result = data["Result"]

        print("result", result)

        resIps = []
        resPorts = []
        for ipRes in result:
            ip = ipRes["Ip"]
            Port = ipRes["Port"]

            resIps.append(ip)
            resPorts.append(Port)

        return resIps, resPorts
    except requests.exceptions.RequestException as e:
        print(e)