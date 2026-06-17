"""
时间相关工具函数
"""
import datetime
import pytz
import time


def get_current_time_str() -> str:
    """
    获取当前北京时间字符串，格式为%Y-%m-%dT%H:%M:%S
    """
    now = datetime.datetime.now()
    # 获取东八区的时区信息
    timezone = pytz.timezone('Asia/Shanghai')
    # 将当前时间转换为东八区时间
    now_east_8 = now.astimezone(timezone)
    return now_east_8.strftime('%Y-%m-%dT%H:%M:%S')

def get_current_timestamp(use_second: bool = False) -> int:
    """
    获取秒级或者毫秒级时间戳
    """
    if use_second:
        return int(time.time())
    else:
        return int(time.time() * 1000)
    
def timestamp_to_str(time_stamp: int, use_second: bool = False) -> str:
    # 将毫秒时间戳转换为秒时间戳
    if not use_second:
        timestamp_sec = time_stamp / 1000.0
    else:
        timestamp_sec = time_stamp

    # 创建东八区时区对象（UTC+8）
    tz_utc_8 = datetime.timezone(datetime.timedelta(hours=8))

    # 使用支持时区的对象转换为东八区时间
    dt = datetime.datetime.fromtimestamp(timestamp_sec, tz=tz_utc_8)

    # 格式化为 'yyyyMMddThhmmssZ' 格式
    # 如果你想保留 Z，表示 UTC 的话可以加其他标记,比如" +08:00"
    time_str = dt.strftime('%Y%m%dT%H%M%SZ')    
    return time_str
