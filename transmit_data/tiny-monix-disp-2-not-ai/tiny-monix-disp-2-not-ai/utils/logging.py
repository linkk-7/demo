"""
自定义日志类
"""
import os
import re
class MyLog():
    need_output: bool  #是否需要输出
    def __init__(self, path: str, need_output: bool = True, need_clear_file: bool = False) -> None:
        """
        path: 输出日志的文件路径
        """
        self.__path = path
        self.need_output = need_output
        # 使用正则表达式提取目录部分
        dir_path = re.match(r'(.*/)', path)
        if dir_path:
            dir_path = dir_path.group(1)
            print("dir_path", dir_path)
            # 确保目录存在
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as _:
                pass  # 创建空文件
        if need_clear_file:
            with open(path, 'w', encoding='utf-8') as _:
                pass  # 创建空文件

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"MyLog(path={self.__path}, need_output={self.need_output})"



    def print_log(self, *args, **kwargs):
        if self.need_output:
            # 将输入的内容转换为字符串
            log_message = ' '.join(map(str, args))
            
            # 打开文件以追加的方式写入内容
            with open(self.__path, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
            
            # 调用内置的 print 函数打印到控制台
            print(*args, **kwargs)

if __name__ == "__main__":
    a = MyLog("test.log")