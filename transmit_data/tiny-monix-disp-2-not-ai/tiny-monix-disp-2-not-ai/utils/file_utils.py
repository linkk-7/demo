"""
文件操作工具类
"""
import os
import shutil
import glob
from typing import List

def make_new_dir(path: str, clear: bool = False) -> None:
    """
    创建一个新的文件目录
    path: str 文件路径
    clear: bool 当clear为True时，如果该文件目录本来已经存在，则删掉它
    """
    if clear:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)

def delete_file(output_file: str) -> None:
    """
    如果存在文件，则删除文件
    """
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"{output_file} 已删除。")


def get_file_names_in_folder(path: str, complete: bool = False):
    # 获取文件夹下的所有文件的名称
    file_names = glob.glob(f"{path}/*")

    if complete:
        return file_names

    # 如果只想要文件名而不是完整路径
    file_names = [f.split("/")[-1] for f in file_names]
    return file_names

def copy_and_rename_file(src_file: str, dest_folder: str, new_name: str) -> None:
    """
    将文件 src_file 复制到 dest_folder下并改名为new_name， 如果new_name已经存在，则替换该文件
    """
    # 确保目标文件夹存在
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 构建目标文件路径
    dest_file = os.path.join(dest_folder, new_name)
    
    # 复制并重命名文件，存在则替换
    shutil.copy2(src_file, dest_file)
    print(f"文件已复制到: {dest_file}")


def get_subfolders(path: str)-> List[str]:
    """获取某个文件路径下的所有子文件夹（不递归）

    Args:
        path (str): 文件路径

    Returns:
        List[str]: 文件夹名称数组
    """
    return [f.name for f in os.scandir(path) if f.is_dir()]

def delete_files_by_prefix(folder_path: str, prefix_value: str):
    """
    删除文件夹中所有文件名以指定 prefix_value 值开头的文件。

    :param folder_path: 文件夹路径
    :param prefix_value: prefix 的值（数字）
    """
    # 构造匹配模式，例如 "34_*.jpg" 或 "34_*.png"
    pattern = os.path.join(folder_path, f"{prefix_value}_*.jpg")  # 匹配 .jpg 文件
    pattern_png = os.path.join(folder_path, f"{prefix_value}_*.png")  # 匹配 .png 文件

    # 查找所有匹配的文件
    matching_files = glob.glob(pattern) + glob.glob(pattern_png)

    print("匹配到的文件:", matching_files)
    # 删除匹配的文件
    for file_path in matching_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件失败: {file_path}, 错误: {e}")


if __name__ == "__main__":
    a = get_file_names_in_folder("/opt/tiny-monix/tiny-monix-displacement/data/4_7/left")
    print(a)