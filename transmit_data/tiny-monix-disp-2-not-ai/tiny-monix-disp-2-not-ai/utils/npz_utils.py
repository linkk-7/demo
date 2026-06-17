"""npz操作工具
"""
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray

def get_points_from_npz(filr_path: str) -> List[Tuple[float, float]]:
    """从npz文件中获取点数组
    Args:
        filr_path (str): 文件路径

    Returns:
        List[Tuple[float, float]]: 二维点集
    """
    npz_file = np.load(filr_path)
    keypoints = npz_file['pts'] # Assuming 'pts' contains the keypoints
    xy_points = [(point[0], point[1]) for point in keypoints]
    return xy_points

def generate_npz(filr_path: str, keys: List[str], arrays: List[NDArray]) -> None:
    """生成npz文件

    Args:
        filr_path (str): 文件路径
        keys (List[str]): npz字典的key值列表
        arrays (List[NDArray]): npz字典的value值列表
    """
    if len(keys) != len(arrays):
        print("key与value的数量不对等")
        return
    # 使用字典解包方式将 keys 和 arrays 组合成一个字典
    data_dict = {key: array for key, array in zip(keys, arrays)}
    np.savez(filr_path, **data_dict)