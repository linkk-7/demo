from dao.base_dao import T, BaseDao, global_conn, global_cur
from utils.sqlite_utils import find_all
from datas.camera import Camera
from typing import List


class CameraDao(BaseDao[T]):
    def find_by_obj_param_id(self, obj_param_id: int) -> List[T]:
        """
        根据param_id查询监测对象
        """
        columns = [column[1] for column in find_all(self.cur, f"PRAGMA table_info({self.table_name})")]  # 列名列表
        datas = find_all(self.cur, f"select * from {self.table_name} where obj_param_id = {obj_param_id};")
        datas = [self.table_class(**{columns[i]: data[i] for i in range(len(data)) }) for data in datas]
        return datas
    
    def find_by_sensor_param_id(self, sensor_param_id: int) -> List[T]:
        """
        根据sensor_param_id查询监测对象
        """
        columns = [column[1] for column in find_all(self.cur, f"PRAGMA table_info({self.table_name})")]  # 列名列表
        datas = find_all(self.cur, f"select * from {self.table_name} where sensor_param_id = {sensor_param_id};")
        datas = [self.table_class(**{columns[i]: data[i] for i in range(len(data)) }) for data in datas]
        return datas

"""
dao工具类
"""
cameraDao = CameraDao[Camera]("camera", Camera, global_conn, global_cur)


