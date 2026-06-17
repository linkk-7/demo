from typing import List, Tuple
from roifile import ROI_OPTIONS, ROI_TYPE, ImagejRoi

from read_roi import read_roi_file


def generate_roi_file(file_path: str, feature_points: List[Tuple[float, float]]) -> None:
    roi = ImagejRoi.frompoints(feature_points)
    roi.roitype = ROI_TYPE.POINT
    roi.options |= ROI_OPTIONS.SHOW_LABELS
    roi.tofile(file_path)

def get_points_from_roi(flie_path: str, x_size: float = 1, y_size: float = 1) -> List[Tuple[float, float]]:
    roi_data = read_roi_file(flie_path)
    points = []
    for key, item in roi_data.items():
        if item['type'] == 'point':
            for i in range(item['n']):
                points.append((item['x'][i]/x_size, item['y'][i]/y_size))
    return points


    