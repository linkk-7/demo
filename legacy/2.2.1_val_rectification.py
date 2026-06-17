import cv2
import numpy as np
import os

def draw_epipolar_lines(left_img_path, right_img_path, save_path, line_interval=50):
    # 读取左右图像
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    if img_left is None or img_right is None:
        raise ValueError("无法读取左右图像，请检查路径是否正确")

    # 如果尺寸不同，先统一高度
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    if h1 != h2:
        target_h = min(h1, h2)
        img_left = cv2.resize(img_left, (int(w1 * target_h / h1), target_h))
        img_right = cv2.resize(img_right, (int(w2 * target_h / h2), target_h))

    # 横向拼接
    combined = np.hstack((img_left, img_right))

    h, w = combined.shape[:2]

    # 画水平极线
    for y in range(0, h, line_interval):
        cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

    # 保存
    cv2.imwrite(save_path, combined)

    # 显示
    cv2.namedWindow("Epipolar Check", cv2.WINDOW_NORMAL)
    cv2.imshow("Epipolar Check", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"已保存结果到: {save_path}")


if __name__ == "__main__":
    left_img_path = r"new_data5\cab3\left_rec\2026032217433429070.jpg"
    right_img_path = r"new_data5\cab3\right_rec\2026032217433433570.jpg"
    save_path = r"new_data5\cab3\epipolar_check.jpg"

    draw_epipolar_lines(left_img_path, right_img_path, save_path, line_interval=50)