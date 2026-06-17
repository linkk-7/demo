
import os


#将指定文件夹中的图片根据其原名中时间先后命名为123顺序
def rename_images(folder_path,model):
    # 获取文件夹中的所有图片
    images = [f for f in os.listdir(folder_path) if f.endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    
    # 按时间排序    
    images.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    if model == "left":
        # 重命名图片
        for i, img in enumerate(images):
            new_name = f"{i+1}l.bmp"
            os.rename(os.path.join(folder_path, img), os.path.join(folder_path, new_name))
    elif model == "right":
        for i, img in enumerate(images):
            new_name = f"{i+1}r.bmp"
            os.rename(os.path.join(folder_path, img), os.path.join(folder_path, new_name))



rename_images(r"new_data5/cab/Camera_1", "left")
rename_images(r"new_data5/cab/Camera_0", "right")