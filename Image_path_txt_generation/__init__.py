#将所有图片的路径写入一个txt文件中，一张图片一行
import os
def imgs_name_txt(num_img,filepath,mode = 'left'):
    if mode=='left':
        txt_name = 'left_image_paths.txt'
        images_file = os.path.join(filepath, txt_name)
        with open(images_file, 'w') as file:
            for i in range(num_img):
                file.write(filepath + '\\' + 'left_rec\\' + str(i + 1) + 'l.jpg' + '\n')
    else:
        txt_name = 'right_image_paths.txt'
        images_file = os.path.join(filepath, txt_name)
        with open(images_file, 'w') as file:
            for i in range(num_img):
                file.write(filepath + '\\' + 'right_rec\\' + str(i + 1) + 'r.jpg' + '\n')

