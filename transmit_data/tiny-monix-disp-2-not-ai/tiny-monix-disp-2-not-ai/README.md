# tiny-monix-displacement

小监测系统的视频位移识别

# 1.环境安装

```bash
conda create -n py310-sp python=3.10
source activate py310-sp
sudo apt update
sudo apt install build-essential
sudo apt install libgl1-mesa-glx
sudo apt install ffmpeg
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytz
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple configobj
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple kornia
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imutils
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple paramiko
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scp
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple psutil
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pynvml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pydantic
```


# 2.项目初始化
conda activate py310-sp
python generate_from_template.py
python run_sql.py