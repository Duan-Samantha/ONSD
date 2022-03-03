# KCF-DSST-py
- 使用方法，`python run.py`勾选ROI区域，然后松开鼠标按回车或者空格健即可
- 追踪区域会自动被保存在当前目录下的`save_cut`下，请运行`python run.py`前先在当前目录`KCF-DSST-py/`下新建`save_cut`文件夹以保存追踪结果切片。
    - 默认是每10帧保存一次结果
    - 可在`run.py`->`if cot % 10 == 0:` 处修改
    - 读取视频地址修改：`run.py`-> `video = cv2.VideoCapture(...)`处修改
