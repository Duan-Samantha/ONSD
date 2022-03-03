# 数据预处理与标记
`preprocessing.py`：该程序将视频数据保存为图片用于后续标注
- usage
  - 在`preprocessing/`下新建`imgs`文件夹用于保存图片
  - 运行`python preprocessing.py`
  - 选取RIO区域如下图所示，尽量不要包含黑色区域(都点卡)
  - 回车，终端显示`start processing`即表示正在处理