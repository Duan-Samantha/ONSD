import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

video = cv2.VideoCapture("D:\桌面\样本数据/2.mp4")

cot = 0
frames = 10
while True:
    ok, frame = video.read()
    if not ok:
        print("finished")
        video.release()
        # cv2.destroyAllWindows()
        break
    if cot % frames == 0:

        if cot == 0:
            bbox = cv2.selectROI('Select ROI', frame, False)
            print("start processing ...")
        cut_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]

        img = Image.fromarray(np.uint8(cut_frame))
        img.resize((512, 512))
        img.save(f"D:\桌面\样本数据/imgs/cut_frame_{cot}.png")
    cot += 1
