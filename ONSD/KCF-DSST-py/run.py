import cv2
from tracker import KCFTracker

def tracker(cam, frame, bbox):
    tracker = KCFTracker(True, True, True) # (hog, fixed_Window, multi_scale)
    tracker.init(bbox, frame)
    cot = 0
    while True:
        ok, frame = cam.read()

        timer = cv2.getTickCount()
        bbox = tracker.update(frame)
        if bbox == False:
            break
        bbox = list(map(int, bbox))
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        if cot % 10 == 0:
            # 保存追踪图片
            save_cut = frame[p1[1]:p2[1], p1[0]:p2[0], :]
            cv2.imwrite(f"D:\桌面\样本数据/imgs/{cot}_cut.png", save_cut)
        cot += 1

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Put FPS
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)
        # index = tracker.roi_total[cot]
        # index = [int(i) for i in index]
        # save_cut = frame[index[1]:index[1]+index[3], index[0]:index[0]+index[2], :]
        # cv2.imwrite("cut.png", save_cut)
        # cot += 1
        # # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    return tracker.roi_total


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    video = cv2.VideoCapture("D:\桌面\样本数据/2.mp4")
    ok, frame = video.read()
    bbox = cv2.selectROI('Select ROI', frame, False)

    if min(bbox) == 0: exit(0)
    roi_total = tracker(video, frame, bbox)
    save_cut_index = frame[index[1]:index[1]+index[3], index[0]:index[0]+index[2], :]
