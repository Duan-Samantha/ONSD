from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utilis import *

img_path = ""

middle_density = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        if file.endswith(".png"):
            img = Image.open(os.path.join(root, file))
            img_arr = np.array(img)
            img_arr = img_arr[:,:,0]

        middle_density.append([file, np.sum(img_arr[:,int(img_arr.shape[1]/2)])])
middle_density.sort(key = lambda x:x[1])
best_ = middle_density[:5]

scores = 0
for frame in range(len(best_)):
    img = Image.open(f"/KCF-DSST-py/save_cut/{best_[frame][0]}")
    img_arr = np.array(img)
    # 481, 729, 435, 581
    img_arr = img_arr[:, :, 0]

    thres_img = np.where(img_arr < np.mean(img_arr), 255, 0)

    plot_list = []
    for i in range(img_arr.shape[1]):
        plot_list.append(np.sum(img_arr[:, i]))

    nor_plot_list = normalize(plot_list, 0, img_arr.shape[0])
    x = [i for i in range(0, img_arr.shape[1])]

    new_index = []
    r = 5
    for i in range(r, len(nor_plot_list) - r, 1):
        if np.max(nor_plot_list[i - r:i + 1]) == np.max(nor_plot_list[i:i + r]):
            new_index.append(i)

    plt.gray()
    plt.xlim(0, img_arr.shape[1])
    plt.ylim(0, img_arr.shape[0])
    plt.imshow(img_arr)
    plt.plot(nor_plot_list, color='r')

    # 限制选取最好的两条直线
    # sorted_index = []
    # for i in range(len(new_index)):
    #     sorted_index.append(nor_plot_list[new_index[i]])
    # two_index = []
    # two_index.append(new_index[np.argmax(sorted_index)])
    # sorted_index[np.argmax(sorted_index)] = 0
    # two_index.append(new_index[np.argmax(sorted_index)])

    max_index = [0, 0]
    max_l, max_r = 0, 0
    for i in new_index:
        if i < img_arr.shape[1] / 2:
            if nor_plot_list[i] > max_l:
                max_l = nor_plot_list[i]
                max_index[0] = i
        if i > img_arr.shape[1] / 2:
            if nor_plot_list[i] > max_r:
                max_r = nor_plot_list[i]
                max_index[1] = i
    if max_r == 0:
        max_index[1] = img_arr.shape[1] - 2

    plt.vlines(max_index[0], ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')
    plt.vlines(max_index[1], ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')

    # TODO 这里是灰度值占比
    # verified_score = []
    # for i in range(len(new_index) - 1):
    #     for j in range(i + 1, len(new_index), 1):
    #         a, b = new_index[i], new_index[j]
    #         if a < img_arr.shape[0] / 2 and b > img_arr.shape[0] / 2:
    #             print(a, b)
    #             verified_score.append([np.sum(thres_img[:, a:b] == 255) / (thres_img.shape[0] * (b - a)), (a, b)])
    #
    # verified_score.sort(key=lambda x: x[0])

    # for i in new_index:
    #     plt.vlines(i, ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')

    plt.legend(['density', 'D_L', 'D_R'], loc=1, bbox_to_anchor=(2, 1))
    plt.show()

    # plt.imshow(thres_img)
    # for i in range(len(new_index)):
    #     plt.vlines(new_index[i], ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')
    # plt.show()

    scores += (max_index[1] - max_index[0]) * 0.05
print(f"average distance is {scores / 5} mm")

# verified_score = []
# for i in range(len(new_index)-1):
#     for j in range(i+1, len(new_index), 1):
#         a, b = new_index[i], new_index[j]
#         if a < img_arr.shape[0]/2 and b > img_arr.shape[0]/2:
#             print(a, b)
#             verified_score.append([np.sum(thres_img[:, a:b] == 255) / (thres_img.shape[0] * (b - a)), (a, b)])
#
# verified_score.sort(key = lambda x:x[0])
# print(verified_score[-1][1])
#
# plt.imshow(thres_img)
# for i in verified_score[-1][1]:
#     plt.vlines(i, ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')
# plt.show()