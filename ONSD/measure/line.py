from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utilis import *




img_path = ""


img = Image.open("")
img_arr = np.array(img)
img_arr = img_arr[:,:,0]

plot_list = []
for i in range(img_arr.shape[1]):
    plot_list.append(np.sum(img_arr[:,i]))

nor_plot_list = normalize(plot_list, 0, img_arr.shape[0])
x = [i for i in range(0, img_arr.shape[1])]

plt.gray()
plt.xlim(0, img_arr.shape[1])
plt.ylim(0, img_arr.shape[0])
plt.imshow(img_arr)
plt.plot(x, nor_plot_list, color='r')
plt.vlines(x=np.where(nor_plot_list == 0), ymin=0, ymax=img_arr.shape[0], linestyles='dashed', colors='gold')
plt.vlines(x=img_arr.shape[1]/2, ymin=0, ymax=img_arr.shape[0], linestyles='dashed')
# plt.plot(img_arr[100,:])
plt.legend(['density', 'nor-zero-density', 'center'], loc=1, bbox_to_anchor=(2, 1))
plt.show()

# plt.subplot(3,1,1)
# plt.plot(img_arr[20,:])
# plt.title("start")
# plt.subplot(3,1,2)
# plt.plot(img_arr[200,:])
# plt.title("medium")
# plt.subplot(3,1,3)
# plt.plot(img_arr[300,:])
# plt.title("end")

plt.show()
