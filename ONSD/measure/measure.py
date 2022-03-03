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

nor_new = normalize(img_arr[50,:], 0, img_arr.shape[0])
d = np.abs(np.where(nor_plot_list == 0)[0][0] - int(img_arr.shape[1]/2))

modified_list = []
for i in range(d,len(nor_plot_list),1):
    modified_list.append(nor_plot_list[i])

index = find_roots(nor_new, np.mean(nor_new), int(img_arr.shape[1]/2))

plt.gray()
plt.xlim(0, img_arr.shape[1])
plt.ylim(0, img_arr.shape[0])
plt.imshow(img_arr)
plt.plot(modified_list, color='r')
plt.vlines(index[0], ymin=0, ymax=img_arr.shape[0], colors='blue', linestyles='dashed')
plt.vlines(index[1], ymin=0, ymax=img_arr.shape[0], color='blue', linestyles='dashed')
# plt.vlines(x=np.where(nor_plot_list == 0), ymin=0, ymax=img_arr.shape[0], linestyles='dashed', colors='gold')
# plt.vlines(x=img_arr.shape[1]/2, ymin=0, ymax=img_arr.shape[0], linestyles='dashed')
# plt.plot(img_arr[int(img_arr.shape[0]/2),:])
plt.plot(nor_new, color='green')
plt.hlines(np.mean(nor_new), xmin=0, xmax=len(img_arr[50,:]))
plt.legend(['density', 'nor-zero-density', 'center'], loc=1, bbox_to_anchor=(2, 1))
plt.show()

print((index[1]-index[0])*0.05)