import numpy as np
from numpy import abs, sqrt, arctan2, arctan, pi, real
from numpy.fft import fft2, ifft2
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(im):
    b = np.array([[2, 4,  5,  2,  2],
               [4, 9,  12, 9,  4],
               [5, 12, 15, 12, 5],
               [4, 9,  12, 9,  4],
               [2, 4,  5,  4,  2]]) / 156
    kernel = np.zeros(im.shape)
    kernel[:b.shape[0], :b.shape[1]] = b

    fim = fft2(im)
    fkernel = fft2(kernel)
    fil_im = ifft2(fim * fkernel)

    return abs(fil_im).astype(int)

def gradient(im):
    # Sobel operator
    op1 = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]])
    op2 = np.array([[-1, -2, -1],
                 [ 0,  0,  0],
                 [ 1,  2,  1]])
    kernel1 = np.zeros(im.shape)
    kernel1[:op1.shape[0], :op1.shape[1]] = op1
    kernel1 = fft2(kernel1)

    kernel2 = np.zeros(im.shape)
    kernel2[:op2.shape[0], :op2.shape[1]] = op2
    kernel2 = fft2(kernel2)

    fim = fft2(im)

    Gx = real(ifft2(kernel1 * fim)).astype(float)
    Gy = real(ifft2(kernel2 * fim)).astype(float)

    G = sqrt(Gx**2 + Gy**2)
    Theta = arctan2(Gy, Gx) * 180 / pi
    return G, Theta

def maximum(det, phase):
  gmax = np.zeros(det.shape)
  for i in range(gmax.shape[0]):
    for j in range(gmax.shape[1]):
      if phase[i][j] < 0:
        phase[i][j] += 360

      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
        # 0 degrees
        if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
          if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
            gmax[i][j] = det[i][j]
        # 45 degrees
        if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
          if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
            gmax[i][j] = det[i][j]
        # 90 degrees
        if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
          if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
            gmax[i][j] = det[i][j]
        # 135 degrees
        if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
          if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
            gmax[i][j] = det[i][j]
  return gmax

def thresholding(im):
    thres  = np.zeros(im.shape)
    strong = 1.0
    weak   = 0.5
    mmax = np.max(im)
    lo, hi = 0.1 * mmax, 0.8 * mmax
    strongs = []
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            px = im[i][j]
            if px >= hi:
                thres[i][j] = strong
                strongs.append((i, j))
            elif px >= lo:
                thres[i][j] = weak
    return thres, strongs

class tracking:
    def __init__(self, tr):
        self.im = tr[0]
        strongs = tr[1]

        self.vis = np.zeros(self.im.shape, bool)
        self.dx = [1, 0, -1,  0, -1, -1, 1,  1]
        self.dy = [0, 1,  0, -1,  1, -1, 1, -1]
        for s in strongs:
            if not self.vis[s]:
                self.dfs(s)
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                self.im[i, j] = 1.0 if self.vis[i, j] else 0.0

    def dfs(self, origin):
        q = [origin]
        while len(q) > 0:
            s = q.pop()
            self.vis[s] = True
            self.im[s] = 1
            for k in range(len(self.dx)):
                for c in range(1, 16):
                    nx, ny = s[0] + c * self.dx[k], s[1] + c * self.dy[k]
                    if self.exists(nx, ny) and (self.im[nx, ny] >= 0.5) and (not self.vis[nx, ny]):
                        q.append((nx, ny))
        pass

    def exists(self, x, y):
        return x >= 0 and x < self.im.shape[0] and y >= 0 and y < self.im.shape[1]


if __name__ == '__main__':
    # preprocessing
    im = np.array(Image.open("/Desktop/视神经鞘/save_cut/1_cut.png"))
    im = im[:, :, 0]

    # canny algorithm
    gim = gaussian(im)
    grim, gphase = gradient(gim)
    gmax = maximum(grim, gphase)
    thres = thresholding(gmax)
    edge = tracking(thres)

    # plot resule
    plt.gray()

    fig = plt.figure()
    fig.add_subplot(121)
    plt.title("Original image")
    plt.imshow(im)
    fig.add_subplot(122)
    plt.title("canny edge output image")
    plt.imshow(edge.im)

    plt.show()