import numpy as np
import scipy.ndimage as ndimage
import pandas as pd
from cv2 import cv2

def gaussianfilter(shape=(5,5), sigma = 1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma**2))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def excelRead(path):
    data = pd.read_excel(path)
    return pd.DataFrame.to_numpy(data)

def ConvGauss(L: np.ndarray, filter: np.ndarray, k):
    L2 = L.copy()
    for i in range(0,k):
        L2 = ndimage.correlate(L2, filter, mode='nearest')
    return L2

def getDiffGauss(I: np.ndarray, filter: np.ndarray):
    nr, nc = I.shape
    stack = np.full((4, nr, nc), 100)
    stack[0] = I
    diff = np.zeros(shape=(3, nr, nc))
    for i in range(1, 4):
        stack[i] = ConvGauss(I, filter, i)
        diff[i-1] = stack[i] - stack[i-1]
    return diff

def getExtrema(diff: np.ndarray):
    points = []
    patch = np.zeros(shape=(27))
    l,nr,nc = diff.shape
    for r in range(0, nr):
        for c in range(0, nc):
            for s in range(0,3):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if r+i < 0 or c+j < 0:
                            patch[3*s+(i+1)+(j+1)] = 0
                        else:
                            patch[3*s+(i+1)+(j+1)] = diff[s, r+i-1, c+j-1]
            #print(patch)
            if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                points.append((r, c))
    return points

I = excelRead("image.xlsx")
g1=gaussianfilter()
diff = getDiffGauss(I, g1)
points = getExtrema(diff)
print(points)