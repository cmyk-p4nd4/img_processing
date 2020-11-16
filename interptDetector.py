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

def getExtremaSingle(diff: np.ndarray, layer: int):
    points = []
    nr, nc = diff[layer].shape
    patch = np.zeros(shape=(3,3))
    for r in range(nr):
        for c in range(nc):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if r+i < 0 or c+j < 0 or r+i >= nr or c+j >= nc:
                        patch[i+1, j+1] = 0
                    else:
                        patch[i+1, j+1] = diff[layer, r+i, c+j]
            if (np.argmax(patch) == 4 or np.argmin(patch) == 4):
                points.append((r, c))
    return points
    
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

def saveKeypoints(I:np.ndarray, points: list, diff: np.ndarray):
    import xlwt
    from xlwt import Workbook

    style1 = xlwt.easyxf('pattern: pattern solid, fore_colour yellow;''align: vert centre, horiz center')
    style2 = xlwt.easyxf('pattern: pattern solid, fore_colour aqua;''align: vert centre, horiz center')
    reset = xlwt.easyxf('align: vert centre, horiz center')

    book = Workbook()
    sheet1 = book.add_sheet("highlight", cell_overwrite_ok=True)
    sheet2 = book.add_sheet("weak", cell_overwrite_ok=True)
    sheet3 = book.add_sheet("combine", cell_overwrite_ok=True)

    nr, nc = I.shape
    wPoints = []
    for i in range(nr):
        for j in range(nc):
            sheet1.write(i, j, int(I[i, j]), reset)
            sheet2.write(i, j, int(I[i, j]), reset)
            sheet3.write(i, j, int(I[i, j]), reset)
            if (i,j) in points:
                sheet1.write(i, j, int(I[i, j]), style1)
                sheet3.write(i, j, int(I[i, j]), style1)
                if diff[0, i, j] < 1.0:
                    sheet2.write(i, j, int(I[i, j]), style2)
                    sheet3.write(i, j, int(I[i, j]), style2)
                    wPoints.append((i, j))
    book.save("highlight.xls")
    return [x for x in points if x not in wPoints]

def getPointR(I: np.ndarray, points: list, diff:np.ndarray):
    import copy
    sPoints = copy.deepcopy(points)
    Sx = np.array([-1, 0, 1])
    Sy = Sx.reshape(3,1)
    Dx = ndimage.correlate1d(diff, Sx, mode='constant')
    Dxx = ndimage.correlate1d(Dx, Sx, mode= 'constant')
    Dy = ndimage.correlate(diff, Sy, mode='constant')
    Dyy = ndimage.correlate(Dy, Sy, mode='constant')
    Dxy = ndimage.correlate(Dx, Sy, mode='constant')
    R = np.zeros(shape=diff.shape)
    nr, nc = R.shape
    for i in range(nr):
        for j in range(nc):
            if (i, j) in points:
                tr = Dxx[i, j] + Dyy[i, j]
                det = Dxx[i, j] * Dyy[i, j] - Dxy[i, j] **2
                R[i, j] = tr**2/det
    return np.around(R, 2)

def orientationAssg(I: np.ndarray, g1: np.ndarray, keypts: list):
    Sx = np.array([-1, 0, 1])
    Sy = Sx.reshape(3,1)
    Dx = ndimage.correlate1d(I, Sx, mode='nearest')
    Dy = ndimage.correlate(I, Sy, mode='nearest')
    nr, nc = I.shape
    L_mag = np.round(np.sqrt(Dx**2+Dy**2),2)
    L_ang = np.round(np.rad2deg(np.arctan(Dy/(Dx+np.finfo(float).eps))), 2)
    L_ang = np.where(Dx < 0, L_ang+180, L_ang)
    L_ang = np.where(L_ang < 0, 360+L_ang, L_ang)
    L_ang = np.round(L_ang / 45)*45
    khist = np.empty(shape=(len(keypts),8))
    l = 0
    print(L_mag)
    for i in range(nr):
        for j in range(nc):
            if (i, j) in keypts:
                obin = np.zeros(shape=(8))
                for r in range(-2, 3):
                    for c in range(-2,3):
                        if r+i < 0 or c+j < 0 or r+i >= nr or c+j >= nc:
                            continue
                        else:
                            obin[int(L_ang[r+i, c+j]//45)] += L_mag[r+i, c+j]
                khist[l] = obin
                l+=1
                if l >= len(keypts):
                    return khist
    return -1
    

I = excelRead("image.xlsx")
g1=gaussianfilter()
diff = getDiffGauss(I, g1)
points = getExtremaSingle(diff, 0)
sPoints = saveKeypoints(I, points, diff)
R = getPointR(I, sPoints, diff[0])
hist = orientationAssg(I, g1, sPoints)