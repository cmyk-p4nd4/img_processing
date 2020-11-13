import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from cv2 import cv2

def mat2gray(A):
    A = np.double(A)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return normalized

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

A = plt.imread('sampleImg/test_color.jpg')
A_gray = rgb2gray(A)
nc, nr = A_gray.shape
plt.imsave("test_gray.jpg", A_gray)
Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Sobel operator
Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # Sobel operator
Ax = signal.correlate2d(A_gray, Sx, mode='same')
Ay = signal.correlate2d(A_gray, Sy, mode='same')
plt.figure(num="Image")
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
plt.subplot(2,2,1); plt.imshow(A); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(A_gray, cmap='gray'); plt.axis('off')
plt.subplot(2,2,3); plt.imshow(np.uint8(mat2gray(Ax)*255), cmap='gray')
plt.axis('off')
plt.subplot(2,2,4); plt.imshow(np.uint8(mat2gray(Ay)*255), cmap='gray')
plt.axis('off')
plt.show()
print(A.shape,A_gray.shape)