import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math

def extractImg(class_no=1, image_no=1, type="training"):
    type = "training" if (type == "training") else "test"
    path = "sampleImg/"+str(class_no) + "/" + str(class_no) + str(image_no) + '_'+type.capitalize()+".bmp"
    return plt.imread(path)

def rgb2gray(rgb):
    if (rgb.ndim != 3):
        return rgb
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def SobelTransform(I):
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = signal.correlate2d(I, Sx, mode='same')
    Iy = signal.correlate2d(I, Sy, mode='same')
    I_mag=np.sqrt(Ix**2 + Iy**2)                            # I_mag: gradient magnitude
    
    # Gradient orientation
    nr, nc = I.shape
    Ipr = np.zeros(shape=(nr, nc))
    I_angle = np.zeros(shape=(nr, nc))
    for j in range(nr):
        for i in range(nc):
            if abs(Ix[j, i]) <= 0.0001 and abs(Iy[j, i]) <= 0.0001: # Both Ix and Iy are close to zero
                I_angle[j, i] = 0.00
            else:
                Ipr[j, i] = math.atan(Iy[j,i]/(Ix[j,i]+np.finfo(float).eps))      # Compute the angle in radians
                I_angle[j, i] = Ipr[j, i]*180/math.pi       # Compute the angle in degrees
                if Ix[j, i] < 0:                            # If Ix is negative, 180 degrees added
                    I_angle[j, i] = 180+I_angle[j, i]
                if I_angle[j, i] < 0:                       # If the angle is negative, 360 degrees added
                    I_angle[j, i] = 360+I_angle[j, i]
    return I_mag, I_angle                


def HoG1(Im, Ip, nbin):
    # Unsigned gradient (i.e. 0-180)
    # Im: magnitude of the image block
    # Ip: orientation of the image block 
    ghist = np.zeros(shape=(1,nbin))
    [nr1, nc1] = Im.shape
    interval = np.round(180/nbin, 0)
    for i in range(nr1):
        for j in range(nc1):
            if Ip[i, j] > 180:
                Ip[i, j] = abs(Ip[i, j] - 360)
            index = int(np.int(Ip[i, j]/interval))
            if index >= nbin:
                index = index - 1
            ghist[0, index] += np.square(Im[i,j]) #Stack 
    return ghist

def Histogram_Normalization(ihist):
    # Normailize input histogram ihist to a unit histogram
    total_sum = np.sum(ihist)
    nhist = ihist / total_sum
    return nhist

nr_b, nc_b = 3,3
nbin = 9

def getFeatureVec(I, I_mag, I_angle, nr_b, nc_b):
    # Use 2x2 blocks
    nr, nc = I.shape
    nbin = 9
    nr_size = int(nr/nr_b)
    nc_size = int(nc/nc_b)
    Image_HoG = np.zeros(shape=(1, nbin*nr_b*nc_b))
    for i in range(nr_b):
        for j in range(nc_b):
            I_mag_block = I_mag[i*nr_size: (i+1)*nr_size, j*nc_size: (j+1)*nc_size]
            I_angle_block = I_angle[i*nr_size: (i+1)*nr_size, j*nc_size: (j+1)*nc_size]
            # HoG1 is a function which create the HoG histogram
            gh = HoG1(I_mag_block, I_angle_block, nbin)
            # Histogram_Normalization is a function to normalize the input histogram gh
            ngh = Histogram_Normalization(gh)
            pos = j*nbin+i*nc_b*nbin
            Image_HoG[:, pos:pos+nbin] = ngh
    return Image_HoG

h1 = np.zeros(shape=(25, nbin*nr_b*nc_b)) # training
h2 = np.zeros(shape=(25, nbin*nr_b*nc_b)) # test
d1 = np.zeros (shape=(25,25))
d2 = d1.copy()
chi = d2.copy()

for i in range(1,6):
    for j in range(1,6):
        I = rgb2gray(extractImg(i,j,"training"))
        h1[(j-1)+(i-1)*5] = getFeatureVec(I,SobelTransform(I)[0],SobelTransform(I)[1], nr_b, nc_b)
        I = rgb2gray(extractImg(i,j,"test"))
        h2[(j-1)+(i-1)*5] = getFeatureVec(I,SobelTransform(I)[0],SobelTransform(I)[1], nr_b, nc_b)

for i in range(25):
    for j in range(25):
        d1[i, j] = np.around(np.sum(np.abs(h2[i, :]-h1[j, :])),4)
        d2[i, j] = np.around(np.sum(np.square(np.abs(h2[i, :]-h1[j, :]))), 4)
        chi[i, j] =  np.around(np.sum(np.square(np.abs(h2[i, :]-h1[j, :])) / (h2[i, :]+h1[j, :]+np.finfo(float).eps)), 4)

# 11 12 13 ... 53 54 55

d1_min = np.argmin(d1,axis=1)
d2_min = np.argmin(d2,axis=1)
chi_min = np.argmin(chi,axis=1)

acc = np.zeros(shape=(25))
for i in range(5):
    for j in range(5):
        acc[j+5*i] = 5*i

print(f'Matching Type: \n{d1_min+1}')
print(f'Matching Type: \n{d2_min+1}')
print(f'Matching Type: \n{chi_min+1}')

d1_same = d1_min - acc
d2_same = d2_min - acc
chi_same = chi_min - acc
print(np.sum(x in range(0,5) for x in d1_same) / d1_same.size)
print(np.sum(x in range(0,5) for x in d2_same) / d2_same.size)
print(np.sum(x in range(0,5) for x in chi_same) / chi_same.size)