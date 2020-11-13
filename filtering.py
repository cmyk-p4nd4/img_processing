import numpy as np
from scipy.ndimage import correlate1d,correlate

x = np.full((5,7),100)
x[1,2:6] = 200
x[2,1::4]=200
x[2,2:5]=160
x[3,1:6]=200
dummy = np.full((7,9),100)
dummy[1:6,1:8]=x

Sx = np.arange(-1,2,1)
Sy = Sx.reshape(3,1)
Sxx = np.tile(Sx,(3,1))
Sxx[1,:]= 2*Sx
Syy = np.transpose(Sxx)

def HxFinder(ix,ixy,iy):
    return np.array([[ix,ixy],[ixy,iy]]), (ix*iy-ixy*ixy-0.05*(ix+iy)*(ix+iy))

Cx = correlate1d(dummy,Sx,output=np.float64)
Cy = correlate(dummy,Sy,output=np.float64)

I2x = Cx*Cx
Ixy = Cx*Cy
I2y = Cy*Cy
I2x = I2x[1:6,1:8]
I2y = I2y[1:6,1:8]
Ixy = Ixy[1:6,1:8]

print("I2x:\n",I2x)
print()
print("I2y:\n",I2y)
print()
print("Ixy:\n",Ixy)
print()
for m in range(5):
    for n in range(7):
        print(m,n)
        print(HxFinder(I2x[m,n],Ixy[m,n],I2y[m,n]))
        print()