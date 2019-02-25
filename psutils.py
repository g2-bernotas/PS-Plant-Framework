import numpy as np
import math as m
from LensSensorFOVCalc import LensSensorFOVCalc
import matplotlib.pyplot as plt
from retrying import retry
from PSConfig import PSConfig
import os
import cv2
import glob

class Lens:
    
    def __init__(self, focalLen, sensorSize):
        self.focalLen=focalLen
        self.sensorSize=sensorSize

def PGLens(focalLength):
    return Lens(focalLength, [8.8, 8.8]) 


def cart2sph2(x, y, z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))
    az = np.arctan2(y,x)
    return r, elev, az

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))
    az = m.atan2(y,x)
    return r, elev, az
    
def getLSCoordsForNLS(N_LS, phi):
    r=1
    
    angle=m.radians(360)/N_LS
    theta=[]
    for i in range(0,N_LS):
        theta.append(i*angle)
        
    phi=m.radians(phi);

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    z= np.ones(np.shape(x))*z
    
    LS=np.transpose(np.array([x, y, z]))
    
    return LS
    
def test():
    LS=getLSCoordsForNLS(4,20);
    print(LS)
    
def getLSCoordsForAzZe(az, ze):
    phi =np.radians(ze)
    theta= np.radians(az)
    r=1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    z= np.ones(np.shape(x))*z
    
    LS=np.transpose(np.array([x, y, z]))
    return LS
def showMap(m):
    plt.figure()
    plt.imshow(m, cmap='gray')
    plt.show();


def generateWeightMap(LS, shape, lens, roi):

    H=LS[2]
    w,h = LensSensorFOVCalc(lens.focalLen, lens.sensorSize, H)
    
    px_x = w / shape[1];
    px_y = h / shape[0] 
    
    LS[0]=LS[0] /px_x
    LS[1]=LS[1] /px_y

    LS_rw=[int(shape[0]/2) + LS[0], int(shape[1]/2)+LS[1]]

    LS_rw[0]=LS_rw[0]*px_x
    LS_rw[1]=LS_rw[1]*px_y
    
    X,Y = np.meshgrid(range(0,shape[1]), range(0,shape[0]));
    
    X=X*px_x
    Y=Y*px_y
   
    coords=np.vstack((X.flatten(), Y.flatten()))

    r=LS_rw[0] - coords[0]
    s=LS_rw[1] - coords[1]
    
    m=np.sqrt(r**2 + s**2)

    m=np.sqrt(m**2 + H**2)
    m=m**2
   
    m=m/m.min()

    m=np.reshape(m,shape)
    

    if np.size(roi) > 0:
        x1=roi[0]
        x2=roi[1]
        y1=roi[2]
        y2=roi[3]
        
        m=m[y1:y2, x1:x2]
        
    return m
    
def generateWeightMaps(LS, shape, lens, roi):
    maps=[]
    for ls in LS:
        map=generateWeightMap(ls, shape, lens, roi)
        if np.size(maps) ==0:
            maps=map.flatten().copy()
        else:
            maps = np.vstack((maps, map.flatten().copy()))
        
    return maps
    

'''
Calculates the shadow images (min val per pixel/max val per pixel)
for a given set of images (in the form of eg Nx2048x2048)

This boosts the shadowed regions, hopefully aiding detecting and overlapping of overlapping leafs
'''
def shadowImage(imgs):
    minImg=imgs.min(0)
    maxImg=imgs.max(0)
    shadowIm=(minImg/(maxImg+np.spacing(1))) 
    return shadowIm, minImg, maxImg

def shadowImageFromPth(pth, ext='im?.bmp'):
    filelist = glob.glob(os.path.join(pth, ext));
    imgs = np.array([np.array(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in filelist])
    
    return shadowImage(imgs)

@retry(stop_max_attempt_number=5, wait_fixed=3000) 
def loadIms(pth, N, flatten=True):
    imgs=[]
    for j in np.arange(0,N):
        im=cv2.imread(os.path.join(pth,'im{}.bmp'.format(j)), cv2.IMREAD_GRAYSCALE)
        imsz=im.shape
        if flatten:
            imgs.append(im.flatten())
        else:
            imgs.append(im)
    return imgs, imsz
    
@retry(stop_max_attempt_number=5, wait_fixed=3000) 
def loadProperties(pth, filename):
    psconfig=PSConfig.fromPropertiesFile(os.path.join(pth, filename))
    return psconfig
    
@retry(stop_max_attempt_number=5, wait_fixed=3000)
def saveData(pth, z, A, snIm, shadowIm):
    np.savez_compressed(pth, z=z, A=A, snIm=snIm, shadowIm=shadowIm)

def getLSFromRunMode(runMode, LS_VIS, LS_NIR, LS_VISNIR):
    LS=[]
    runMode=runMode.strip(os.sep)
    
    if runMode == 'VIS':
        LS=LS_VIS
    elif runMode == 'NIR':
        LS=LS_NIR
    elif runMode == 'VISNIR':
        LS=LS_VISNIR
    else:
        raise HaltException('Invalid runMode specified. Must be VIS, NIR or VISNIR: ' + runMode)
        
    return LS